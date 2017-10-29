import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

softmax = nn.Softmax()
sigmoid = nn.Sigmoid()

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)

class AttentionWordRNN(nn.Module):
    
    
    def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, bidirectional= True, init_range=0.1, use_lstm=False):        
        
        super(AttentionWordRNN, self).__init__()
        
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.use_lstm = use_lstm       
 
        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            if use_lstm:
                print("inside using LSTM")
                self.word_gru = nn.LSTM(embed_size, word_gru_hidden, bidirectional= True)
            else:
                self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2* word_gru_hidden, 2*word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2*word_gru_hidden, 1))
        else:
            if use_lstm:
                self.word_gru = nn.LSTM(embed_size, word_gru_hidden, bidirectional= False)
            else:
                self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-init_range, init_range)
        self.weight_proj_word.data.uniform_(-init_range, init_range)

    def forward(self, embed, state_word):
        # embeddings
        embedded = self.lookup(embed)
        # word level gru
        output_word, state_word = self.word_gru(embedded, state_word)
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1,0))        
        return word_attn_vectors, state_word, word_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            if self.use_lstm == True:
                return [Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden)), Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden)) ] 
            else:
                return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        else:
            if self.use_lstm == True:
                return [Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden)), Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden)) ]
            else:
                return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))
        
class MixtureSoftmax(nn.Module):

    def __init__(self, batch_size, word_gru_hidden, feature_dim, n_classes, bidirectional=True):        
        
        super(MixtureSoftmax, self).__init__()
        
        # for feature only model 
        word_gru_hidden = 0
        # end
        
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.feature_dim = feature_dim
        
        if bidirectional == True:
            self.linear = nn.Linear(2 * 2 * word_gru_hidden + feature_dim, n_classes)
        else:
            self.linear = nn.Linear(2 * word_gru_hidden + feature_dim, n_classes)
        
    def forward(self, word_attention_vectors, features):
#         print 'word_attention_vectors', word_attention_vectors.size()
#         print word_attention_vectors[0].size(), word_attention_vectors[1].size(), features.size()
        
        # mixture_input = torch.cat((word_attention_vectors[0], word_attention_vectors[1], features), 1)
        mixture_input = features
        
#         mixture_input = torch.cat((word_attention_vectors[0], word_attention_vectors[1]), 1)
        # print 'mixture_input', mixture_input.size()
        final_map = self.linear(mixture_input)
        return final_map

def train_data(mini_batch, feature_batch, targets, word_attn_model, mix_softmax, optimizer, criterion, do_step=True, cuda=False, lstm=False):
    state_word = word_attn_model.init_hidden()
    optimizer.zero_grad()
   
    #print("inside cuda", cuda)
 
    if cuda:
        if lstm:
            state_word[0] = state_word[0].cuda()
            state_word[1] = state_word[1].cuda()
        else:
            state_word = state_word.cuda()
        mini_batch[0] = mini_batch[0].cuda()
        mini_batch[1] = mini_batch[1].cuda()
        feature_batch = feature_batch.cuda()
#     word_optimizer.zero_grad()
#     mix_optimizer.zero_grad()
#     print mini_batch[0].unsqueeze(1).size()
#     print mini_batch[1].unsqueeze(1).size()
    s1, state_word, _ = word_attn_model(mini_batch[0].transpose(0,1), state_word)
    s2, state_word, _ = word_attn_model(mini_batch[1].transpose(0,1), state_word)
    s = torch.cat((s1, s2),0)
    
    y_pred = mix_softmax(s, feature_batch)
#     y_pred = mix_softmax(feature_batch)
    if cuda:
        y_pred = y_pred.cuda()
        targets = targets.cuda() 

    # print y_pred.size(), targets.size(), "pred", y_pred, "targets", targets

    loss = criterion(y_pred, targets)
    loss.backward()
    
    if do_step:
        optimizer.step()
#     word_optimizer.step()
#     mix_optimizer.step()
    grad_norm = torch.nn.utils.clip_grad_norm(optimizer._var_list, 1.0 * 1e20)
    
    return loss.data[0], grad_norm

def get_predictions(mini_batch, feature_batch, word_attn_model, mix_softmax):
    state_word = word_attn_model.init_hidden()
    s1, state_word, _ = word_attn_model(mini_batch[0].transpose(0,1), state_word)
    s2, state_word, _ = word_attn_model(mini_batch[1].transpose(0,1), state_word)
    s = torch.cat((s1, s2),0)
    y_pred = mix_softmax(s, feature_batch)    
    return sigmoid(y_pred)

def pad_batch(mini_batch):
    mini_batch_size = len(mini_batch)
#     print mini_batch.shape
#     print mini_batch
    max_sent_len1 = int(np.max([len(x[0]) for x in mini_batch]))
    max_sent_len2 = int(np.max([len(x[1]) for x in mini_batch]))
#     print max_sent_len1, max_sent_len2
#     max_token_len = int(np.mean([len(val) for sublist in mini_batch for val in sublist]))
    main_matrix1 = np.zeros((mini_batch_size, max_sent_len1), dtype= np.int)
    main_matrix2 = np.zeros((mini_batch_size, max_sent_len2), dtype= np.int)
    for idx1, i in enumerate(mini_batch):
        for idx2, j in enumerate(i[0]):
            try:
                main_matrix1[i,j] = j
            except IndexError:
                pass
    for idx1, i in enumerate(mini_batch):
        for idx2, j in enumerate(i[1]):
            try:
                main_matrix2[i,j] = j
            except IndexError:
                pass
    main_matrix1_t = Variable(torch.from_numpy(main_matrix1))
    main_matrix2_t = Variable(torch.from_numpy(main_matrix2))
#     print main_matrix1_t.size()
#     print main_matrix2_t.size()
    return [main_matrix1_t, main_matrix2_t]
#     return [Variable(torch.cat((main_matrix1_t, main_matrix2_t), 0))

# def pad_batch(mini_batch):
# #     print mini_batch
# #     print type(mini_batch)
# #     print mini_batch.shape
# #     for i, _ in enumerate(mini_batch):
# #         print i, _
#     return [Variable(torch.from_numpy(np.asarray(_))) for _ in mini_batch[0]]


def test_accuracy_mini_batch(tokens, features, labels, word_attn, sent_attn):
    y_pred = get_predictions(tokens, features, word_attn, sent_attn)
    y_pred = torch.gt(y_pred, 0.5)
    correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
    labels = torch.gt(labels, 0.5)
    labels = np.ndarray.flatten(labels.data.cpu().numpy())

    num_correct = sum(correct == labels)

    return float(num_correct) / len(correct)

def test_accuracy_full_batch(tokens, features, mini_batch_size, word_attn, sent_attn, th=0.5):
    p = []
    l = []
    cnt = 0
    g = gen_minibatch1(tokens, features, mini_batch_size, False)
    for token, feature in g:
        if cnt % 100 == 0:
            print cnt
        cnt +=1
#         print token.size()
#         y_pred = get_predictions(token, word_attn, sent_attn)
#         print y_pred
        y_pred = get_predictions(token, feature, word_attn, sent_attn)
#         print y_pred
#         _, y_pred = torch.max(y_pred, 1)
#         y_pred = y_pred[:, 1]
#         print y_pred
        p.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
    p = [item for sublist in p for item in sublist]
    p = np.array(p)
    return p
    
def iterate_minibatches(inputs, features, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
#         if start_idx >= inputs.shape[0]: break
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
#         print inputs[excerpt]
        yield inputs[excerpt], features[excerpt], targets[excerpt]
    
def gen_minibatch(tokens, features, labels, mini_batch_size, shuffle= True):
    tokens = np.asarray(tokens)[np.where(labels!=0.5)[0]]
    features = np.asarray(features.todense())[np.where(labels!=0.5)[0]]
    labels = np.asarray(labels)[np.where(labels!=0.5)[0]]
#     print tokens.shape
#     print tokens[0]
    for token, feature, label in iterate_minibatches(tokens, features, labels, mini_batch_size, shuffle = shuffle):
#         print 'token', type(token)
#         print token
        token = [_ for _ in pad_batch(token)]
#         print len(token), token[0].size(), token[1].size()
        yield token, Variable(torch.from_numpy(feature)) , Variable(torch.FloatTensor(label), requires_grad= False)
    
def gen_minibatch1(tokens, features, mini_batch_size, shuffle= True):
    tokens = np.asarray(tokens)
    features = np.asarray(features.todense())
    print tokens.shape
    for token, feature, label in iterate_minibatches(tokens, features, features, mini_batch_size, shuffle = shuffle):
#         print token
#         token = pad_batch(token)
#         print token
        token = [_ for _ in pad_batch(token)]
        yield token, Variable(torch.from_numpy(feature))
        

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
