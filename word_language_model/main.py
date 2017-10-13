import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import data
import model

import sys
import os
sys.path.append("../tuner_utils")
from yellowfin import YFOptimizer
from debug_plot import plot_func
import logging

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--logdir', type=str, default='.',
                    help='folder for the logs')
parser.add_argument('--opt_method', type=str, default='YF',
                    help='select the optimizer you are using')
parser.add_argument('--lr_thresh', type=float, default=1.0)
args = parser.parse_args()

if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)
logging.basicConfig(filename=args.logdir + "/num.log", level=logging.DEBUG)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(opt, loss_list,\
    local_curv_list,\
    max_curv_list,\
    min_curv_list,\
    lr_list,\
    lr_t_list,\
    mu_t_list,\
    dr_list,\
    mu_list,\
    dist_list,\
    grad_var_list,\
    lr_g_norm_list,\
    lr_g_norm_squared_list,\
    move_lr_g_norm_list,\
    move_lr_g_norm_squared_list,\
    lr_grad_norm_clamp_act_list,\
    fast_view_act_list):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    train_loss_list = []
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        # for group in optimizer._optimizer.param_groups:
        #     print group['lr'], group['momentum']


        loss_list.append(loss.data[0])
        if args.opt_method == 'YF':        
            local_curv_list.append(opt._global_state['grad_norm_squared'] )
            max_curv_list.append(opt._h_max)
            min_curv_list.append(opt._h_min)
            lr_list.append(opt._lr)
            mu_list.append(opt._mu)
            dr_list.append((opt._h_max + 1e-6) / (opt._h_min + 1e-6))
            dist_list.append(opt._dist_to_opt)
            grad_var_list.append(opt._grad_var)

            lr_g_norm_list.append(opt._lr * np.sqrt(opt._global_state['grad_norm_squared'] ) )
            lr_g_norm_squared_list.append(opt._lr * opt._global_state['grad_norm_squared'] )
            move_lr_g_norm_list.append(opt._optimizer.param_groups[0]["lr"] * np.sqrt(opt._global_state['grad_norm_squared'] ) )
            move_lr_g_norm_squared_list.append(opt._optimizer.param_groups[0]["lr"] * opt._global_state['grad_norm_squared'] )

            lr_t_list.append(opt._lr_t)
            mu_t_list.append(opt._mu_t)


        total_loss += loss.data
        train_loss_list.append(loss.data[0] )

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return train_loss_list,\
    loss_list,\
    local_curv_list,\
    max_curv_list,\
    min_curv_list,\
    lr_list,\
    lr_t_list,\
    mu_t_list,\
    dr_list,\
    mu_list,\
    dist_list,\
    grad_var_list,\
    lr_g_norm_list,\
    lr_g_norm_squared_list,\
    move_lr_g_norm_list,\
    move_lr_g_norm_squared_list,\
    lr_grad_norm_clamp_act_list,\
    fast_view_act_list

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    train_loss_list = []
    val_loss_list = []
    lr_list = []
    mu_list = []

    loss_list = []
    local_curv_list = []
    max_curv_list = []
    min_curv_list = []
    lr_g_norm_list = []
    lr_list = []
    lr_t_list = []
    mu_t_list = []
    dr_list = []
    mu_list = []
    dist_list = []
    grad_var_list = []
    
    lr_g_norm_list = []
    lr_g_norm_squared_list = []
    
    move_lr_g_norm_list = []
    move_lr_g_norm_squared_list = []
    
    lr_grad_norm_clamp_act_list = []
    fast_view_act_list = []

    if args.opt_method == "SGD":
        print("using SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.0)
    elif args.opt_method == "momSGD":
        print("using mom SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    elif args.opt_method == "YF":
        print("using YF")
        optimizer = YFOptimizer(model.parameters(), lr=0.0001, mu=0.0, exploding_grad_elim_fac=args.lr_thresh)
    elif args.opt_method == "Adagrad":
        print("using Adagrad")
        optimizer = torch.optim.Adagrad(model.parameters(), lr)
    elif args.opt_method == "Adam":
        print("using Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        #train_loss = train()
        train_loss, \
        loss_list, \
        local_curv_list,\
        max_curv_list,\
        min_curv_list,\
        lr_list,\
        lr_t_list,\
        mu_t_list,\
        dr_list,\
        mu_list,\
        dist_list,\
        grad_var_list,\
        lr_g_norm_list,\
        lr_g_norm_squared_list,\
        move_lr_g_norm_list,\
        move_lr_g_norm_squared_list,\
        lr_grad_norm_clamp_act_list,\
        fast_view_act_list = \
          train(optimizer,
          loss_list, \
          local_curv_list,\
          max_curv_list,\
          min_curv_list,\
          lr_list,\
          lr_t_list,\
          mu_t_list,\
          dr_list,\
          mu_list,\
          dist_list,\
          grad_var_list,\
          lr_g_norm_list,\
          lr_g_norm_squared_list,\
          move_lr_g_norm_list,\
          move_lr_g_norm_squared_list,\
          lr_grad_norm_clamp_act_list,\
          fast_view_act_list)

        train_loss_list += train_loss
        val_loss = evaluate(val_data)
        val_loss_list.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch,
                                         (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.logdir + "/" + args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            if args.opt_method == "YF":
                optimizer.set_lr_factor(optimizer.get_lr_factor() / 4.0)
            else:
                for group in optimizer.param_groups:
                    group['lr'] /= 4.0
        #if args.opt_method == "YF":
        #    mu_list.append(optimizer._mu)
        #    lr_list.append(optimizer._lr)
        

        if args.opt_method=='YF' and (epoch % 5 == 0 or epoch == 5):
            plot_func(log_dir=args.logdir, iter_id=epoch, loss_list=loss_list,
                local_curv_list=local_curv_list, max_curv_list=max_curv_list,
                min_curv_list=min_curv_list, lr_g_norm_list=lr_g_norm_list, lr_g_norm_squared_list=lr_g_norm_squared_list,
                lr_list=lr_list, lr_t_list=lr_t_list, dr_list=dr_list,
                mu_list=mu_list, mu_t_list=mu_t_list,
                grad_avg_norm_list=[],
                dist_list=dist_list, grad_var_list=grad_var_list,
                move_lr_g_norm_list=move_lr_g_norm_list, move_lr_g_norm_squared_list=move_lr_g_norm_squared_list,
                fast_view_act_list=fast_view_act_list, lr_grad_norm_clamp_act_list=lr_grad_norm_clamp_act_list)


        with open(args.logdir+"/loss.txt", "wb") as f:
            np.savetxt(f, np.array(train_loss_list) )
        with open(args.logdir+"/val_loss.txt", "wb") as f:
            np.savetxt(f, np.array(val_loss_list) )
        with open(args.logdir+"/lr.txt", "wb") as f:
            np.savetxt(f, np.array(lr_list) )
        with open(args.logdir+"/mu.txt", "wb") as f:
            np.savetxt(f, np.array(mu_list) )


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
# with open(args.save, 'rb') as f:
#    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
