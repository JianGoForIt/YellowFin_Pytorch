import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
from layers_torch import *
import time
from char_data_iterator import TextIterator
import numpy
import math
import random
import sys 
sys.path.append("../../tuner_utils")
from yellowfin import YFOptimizer
from debug_plot import plot_func
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='PyTorch Olexa model')
parser.add_argument("--log_dir", type=str, default="./results")
parser.add_argument("--fast_bound_const", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

print("using seed ", args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    #if not args.use_cuda:
    #    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #else:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

import numpy as np
import logging
#log_dir = "./results/lr_grad_clamp_1.0_h_max_log_h_min_log_pure_fast_view_win_200"
log_dir = args.log_dir
fast_bound_const = args.fast_bound_const
if not os.path.isdir(log_dir):
   os.makedirs(log_dir)
logging.basicConfig(filename=log_dir + "/num.log",level=logging.DEBUG)

length = 784
input_size = 1
rnn_dim = 1000
num_layers = 2
num_classes = 2
batch_size = 32
valid_batch_size = 32
num_epochs = 15 
lr = 0.0001
n_words=2
maxlen=785
dataset='train.txt'
valid_dataset='test.txt'
dictionary='dict_bin_mnist.npz'
sequence_length = 28
truncate_length = 10
attn_every_k = 10


if not os.path.isdir("mnist_logs"):
    os.mkdir("mnist_logs")
file_name = 'mnist_logs/mnist_trun_len_' + str(truncate_length) + '_full_attn' + str(random.randint(1000,9999)) + '.txt'


train = TextIterator(dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen, 
                         minlen=length)
valid = TextIterator(valid_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen, 
                         minlen=length)


rnn = RNN_LSTM(input_size, rnn_dim, num_layers, num_classes)
rnn.cuda()

criterion = nn.CrossEntropyLoss()

#opt = torch.optim.Adam(rnn.parameters(), lr=lr)
opt = YFOptimizer(rnn.parameters() )


def evaluate_valid(valid):
    valid_loss = []
    acc        = 0.0
    N          = 0
    for x in valid:
        x = numpy.asarray(x, dtype=numpy.float32)
        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        y = torch.cat(( x[:, 1:, :], torch.zeros([x.size()[0], 1, input_size])), 1)
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()
        opt.zero_grad()
        outputs = rnn(images)
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1])
        loss = criterion(outputs_reshp, labels_reshp)
        valid_loss.append(784 * float(loss.data[0]))
        acc += outputs_reshp.max(1)[1].eq(labels_reshp).long().data.sum()
        N   += len(labels_reshp)
    acc /= float(N)
    log_line = 'Epoch [%d/%d], truncate_length: %d,  average Loss: %f, avg Acc %f, validation ' %(epoch, num_epochs, truncate_length, numpy.asarray(valid_loss).mean(), acc)
    print  (log_line)
    with open(file_name, 'a') as f:
        f.write(log_line)

# def plot_func(log_dir, iter_id, loss_list, local_curv_list, max_curv_list, min_curv_list,
#              lr_g_norm_list, lr_list, dr_list, mu_list, grad_avg_norm_list,
#              dist_list, grad_var_list, move_list, recover_move_list):
#     def running_mean(x, N):
#         cumsum = np.cumsum(np.insert(x, 0, 0)) 
#         return (cumsum[N:] - cumsum[:-N]) / N 
#     plt.figure()
#     plt.semilogy(loss_list, '.', alpha=0.2, label="Loss")
#     plt.semilogy(running_mean(loss_list,100), label="Average Loss")
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid()
#     ax = plt.subplot(111)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True)
#     plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
#     plt.close()

#     plt.figure()
#     plt.semilogy(local_curv_list, label="local curvature")
#     plt.semilogy(max_curv_list, label="max curv in win")
#     plt.semilogy(min_curv_list, label="min curv in win")
# #         plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
#     plt.semilogy(lr_g_norm_list, label="lr * grad norm")
#     plt.semilogy(move_list, label="move")
#     plt.semilogy(recover_move_list, label="recover move")
#     plt.title("On local curvature")
#     plt.grid()
#     ax = plt.subplot(111)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=2, fancybox=True, shadow=True)
#     plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
#     plt.close()

#     plt.figure()
#     plt.semilogy(lr_list, label="lr min")
#     plt.semilogy(dr_list, label="dynamic range")
#     plt.semilogy(mu_list, label="mu")
#     plt.semilogy(grad_avg_norm_list, label="Grad avg norm")
#     plt.semilogy(dist_list, label="Est dist from opt")
#     plt.semilogy(grad_var_list, label="Grad variance")
#     plt.title('LR='+str(lr_list[-1])+' mu='+str(mu_list[-1] ) )
#     plt.grid()
#     plt.legend(loc="upper right")
#     plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
#     plt.close()

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

clip_thresh_list = []

unclip_g_norm_list = []

for epoch in range(num_epochs):
    i = 0
    for x in train:
        t = -time.time()
        #import ipdb; ipdb.set_trace()
        
        #if not (i == 0 and epoch == 0):       
            #fast_view_act_list.append(opt._fast_bound_const/( (math.sqrt(opt._global_state["grad_norm_squared"] ) + math.sqrt(opt._h_min) )**2 + 1e-6))
            #lr_grad_norm_clamp_act_list.append(opt._lr_grad_norm_thresh / (math.sqrt(opt._global_state["grad_norm_squared"] ) + 1e-6) )

        x = numpy.asarray(x, dtype=numpy.float32)
        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        y = torch.cat(( x[:, 1:, :], torch.zeros([x.size()[0], 1, input_size])), 1) 
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()
        opt.zero_grad()
        outputs = rnn(images)
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1])
        loss = criterion(outputs_reshp, labels_reshp)
        loss.backward()
        opt.step()

        # For plotting
        #unclip_g_norm_list.append(opt._unclip_grad_norm)

        t += time.time()
	
        loss_list.append(loss.data[0]) 
        
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
        clip_thresh_list.append(opt._exploding_grad_clip_target_value**2)


        if not (i == 0 and epoch == 0):
            lr_t_list.append(opt._lr_t)
            mu_t_list.append(opt._mu_t)

        if (i+1) % 10 == 0:
            log_line = 'Epoch [%d/%d], Step %d, Loss: %f, batch_time: %f \n' %(epoch, num_epochs, i+1, 784 * loss.data[0], t)
            print (log_line)
            with open(file_name, 'a') as f:
                f.write(log_line)
        if (i + 1) % 500 == 0:
           # plot_func(log_dir, epoch * 500 + i, loss_list, local_curv_list, max_curv_list, min_curv_list,
           #   lr_g_norm_list, lr_list, dr_list, mu_list, grad_avg_norm_list=[],
           #   dist_list=dist_list, grad_var_list=grad_var_list, move_list=[], recover_move_list=[])
            plot_func(log_dir=log_dir, iter_id=i + epoch * 1500, loss_list=loss_list, 
                local_curv_list=local_curv_list, max_curv_list=max_curv_list, 
                min_curv_list=min_curv_list, lr_g_norm_list=lr_g_norm_list, lr_g_norm_squared_list=lr_g_norm_squared_list, 
                lr_list=lr_list, lr_t_list=lr_t_list, dr_list=dr_list, 
                mu_list=mu_list, mu_t_list=mu_t_list,
                grad_avg_norm_list=[],
                dist_list=dist_list, grad_var_list=grad_var_list, 
                move_lr_g_norm_list=move_lr_g_norm_list, move_lr_g_norm_squared_list=move_lr_g_norm_squared_list,
                fast_view_act_list=fast_view_act_list, lr_grad_norm_clamp_act_list=lr_grad_norm_clamp_act_list, clip_thresh_list=clip_thresh_list)
            print "figure plotted"


            with open(log_dir + "/mu_list.txt", "w") as f:
                np.savetxt(f, mu_list)
                
            with open(log_dir + "/lr_list.txt", "w") as f:
                np.savetxt(f, lr_list)

           
            with open(log_dir + "/mu_t.txt", "w") as f:
                np.savetxt(f, mu_t_list)
            with open(log_dir + "/lr_t.txt", "w") as f:
                np.savetxt(f, lr_t_list)

            with open(log_dir + "/clip_thresh.txt", "w") as f:
                clip_thresh_array = np.array( [clip_thresh_list, local_curv_list] ).T
                np.savetxt(f, clip_thresh_array)

            with open(log_dir + "/h_max.txt", "w") as f:
		np.savetxt(f, np.array(max_curv_list))

            with open(log_dir + "/unclip_g_norm.txt", "w") as f:
	    	np.savetxt(f, np.array(unclip_g_norm_list))

        #if (i + 1) % 1000 == 0:
        #    evaluate_valid(valid) 

        i += 1
      
    # evaluate per epoch
    print '--- Epoch finished ----'
    evaluate_valid(valid)


