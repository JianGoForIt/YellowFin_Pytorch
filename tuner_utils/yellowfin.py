import math
# for torch optim sgd
import torch

class YFOptimizer(object):
  def __init__(self, lr=0.1, mu=0.0, clip_thresh=None, beta=0.999, curv_win_width=20,
    mu_update_interval=1, zero_debias=True, delta_mu=0.0):
    '''
    clip thresh is the threshold value on ||lr * gradient||
    delta_mu can be place holder/variable/python scalar. They are used for additional
    momentum in situations such as asynchronous-parallel training. The default is 0.0
    for basic usage of the optimizer.
    Args:
      lr: python scalar. The initial value of learning rate, we use 1.0 in our paper.
      mu: python scalar. The initial value of momentum, we use 0.0 in our paper.
      clip_thresh: python scalar. The cliping threshold for tf.clip_by_global_norm.
        if None, no clipping will be carried out. 
      beta: python scalar. The smoothing parameter for estimations.
      delta_mu: for extensions. Not necessary in the basic use.
    Other features:
      If you want to manually control the learning rates, self.lr_factor is
      an interface to the outside, it is an multiplier for the internal learning rate
      in YellowFin. It is helpful when you want to do additional hand tuning
      or some decaying scheme to the tuned learning rate in YellowFin. 
      Example on using lr_factor can be found here:
      https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L140
    '''