import math
# for torch optim sgd
import numpy as np
import torch

class YFOptimizer(object):
  def __init__(self, var_list, lr=0.1, mu=0.0, clip_thresh=None, weight_decay=0.0,
    beta=0.999, curv_win_width=20, zero_debias=True, delta_mu=0.0):
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
      delta_mu: for extensions. Not necessary in the basic use. (TODO)
    Other features:
      If you want to manually control the learning rates, self.lr_factor is
      an interface to the outside, it is an multiplier for the internal learning rate
      in YellowFin. It is helpful when you want to do additional hand tuning
      or some decaying scheme to the tuned learning rate in YellowFin. 
      Example on using lr_factor can be found here:
      (TODO)
    '''
    self._lr = lr
    self._mu = mu
    # we convert var_list from generator to list so that
    # it can be used for multiple times
    self._var_list = list(var_list)
    self._clip_thresh = clip_thresh
    self._beta = beta
    self._curv_win_width = curv_win_width
    self._zero_debias = zero_debias
    self._optimizer = torch.optim.SGD(self._var_list, lr=self._lr, 
      momentum=self._mu, weight_decay=weight_decay)
    self._iter = 0
    # global states are the statistics
    self._global_state = {}

    # for decaying learning rate and etc.
    self._lr_factor = 1.0
    pass


  def set_lr_factor(self, factor):
    self._lr_factor = factor
    return


  def get_lr_factor(self):
    return self._lr_factor


  def zero_grad(self):
    self._optimizer.zero_grad()


  def zero_debias_factor(self):
    return 1.0 - self._beta ** (self._iter + 1)


  def curvature_range(self):
    global_state = self._global_state
    if self._iter == 0:
      global_state["curv_win"] = torch.FloatTensor(self._curv_win_width, 1).zero_()
    curv_win = global_state["curv_win"]
    grad_norm_squared = self._global_state["grad_norm_squared"]
    curv_win[self._iter % self._curv_win_width] = grad_norm_squared
    valid_end = min(self._curv_win_width, self._iter + 1)
    beta = self._beta
    if self._iter == 0:
      global_state["h_min_avg"] = 0.0
      global_state["h_max_avg"] = 0.0
      self._h_min = 0.0
      self._h_max = 0.0
    global_state["h_min_avg"] = \
      global_state["h_min_avg"] * beta + (1 - beta) * torch.min(curv_win[:valid_end] )
    global_state["h_max_avg"] = \
      global_state["h_max_avg"] * beta + (1 - beta) * torch.max(curv_win[:valid_end] )
    if self._zero_debias:
      debias_factor = self.zero_debias_factor()
      self._h_min = global_state["h_min_avg"] / debias_factor
      self._h_max = global_state["h_max_avg"] / debias_factor
    else:
      self._h_min = global_state["h_min_avg"]
      self._h_max = global_state["h_max_avg"]
    return


  def grad_variance(self):
    global_state = self._global_state
    beta = self._beta
    self._grad_var = np.array(0.0, dtype=np.float32)
    for group in self._optimizer.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        state = self._optimizer.state[p]

        if self._iter == 0:
          state["grad_avg"] = grad.new().resize_as_(grad).zero_()
          state["grad_avg_squared"] = 0.0
        state["grad_avg"].mul_(beta).add_(1 - beta, grad)
        self._grad_var += torch.sum(state["grad_avg"] * state["grad_avg"] )
        
    if self._zero_debias:
      debias_factor = self.zero_debias_factor()
    else:
      debias_factor = 1.0

    self._grad_var /= -(debias_factor**2)
    self._grad_var += global_state['grad_norm_squared_avg'] / debias_factor
    return


  def dist_to_opt(self):
    global_state = self._global_state
    beta = self._beta
    if self._iter == 0:
      global_state["grad_norm_avg"] = 0.0
      global_state["dist_to_opt_avg"] = 0.0
    global_state["grad_norm_avg"] = \
      global_state["grad_norm_avg"] * beta + (1 - beta) * math.sqrt(global_state["grad_norm_squared"] )
    global_state["dist_to_opt_avg"] = \
      global_state["dist_to_opt_avg"] * beta \
      + (1 - beta) * global_state["grad_norm_avg"] / global_state['grad_norm_squared_avg']
    if self._zero_debias:
      debias_factor = self.zero_debias_factor()
      self._dist_to_opt = global_state["dist_to_opt_avg"] / debias_factor
    else:
      self._dist_to_opt = global_state["dist_to_opt_avg"]
    return


  def after_apply(self):
    # compute running average of gradient and norm of gradient
    beta = self._beta
    global_state = self._global_state
    if self._iter == 0:
      global_state["grad_norm_squared_avg"] = 0.0

    global_state["grad_norm_squared"] = 0.0
    for group in self._optimizer.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        # global_state['grad_norm_squared'] += torch.dot(grad, grad)
        global_state['grad_norm_squared'] += torch.sum(grad * grad)
        
    global_state['grad_norm_squared_avg'] = \
      global_state['grad_norm_squared_avg'] * beta + (1 - beta) * global_state['grad_norm_squared']
    # global_state['grad_norm_squared_avg'].mul_(beta).add_(1 - beta, global_state['grad_norm_squared'] )
        
    self.curvature_range()
    self.grad_variance()
    self.dist_to_opt()
    if self._iter > 0:
      self.get_mu()    
      self.get_lr()
      self._lr = beta * self._lr + (1 - beta) * self._lr_t
      self._mu = beta * self._mu + (1 - beta) * self._mu_t
    return


  def get_lr(self):
    self._lr_t = (1.0 - math.sqrt(self._mu_t) )**2 / self._h_min
    return


  def get_mu(self):
    coef = [-1.0, 3.0, 0.0, 1.0]
    coef[2] = -(3 + self._dist_to_opt**2 * self._h_min**2 / 2 / self._grad_var)
    roots = np.roots(coef)
    root = roots[np.logical_and(np.logical_and(np.real(roots) > 0.0, 
      np.real(roots) < 1.0), np.imag(roots) < 1e-5) ]
    assert root.size == 1
    dr = self._h_max / self._h_min
    self._mu_t = max(np.real(root)[0]**2, ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2 )
    return 


  def update_hyper_param(self):
    for group in self._optimizer.param_groups:
      group['momentum'] = self._mu
      group['lr'] = self._lr * self._lr_factor
    return


  def step(self):
    # add weight decay
    for group in self._optimizer.param_groups:
      for p in group['params']:
        if p.grad is None:
            continue
        grad = p.grad.data

        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)
    
    if self._clip_thresh != None:
      torch.nn.utils.clip_grad_norm(self._var_list, self._clip_thresh)

    # apply update
    self._optimizer.step()

    # after appply
    self.after_apply()

    # update learning rate and momentum
    self.update_hyper_param()

    self._iter += 1
    return 

