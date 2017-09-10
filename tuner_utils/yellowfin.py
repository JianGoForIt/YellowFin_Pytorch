import math
# for torch optim sgd
import numpy as np
import torch

class YFOptimizer(object):
  def __init__(self, var_list, lr=0.1, mu=0.0, clip_thresh=None, weight_decay=0.0,
    beta=0.999, curv_win_width=20, zero_debias=True, sparsity_debias=True, delta_mu=0.0, 
    auto_clip_fac=None, force_non_inc_step_after_iter=None):
    '''
    clip thresh is the threshold value on ||lr * gradient||
    delta_mu can be place holder/variable/python scalar. They are used for additional
    momentum in situations such as asynchronous-parallel training. The default is 0.0
    for basic usage of the optimizer.
    Args:
      lr: python scalar. The initial value of learning rate, we use 1.0 in our paper.
      mu: python scalar. The initial value of momentum, we use 0.0 in our paper.
      clip_thresh: python scalar. The manaully-set clipping threshold for tf.clip_by_global_norm.
        if None, the automatic clipping can be carried out. The automatic clipping 
        feature is parameterized by argument auto_clip_fac. The auto clip feature
        can be switched off with auto_clip_fac = None
      beta: python scalar. The smoothing parameter for estimations.
      sparsity_debias: gradient norm and curvature are biased to larger values when 
      calculated with sparse gradient. This is useful when the model is very sparse,
      e.g. LSTM with word embedding. For non-sparse CNN, turning it off could slightly
      accelerate the speed.
      delta_mu: for extensions. Not necessary in the basic use. 
      force_non_inc_step_after_iter: in some rare cases, it is necessary to force ||lr * gradient||
      to be non-increasing for stableness after some iterations. 
      Default is turning off this feature.
    Other features:
      If you want to manually control the learning rates, self.lr_factor is
      an interface to the outside, it is an multiplier for the internal learning rate
      in YellowFin. It is helpful when you want to do additional hand tuning
      or some decaying scheme to the tuned learning rate in YellowFin. 
      Example on using lr_factor can be found here:
      https://github.com/JianGoForIt/YellowFin_Pytorch/blob/master/pytorch-cifar/main.py#L109
    '''
    self._lr = lr
    self._mu = mu
    # we convert var_list from generator to list so that
    # it can be used for multiple times
    self._var_list = list(var_list)
    self._clip_thresh = clip_thresh
    self._auto_clip_fac = auto_clip_fac
    self._beta = beta
    self._curv_win_width = curv_win_width
    self._zero_debias = zero_debias
    self._sparsity_debias = sparsity_debias
    self._force_non_inc_step_after_iter = force_non_inc_step_after_iter
    self._optimizer = torch.optim.SGD(self._var_list, lr=self._lr, 
      momentum=self._mu, weight_decay=weight_decay)
    self._iter = 0
    # global states are the statistics
    self._global_state = {}

    # for decaying learning rate and etc.
    self._lr_factor = 1.0
    pass


  def state_dict(self):
    sgd_state_dict = self._optimizer.state_dict()
    global_state = self._global_state
    lr_factor = self._lr_factor
    iter = self._iter
    lr = self._lr
    mu = self._mu
    clip_thresh = self._clip_thresh
    beta = self._beta
    curv_win_width = self._curv_win_width
    zero_debias = self._zero_debias

    return {
      "sgd_state_dict": sgd_state_dict,
      "global_state": global_state,
      "lr_factor": lr_factor,
      "iter": iter,
      "lr": lr,
      "mu": mu,
      "clip_thresh": clip_thresh,
      "beta": beta,
      "curv_win_width": curv_win_width,
      "zero_debias": zero_debias,
    }


  def load_state_dict(self, state_dict):
    self._optimizer.load_state_dict(state_dict['sgd_state_dict'])
    self._global_state = state_dict['global_state']
    self._lr_factor = state_dict['lr_factor']
    self._iter = state_dict['iter']
    self._lr = state_dict['lr']
    self._mu = state_dict['mu']
    self._clip_thresh = state_dict['clip_thresh']
    self._beta = state_dict['beta']
    self._curv_win_width = state_dict['curv_win_width']
    self._zero_debias = state_dict['zero_debias']


  def set_lr_factor(self, factor):
    self._lr_factor = factor
    return


  def get_lr_factor(self):
    return self._lr_factor


  def zero_grad(self):
    self._optimizer.zero_grad()


  def zero_debias_factor(self):
    return 1.0 - self._beta ** (self._iter + 1)


  def zero_debias_factor_delay(self, delay):
    # for exponentially averaged stat which starts at non-zero iter
    return 1.0 - self._beta ** (self._iter - delay + 1)


  def curvature_range(self):
    global_state = self._global_state
    if self._iter == 0:
      global_state["curv_win"] = torch.FloatTensor(self._curv_win_width, 1).zero_()
    curv_win = global_state["curv_win"]
    grad_norm_squared = self._global_state["grad_norm_squared"]
    curv_win[self._iter % self._curv_win_width] = np.log(grad_norm_squared)
    valid_end = min(self._curv_win_width, self._iter + 1)
    # we use running average over log scale, accelerating 
    # h_max / min in the begining to follow the varying trend of curvature.
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
      self._h_min = np.exp(global_state["h_min_avg"] / debias_factor)
      self._h_max = np.exp(global_state["h_max_avg"] / debias_factor)
    else:
      self._h_min = np.exp(global_state["h_min_avg"] )
      self._h_max = np.exp(global_state["h_max_avg"] )
    if self._sparsity_debias:
      self._h_min *= self._sparsity_avg
      self._h_max *= self._sparsity_avg
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
    # in case of negative variance: the two term are using different debias factors
    self._grad_var = max(self._grad_var, 1e-6)
    if self._sparsity_debias:
      self._grad_var *= self._sparsity_avg
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
    if self._sparsity_debias:
      self._dist_to_opt /= np.sqrt(self._sparsity_avg)
    return


  def grad_sparsity(self):
    global_state = self._global_state
    if self._iter == 0:
      global_state["sparsity_avg"] = 0.0
    non_zero_cnt = 0.0
    all_entry_cnt = 0.0
    for group in self._optimizer.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        grad_non_zero = grad.nonzero()
        if grad_non_zero.dim() > 0:
          non_zero_cnt += grad_non_zero.size()[0]
        all_entry_cnt += torch.numel(grad)
    beta = self._beta
    global_state["sparsity_avg"] = beta * global_state["sparsity_avg"] \
      + (1 - beta) * non_zero_cnt / float(all_entry_cnt)
    self._sparsity_avg = \
      global_state["sparsity_avg"] / self.zero_debias_factor()
    return


  def lr_grad_norm_avg(self):
    # this is for enforcing non-increasing lr * grad_norm after 
    # certain number of iterations. Not necessary for basic use.
    global_state = self._global_state
    beta = self._beta
    if "lr_grad_norm_avg" not in global_state:
      global_state['grad_norm_squared_avg_log'] = 0.0
    global_state['grad_norm_squared_avg_log'] = \
      global_state['grad_norm_squared_avg_log'] * beta + (1 - beta) * np.log(global_state['grad_norm_squared'] )
    if "lr_grad_norm_avg" not in global_state:
      global_state["lr_grad_norm_avg"] = \
        0.0 * beta + (1 - beta) * np.log(self._lr * np.sqrt(global_state['grad_norm_squared'] ) )
    else:
      undebias_val = global_state["lr_grad_norm_avg"] * beta \
        + (1 - beta) * np.log(self._lr * np.sqrt(global_state['grad_norm_squared'] ) )
      debias_factor = self.zero_debias_factor_delay(self._force_non_inc_step_after_iter)
      debias_factor_prev = self.zero_debias_factor_delay(self._force_non_inc_step_after_iter + 1)
      prev_val = global_state["lr_grad_norm_avg"] / debias_factor_prev
      val = undebias_val / debias_factor
      if prev_val > val:
        global_state["lr_grad_norm_avg"] = undebias_val
      else:
        global_state["lr_grad_norm_avg"] = prev_val * debias_factor


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
        global_state['grad_norm_squared'] += torch.sum(grad * grad)
        
    global_state['grad_norm_squared_avg'] = \
      global_state['grad_norm_squared_avg'] * beta + (1 - beta) * global_state['grad_norm_squared']
    # global_state['grad_norm_squared_avg'].mul_(beta).add_(1 - beta, global_state['grad_norm_squared'] )
        
    if self._sparsity_debias:
      self.grad_sparsity()

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


  def get_cubic_root(self):
    # We have the equation x^2 D^2 + (1-x)^4 * C / h_min^2
    # where x = sqrt(mu).
    # We substitute x, which is sqrt(mu), with x = y + 1.
    # It gives y^3 + py = q
    # where p = (D^2 h_min^2)/(2*C) and q = -p.
    # We use the Vieta's substution to compute the root.
    # There is only one real solution y (which is in [0, 1] ).
    # http://mathworld.wolfram.com/VietasSubstitution.html
    assert not math.isnan(self._dist_to_opt)
    assert not math.isnan(self._h_min)
    assert not math.isnan(self._grad_var)
    assert not math.isinf(self._dist_to_opt)
    assert not math.isinf(self._h_min)
    assert not math.isinf(self._grad_var)
    p = self._dist_to_opt**2 * self._h_min**2 / 2 / self._grad_var
    w3 = (-math.sqrt(p**2 + 4.0 / 27.0 * p**3) - p) / 2.0
    w = math.copysign(1.0, w3) * math.pow(math.fabs(w3), 1.0/3.0)
    y = w - p / 3.0 / w
    x = y + 1
    return x


  def get_mu(self):
    root = self.get_cubic_root()
    dr = self._h_max / self._h_min
    self._mu_t = max(root**2, ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2 )
    return 


  def update_hyper_param(self):
    for group in self._optimizer.param_groups:
      group['momentum'] = self._mu
      if self._force_non_inc_step_after_iter == None or self._iter < self._force_non_inc_step_after_iter:
        group['lr'] = self._lr * self._lr_factor
      else:
        # for the learning rate we force to guarantee 
        # lr * grad_norm is non-increasing. Note exponentially
        # averaged stat in lr_grad_norm_avg starts at iteration
        # self._force_non_inc_step_after_iter. Not necessary for basic use.
        self.lr_grad_norm_avg()
        debias_factor = self.zero_debias_factor_delay(self._force_non_inc_step_after_iter)
        group['lr'] = min(self._lr * self._lr_factor,
          np.exp(self._global_state["lr_grad_norm_avg"] / debias_factor) \
          / np.sqrt(np.exp(self._global_state['grad_norm_squared_avg_log'] / debias_factor) ) )
    return


  def auto_clip_thresh(self):
    # Heuristic to automatically prevent sudden exploding gradient
    # Not necessary for basic use.
    return math.sqrt(self._h_max) * self._auto_clip_fac


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
    elif (hasattr(self, '_h_max') and self._auto_clip_fac != None):
      # do not clip the first iteration
      torch.nn.utils.clip_grad_norm(self._var_list, self.auto_clip_thresh() )


    # apply update
    self._optimizer.step()

    # after appply
    self.after_apply()

    # update learning rate and momentum
    self.update_hyper_param()

    self._iter += 1
    return 

