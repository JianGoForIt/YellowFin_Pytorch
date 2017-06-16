import os
import torch
import numpy as np
from yellowfin import YFOptimizer
from torch.autograd import Variable
import time


n_dim = 1000000
n_iter = 50

def tune_everything(x0squared, C, T, gmin, gmax):
  # First tune based on dynamic range    
  if C==0:
    dr=gmax/gmin
    mustar=((np.sqrt(dr)-1)/(np.sqrt(dr)+1))**2
    alpha_star = (1+np.sqrt(mustar))**2/gmax
    
    return alpha_star,mustar

  dist_to_opt = x0squared
  grad_var = C
  max_curv = gmax
  min_curv = gmin
  const_fact = dist_to_opt * min_curv**2 / 2 / grad_var
  coef = [-1, 3, -(3 + const_fact), 1]
  roots = np.roots(coef)
  roots = roots[np.real(roots) > 0]
  roots = roots[np.real(roots) < 1]
  root = roots[np.argmin(np.imag(roots) ) ]

  assert root > 0 and root < 1 and np.absolute(root.imag) < 1e-6

  dr = max_curv / min_curv
  assert max_curv >= min_curv
  mu = max( ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2, root**2)

  lr_min = (1 - np.sqrt(mu) )**2 / min_curv
  lr_max = (1 + np.sqrt(mu) )**2 / max_curv

  alpha_star = lr_min
  mustar = mu

  return alpha_star, mustar


def test_measurement(zero_debias=True):
  dtype = torch.FloatTensor
  w = Variable(torch.ones(n_dim, 1).type(dtype), requires_grad=True)
  b = Variable(torch.ones(1).type(dtype), requires_grad=True)
  x = Variable(torch.ones(1, n_dim).type(dtype), requires_grad=False)
  opt = YFOptimizer([w, b], lr=1.0, mu=0.0, zero_debias=zero_debias)

  target_h_max = 0.0
  target_h_min = 0.0
  g_norm_squared_avg = 0.0
  g_norm_avg = 0.0
  g_avg = 0.0
  target_dist = 0.0
  for i in range(n_iter):
    opt.zero_grad()

    loss = (x.mm(w) + b).sum()
    loss.backward()
    w.grad.data = (i + 1) * torch.ones( [n_dim, ] ).type(dtype)
    b.grad.data = (i + 1) * torch.ones( [1, ] ).type(dtype)

    opt.step()

    res = [opt._h_max, opt._h_min, opt._grad_var, opt._dist_to_opt]

    g_norm_squared_avg = 0.999 * g_norm_squared_avg  \
      + 0.001 * np.sum(( (i + 1)*np.ones( [n_dim + 1, ] ) )**2)
    g_norm_avg = 0.999 * g_norm_avg  \
      + 0.001 * np.linalg.norm( (i + 1)*np.ones( [n_dim + 1, ] ) )
    g_avg = 0.999 * g_avg + 0.001 * (i + 1)

    target_h_max = 0.999 * target_h_max + 0.001 * (i + 1)**2*(n_dim + 1)
    target_h_min = 0.999 * target_h_min + 0.001 * max(1, i + 2 - 20)**2*(n_dim + 1)
    if zero_debias:
      target_var = g_norm_squared_avg/(1-0.999**(i + 1) ) \
         - g_avg**2 * (n_dim + 1) / (1-0.999**(i + 1) )**2
    else:
      target_var = g_norm_squared_avg - g_avg**2 * (n_dim + 1)
    target_dist = 0.999 * target_dist + 0.001 * g_norm_avg / g_norm_squared_avg

    if i == 0:
      continue
    if zero_debias:
      # print "iter ", i, " h max ", res[0], target_h_max/(1-0.999**(i + 1) ), \
      #   " h min ", res[1], target_h_min/(1-0.999**(i + 1) ), \
      #   " var ", res[2], target_var, \
      #   " dist ", res[3], target_dist/(1-0.999**(i + 1) )
      assert np.abs(target_h_max/(1-0.999**(i + 1) ) - res[0] ) < np.abs(res[0]) * 1e-3
      assert np.abs(target_h_min/(1-0.999**(i + 1) ) - res[1] ) < np.abs(res[1]) * 1e-3
      assert np.abs(target_var - res[2] ) < np.abs(target_var ) * 1e-3
      assert np.abs(target_dist/(1-0.999**(i + 1) ) - res[3] ) < np.abs(res[3] ) * 1e-3
    else:
      # print "iter ", i, " h max ", res[0], target_h_max, " h min ", res[1], target_h_min, \
      # " var ", res[2], target_var, " dist ", res[3], target_dist
      assert np.abs(target_h_max - res[0] ) < np.abs(target_h_max) * 1e-3
      assert np.abs(target_h_min - res[1] ) < np.abs(target_h_min) * 1e-3
      assert np.abs(target_var - res[2] ) < np.abs(res[2] ) * 1e-3
      assert np.abs(target_dist - res[3] ) < np.abs(res[3] ) * 1e-3
  print "sync measurement test passed!"


def test_lr_mu(zero_debias=False):
  dtype = torch.FloatTensor
  w = Variable(torch.ones(n_dim, 1).type(dtype), requires_grad=True)
  b = Variable(torch.ones(1).type(dtype), requires_grad=True)
  x = Variable(torch.ones(1, n_dim).type(dtype), requires_grad=False)
  opt = YFOptimizer([w, b], lr=1.0, mu=0.0, zero_debias=zero_debias)

  target_h_max = 0.0
  target_h_min = 0.0
  g_norm_squared_avg = 0.0
  g_norm_avg = 0.0
  g_avg = 0.0
  target_dist = 0.0
  target_lr = 1.0
  target_mu = 0.0
  for i in range(n_iter):
    opt.zero_grad()

    loss = (x.mm(w) + b).sum()
    loss.backward()
    w.grad.data = (i + 1) * torch.ones( [n_dim, ] ).type(dtype)
    b.grad.data = (i + 1) * torch.ones( [1, ] ).type(dtype)

    opt.step()
    res = [opt._h_max, opt._h_min, opt._grad_var, opt._dist_to_opt, opt._lr, opt._mu]

    g_norm_squared_avg = 0.999 * g_norm_squared_avg  \
      + 0.001 * np.sum(( (i + 1)*np.ones( [n_dim + 1, ] ) )**2)
    g_norm_avg = 0.999 * g_norm_avg  \
      + 0.001 * np.linalg.norm( (i + 1)*np.ones( [n_dim + 1, ] ) )
    g_avg = 0.999 * g_avg + 0.001 * (i + 1)

    target_h_max = 0.999 * target_h_max + 0.001 * (i + 1)**2*(n_dim + 1)
    target_h_min = 0.999 * target_h_min + 0.001 * max(1, i + 2 - 20)**2*(n_dim + 1)
    if zero_debias:
      target_var = g_norm_squared_avg/(1-0.999**(i + 1) ) \
         - g_avg**2 * (n_dim + 1) / (1-0.999**(i + 1) )**2
    else:
      target_var = g_norm_squared_avg - g_avg**2 * (n_dim + 1)
    target_dist = 0.999 * target_dist + 0.001 * g_norm_avg / g_norm_squared_avg

    if i == 0:
      continue
    if zero_debias:
      # print "iter ", i, " h max ", res[0], target_h_max/(1-0.999**(i + 1) ), \
      #   " h min ", res[1], target_h_min/(1-0.999**(i + 1) ), \
      #   " var ", res[2], target_var, \
      #   " dist ", res[3], target_dist/(1-0.999**(i + 1) )
      assert np.abs(target_h_max/(1-0.999**(i + 1) ) - res[0] ) < np.abs(res[0]) * 1e-3
      assert np.abs(target_h_min/(1-0.999**(i + 1) ) - res[1] ) < np.abs(res[1]) * 1e-3
      assert np.abs(target_var - res[2] ) < np.abs(target_var ) * 1e-3
      assert np.abs(target_dist/(1-0.999**(i + 1) ) - res[3] ) < np.abs(res[3] ) * 1e-3
    else:
      # print "iter ", i, " h max ", res[0], target_h_max, " h min ", res[1], target_h_min, \
      # " var ", res[2], target_var, " dist ", res[3], target_dist
      assert np.abs(target_h_max - res[0] ) < np.abs(target_h_max) * 1e-3
      assert np.abs(target_h_min - res[1] ) < np.abs(target_h_min) * 1e-3
      assert np.abs(target_var - res[2] ) < np.abs(res[2] ) * 1e-3
      assert np.abs(target_dist - res[3] ) < np.abs(res[3] ) * 1e-3


    if i > 0:
      if zero_debias:
        lr, mu = tune_everything( (target_dist/(1-0.999**(i + 1) ))**2, 
          target_var, 1, target_h_min/(1-0.999**(i + 1) ), target_h_max/(1-0.999**(i + 1) ) )
      else:
        lr, mu = tune_everything(target_dist**2, target_var, 1, target_h_min, target_h_max)
      lr = np.real(lr)
      mu = np.real(mu)
      target_lr = 0.999 * target_lr + 0.001 * lr
      target_mu = 0.999 * target_mu + 0.001 * mu
      # print "lr ", target_lr, res[4], " mu ", target_mu, res[5]
      assert target_lr == 0.0 or np.abs(target_lr - res[4] ) < np.abs(res[4] ) * 1e-3
      assert target_mu == 0.0 or np.abs(target_mu - res[5] ) < np.abs(res[5] ) * 5e-3 
  print "lr and mu computing test passed!"


if __name__ == "__main__":
  start = time.time()
  test_measurement(zero_debias=True)
  end = time.time()
  print "measurement test with zero_debias done in ", (end - start)/float(n_iter), " s/iter!"
  
  start = time.time()
  test_measurement(zero_debias=False)
  end = time.time()
  print "measurement test without zero_debias done in ", (end - start)/float(n_iter), " s/iter!"

  start = time.time()
  test_lr_mu(zero_debias=False)
  end = time.time()
  print "lr and mu test done with zero_debias in ", (end - start)/float(n_iter), " s/iter!"

  start = time.time()
  test_lr_mu(zero_debias=True)
  end = time.time()
  print "lr and mu test done with zero_debias in ", (end - start)/float(n_iter), " s/iter!"



