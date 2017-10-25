
# coding: utf-8
import torch
import sys
import numpy as np
sys.path.append("./tuner_utils")
from yellowfin import YFOptimizer

# Code in file nn/two_layer_net_optim.py
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Set random seed
torch.manual_seed(1)

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

def torch_list_grad_norm(param_list):
    squared_sum = Variable(torch.zeros(1))
    for param in param_list:
        squared_sum += param.grad.norm()**2
    return squared_sum.sqrt()
        

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.

min_loss_so_far = np.inf

optimizer = YFOptimizer(model.parameters(), lr=0.0001, mu=0.0)
for t in range(4000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)


    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    

####### MEDDLING AND MONITORING
    if loss.data[0]<min_loss_so_far:
        min_loss_so_far=loss.data[0]
        
    #
    if loss.data[0]>5*min_loss_so_far:
        # JIAN: THIS IS THE CHECK FOR BOUNCING BACK
        # We might want to throw an exception here (if that's how TravisCI works)
        raise Exception("loss dramatically bounces back after gradient spike")
        
    if t<10 or t>3500:
        print("loss at step ", t, loss.data[0])

        # Extreme gradient values in here
        if t % 2 == 0:
            target_norm = Variable(10000*torch.ones(1))
        else:
            target_norm = Variable(0.0000001*torch.ones(1))
    else:
        # The middle section
        target_norm = Variable(torch.zeros(1))
    
    grad_norm = torch_list_grad_norm(optimizer._optimizer.param_groups[0]['params'])

    for param in optimizer._optimizer.param_groups[0]['params']:
        param.grad = target_norm * param.grad / grad_norm
    #print 'After', torch_list_grad_norm(optimizer._optimizer.param_groups[0]['params'])
    
    # JIAN: ADD CHECKS FOR NANS HERE
    # JIAN: ADD CHECKS FOR DR>=1
    # JIAN: ADD CHECKS FOR CUBIC SOLVER
####### END MONITORING
    
    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()


