
# coding: utf-8

# In[1]:


import torch


# In[2]:


import sys


# In[3]:


import numpy as np


# In[4]:


# %load_ext autoreload
# %autoreload 2


# In[5]:


sys.path.append("./tuner_utils")
from yellowfin import YFOptimizer


# In[ ]:





# In[ ]:





# In[6]:


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


# In[ ]:





# ### YF

# In[7]:


# Thing that works well
# in tuner
#       self._lr = beta * self._lr + (1 - beta) * self._lr_t/self.zero_debias_factor()
# initialization:
#       lr=0.0001


# In[8]:


def torch_list_grad_norm(param_list):
    squared_sum = Variable(torch.zeros(1))
    for param in param_list:
        squared_sum += param.grad.norm()**2
    return squared_sum.sqrt()
        


# In[9]:


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
for t in range(660):
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
    
    if loss.data[0]>5*min_loss_so_far:
        # JIAN: THIS IS THE CHECK FOR BOUNCING BACK
        # We might want to throw an exception here (if that's how TravisCI works)
        print "*********"
        
    if t>10 and t<650:
        if t==11:
            print 'Zero gradients start'
        if t==649:
            print 'Zero gradients stop'
        target_norm = Variable(0.0*torch.ones(1))
    
        grad_norm = torch_list_grad_norm(optimizer._optimizer.param_groups[0]['params'])

        for param in optimizer._optimizer.param_groups[0]['params']:
            param.grad = target_norm * param.grad / grad_norm
        
        # You can enable this to see some slow reaction from our estimators
        # when the zero gradients start 
        if True and (t>640 or t<20):
            print(t, loss.data[0])
            print 'Curvatures', optimizer._h_max, optimizer._h_min
            print 'mu_t, lr_t', optimizer._mu_t, optimizer._lr_t
            print
    else:
        print(t, loss.data[0])

    #print 'After', torch_list_grad_norm(optimizer._optimizer.param_groups[0]['params'])
    
    
    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()
    
    # Print stuff
    #if t>0:
    #    print([optimizer._lr_t, optimizer._lr])


# In[10]:


grad_norm = torch_list_grad_norm(optimizer._optimizer.param_groups[0]['params'])


# In[11]:


4/(np.sqrt(optimizer._h_max)+np.sqrt(optimizer._h_min))**2


# In[12]:


4/(2*np.sqrt(grad_norm**2))**2


# In[ ]:





# In[13]:


optimizer._lr_t


# In[14]:


optimizer._lr


# In[ ]:





# In[15]:


optimizer._h_min


# In[16]:


optimizer._h_max


# In[ ]:





# In[17]:


dr = optimizer._h_max/optimizer._h_min
dr


# In[18]:


((dr - 1)/(dr+1))**2


# In[19]:


optimizer.get_cubic_root()


# In[20]:


optimizer._grad_var


# In[21]:


optimizer._lr_t


# In[22]:


optimizer._mu_t


# In[ ]:





# In[23]:


param.grad


# In[24]:


grad_norm


# In[ ]:





# In[ ]:





# In[25]:


import numpy as np


# In[26]:


np.log(0)


# In[ ]:





# In[ ]:





# In[27]:


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
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(300):
  # Forward pass: compute predicted y by passing x to the model.
  y_pred = model(x)

  # Compute and print loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  # Before the backward pass, use the optimizer object to zero all of the
  # gradients for the variables it will update (which are the learnable weights
  # of the model)
  optimizer.zero_grad()

  # Backward pass: compute gradient of the loss with respect to model parameters
  loss.backward()

  # Calling the step function on an Optimizer makes an update to its parameters
  optimizer.step()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




