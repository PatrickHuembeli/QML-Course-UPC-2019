import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from RBM_helper import spin_config, spin_list, overlapp_fct, RBM
import gzip
import pickle

batch_size = 512
epochs = 200
all_spins = spin_list(10)
gpu = False

dummy_training = True

#DUMMY TRAINING SET
# ------------------------------------------------------------------------------
#define a simple training set and check if rbm.draw() returns this after training.
if dummy_training:
    data = np.array([[1,0,1,0,1,0,1,0,1,0]]*1000 + [[0,1,0,1,0,1,0,1,0,1]]*1000)
    np.random.shuffle(data)
    data = torch.FloatTensor(data)
# ------------------------------------------------------------------------------

vis = len(data[0]) #input dimension

rbm = RBM(n_vis = vis, n_hin = vis, k=1, gpu = gpu)
if gpu:
    rbm = rbm.cuda()
    all_spins = all_spins.cuda()

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           shuffle=True)

for epoch in range(epochs):
    # What does this code do? (next 5 lines?)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               shuffle=True)
    print(epoch)
    momentum = 1 - 0.1*(epochs-epoch)/epochs #starts at 0.9 and goes up to 1
    lr = (0.1*np.exp(-epoch/epochs*10))+0.0001
    rbm.train(train_loader, lr = lr, momentum = momentum)

print('DRAW SAMPLES')
for i in range(10):
    a = rbm.draw_sample(10)
    print(a)
