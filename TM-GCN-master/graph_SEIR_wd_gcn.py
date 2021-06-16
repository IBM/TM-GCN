# This version of bitcoin experiment imports data preprocessed in Matlab, and uses EvolveGCN

# Imports and aliases
import pickle
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pandas as pd
import datetime
from scipy.sparse import csr_matrix
import os.path
import embedding_help_functions as ehf
import wd_gcn_functions as wgf
import scipy.io as sio
from torch.autograd import Variable
from numpy import linalg as LA
#from read_data import func_create_sparse,func_MProduct
unsq = t.unsqueeze
sq = t.squeeze

# Settings
alpha_vec = [.75, .76, .77, .78, .79, .80, .81, .82, .83, .84, .85, .86, .87, .88, .89, .90, .91, .92, .93, .94, .95]
no_layers = 1
no_epochs = 100
no_trials = 1
no_diag = 20
out_idx = 2

S_train, S_val, S_test = 80, 10, 10
lr = 0.01
momentum = 0.9

def func_create_sparse(A, N, TTT, T, start, end):
    assert (end-start) == T
    idx = (A._indices()[0] >= start) & (A._indices()[0] < end)        
    index = t.LongTensor(A._indices()[0:3,idx].size())
    values = t.DoubleTensor(A._values()[idx].size())    
    index[0:3] = A._indices()[0:3,idx]
    index[0] = index[0] - start
    values = A._values()[idx]
    sub = t.sparse.DoubleTensor(index, values , t.Size([T,N,N]))
    return sub.coalesce()

def func_MProduct(C, M):
    assert C.size()[0] == M.size()[0]
    Tr = C.size()[0]
    N = C.size()[1]
    C_new = t.sparse.DoubleTensor(C.size())
    #C_new = C.clone()
    for j in range(Tr):
        idx = C._indices()[0] == j
        mat = t.sparse.DoubleTensor(C._indices()[1:3,idx], C._values()[idx], t.Size([N,N]))
        tensor_idx = t.zeros([3, mat._nnz()], dtype=t.long)
        tensor_val = t.zeros([mat._nnz()], dtype=t.double)
        tensor_idx[1:3] = mat._indices()[0:2]
        indices = t.nonzero(M[:,j])
        assert indices.size()[0] <= no_diag
        for i in range(indices.size()[0]):
            tensor_idx[0] = indices[i]
            tensor_val = M[indices[i], j] * mat._values()
            C_new = C_new + t.sparse.DoubleTensor(tensor_idx, tensor_val , C.size())
        C_new.coalesce()                      
    return C_new.coalesce()  

def func_MProduct_dense(C, M):
    T = C.shape[0]
    B= C.to_dense()
    B = B.type(t.DoubleTensor)
    X = t.matmul(M, B.reshape(T, -1)).reshape(B.size())
    indices = t.nonzero(X).t()
    values = X[indices[0],indices[1],indices[2]] # modify this based on dimensionality
    Cm = t.sparse.DoubleTensor(indices, values, X.size())
    return Cm

def create_matrix_M(T,no_diag):
    M = np.zeros((T,T))
    for i in range(no_diag):
        A = M[i:, :T-i]
        np.fill_diagonal(A, 1/(i+1))
    L = np.sum(M, axis=1)
#    M = M/L[:,None]
    M = t.tensor(M) 
    return M

def load_data(S_train, S_val, S_test, no_diag):
    # Load stuff from mat file
    data = sio.loadmat("data/Graph_SEIR.mat")
    G = data["DyG"]
    Ys = data["ys"]
    G = G.transpose()
    [T, N, N1]  = G.shape
    A_sz = t.Size([T, N, N])
    C_sz = t.Size([S_train, N, N])
    ij = np.nonzero(G)
    vals = G[ij]
    ij = t.LongTensor(ij)
    vals  = t.DoubleTensor(vals)
    A = t.sparse.DoubleTensor(ij, vals, A_sz)
   
    C_train = func_create_sparse(A, N, T, S_train, 0, S_train)#There is a bug in matlab Bitcoin Alpha
    C_val = func_create_sparse(A, N, T, S_train, S_val, S_train+S_val)
    C_test = func_create_sparse(A, N, T, S_train, S_val+S_test, T)
    
    M = create_matrix_M(S_train,no_diag)
    
    Ct_train = func_MProduct_dense(C_train, M)#There is a bug in matlab Bitcoin Alpha
    Ct_val = func_MProduct_dense(C_val, M)
    Ct_test = func_MProduct_dense(C_test, M)
    Ys = Ys.transpose()
    return A, Ct_train, Ct_val, Ct_test, M, Ys
 
def get_features(A, Ys, out_idx,S_train, S_val, S_test):
    X = t.zeros(A.shape[0], A.shape[1], 2)
    X[:, :, 0] = t.sparse.sum(A, 1).to_dense()
    X[:, :, 1] = t.sparse.sum(A, 2).to_dense()
    Out = t.DoubleTensor(Ys[1:,out_idx,:])
    temp = np.delete(Ys, out_idx, 1)
    temp = t.DoubleTensor(temp[0:Ys.shape[0]-1,:,:])
    temp = t.transpose(temp, 1,2)
    X = X.type(t.DoubleTensor)
    X = t.cat((X,temp),2)
    X_train = X[0:S_train].double()
    X_val = X[S_val:S_train+S_val].double()
    X_test = X[S_val+S_test:].double()
    y_train = Out[0:S_train].double()
    y_val = Out[S_val:S_train+S_val].double()
    y_test = Out[S_val+S_test:].double()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def loss_function(out, y):
    loss = 0
    for ts in range(out.shape[0]):
            loss += F.mse_loss(out[ts,:].float(), y[ts,:].float())
    
    return loss
    

A, Ct_train, Ct_val, Ct_test, M, Ys = load_data(S_train, S_val, S_test, no_diag)
X_train, X_val, X_test, y_train, y_val, y_test = get_features(A, Ys, out_idx,S_train, S_val, S_test)

if no_trials > 1:
    ep_acc_loss_vec = []

for tr in range(no_trials):
# Create gcn for training
    if no_layers == 2: 
        gcn = ehf.EmbeddingGCN2(Ct_train, X_train, y_train, M, hidden_feat=[6,6,2], condensed_W=True, use_Minv=False, nonlin2="selu")
    elif no_layers == 1:
        gcn = wgf.WD_GCN_reg(Ct_train, X_train,  [6,2])

# Train
    optimizer = t.optim.SGD(gcn.parameters(), lr=lr, momentum=momentum)
    ep_acc_loss = np.zeros((no_epochs,12)) # (precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test)
    loss_train = 0
    for ep in range(no_epochs):
        # Compute loss and take step
        optimizer.zero_grad()
        output_train = gcn()
        loss_train = loss_function(output_train, y_train)
        loss_train.backward()
        optimizer.step()
        cur_loss = loss_train.item()

        print("Epoch:", '%04d' % (ep + 1), "Training loss:{}", "{:.5f}".format(cur_loss))  


# Validation set               
output_val = gcn(Ct_val, X_val)
loss = 0 
ratio = 0
for ts in range(output_val.shape[0]):
    out, y  = Variable(output_val[ts,:]), Variable(y_val[ts,:])
    loss += F.l1_loss(out,y,reduction='sum').item()
    ratio+= F.l1_loss(out,y,reduction='sum').item()/ LA.norm(y,1)  

val_loss = loss / output_val.shape[0]
val_ratio = ratio / output_val.shape[0]

print("Validation accuarcy:", val_loss)
print("Validation Error ratio:", val_ratio)


output_test = gcn(Ct_test, X_test)
loss = 0 
ratio = 0
for ts in range(output_test.shape[0]):
    out, y = Variable(output_test[ts,:]), Variable(y_test[ts,:])
    loss += F.l1_loss(out,y,reduction='sum').item()
    ratio+= F.l1_loss(out,y,reduction='sum').item()/ LA.norm(y,1)   

test_loss = loss / output_test.shape[0]
test_ratio = ratio / output_test.shape[0]

print("Test accuarcy:", test_loss)
print("Test Error ratio:", test_ratio)
    