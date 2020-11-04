#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import math
import torch
import scipy.io as sio
    
#Settings
edge_life = True
edge_life_window = 10
no_diag = 20
dataset = 'AMLSim'
make_symmetric = True
print(dataset)

def print_tensor(A, str):
    print('------------------------')
    print(str)
    print(A)
    print(torch.sum(A._values()))
    print('------------------------')

if dataset == 'Bitcoin Alpha':
    data = np.loadtxt('/home/shivmaran/Desktop/Tensor-GCN/data/Bitcoin_Alpha/soc-sign-bitcoinalpha.csv', delimiter=',')
    save_file_location = '/home/shivmaran/Desktop/Tensor-GCN/data/Bitcoin_Alpha/'
    save_file_name = 'saved_content_python_bitcoin_alpha.mat'
    time_delta = 60*60*24*14 # 2 weeks
    no_train_samples = 95
    no_val_samples = 20
    no_test_samples = 20
    data[:,0] = data[:,0] - 1
    data[:,1] = data[:,1] - 1
elif dataset == 'Chess':
    data = np.loadtxt('/home/shivmaran/Desktop/Tensor-GCN/data/chess/out.chess.csv', delimiter=',', skiprows=1)
    #data = readmatrix('./data/chess/out.chess.csv')
    save_file_location = '/home/shivmaran/Desktop/Tensor-GCN/data/chess/'
    save_file_name = 'saved_content_python_chess.mat'
    time_delta = 60*60*24*31 # 31 days
    no_train_samples = 80
    no_val_samples = 10
    no_test_samples = 10
    data[:,0] = data[:,0] - 1
    data[:,1] = data[:,1] - 1
elif dataset == 'AMLSim':
    data = np.loadtxt('/home/shivmaran/Desktop/Tensor-GCN/data/AMLSim/transactions.csv', delimiter=',', skiprows=1, usecols=[1,2,7,5])  
    #data = np.loadtxt('/home/shivmaran/Desktop/Tensor-GCN/data/AMLSim/transactions.csv', dtype={'names':('N1','N2','Time','Label'),'formats':(np.long, np.long,np.long,'|S5')},delimiter=',', skiprows=1, usecols=(1,2,5,6))
    save_file_location = '/home/shivmaran/Desktop/Tensor-GCN/data/AMLSim/'
    save_file_name = 'saved_content_amlsim.mat'
    time_delta = 60*60*24*31 # 31 days
    no_train_samples = 160
    no_val_samples = 20
    no_test_samples = 20        
else:
    print('Invalid dataset')
    #exit
#print(data)
data = torch.tensor(data)

# Create full tensor
if dataset == 'Chess' or dataset == 'AMLSim':
    dates = np.unique(data[:,3])
    no_time_slices = len(dates)
else:
    no_time_slices = math.floor((max(data[:,3]) - min(data[:,3]))/time_delta)

print(type(data), data.size())
N = int(max(max(data[:,0]), max(data[:,1]))) + 1
T = int(no_train_samples)
TT = int(no_time_slices)
print(N, T, TT)


# In[2]:


#Create M
M = np.zeros((T,T))
for i in range(no_diag):
    A = M[i:, :T-i]
    np.fill_diagonal(A, 1)
L = np.sum(M, axis=1)
M = M/L[:,None]
M = torch.tensor(M) 
print(M, type(M))
print(torch.sum(M))


# In[3]:


#Create A and A_labels
if not dataset == 'Chess' or dataset == 'AMLSim':
    data = data[data[:,3] < min(data[:,3])+TT*time_delta]
    start_time = min(data[:,3]);

tensor_idx = torch.zeros([data.size()[0], 3], dtype=torch.long)
tensor_val = torch.ones([data.size()[0]], dtype=torch.double)
tensor_labels = torch.zeros([data.size()[0]], dtype=torch.double)#Discuss with Venkat to assign a type

for t in range(TT):
    if dataset == 'Chess' or dataset == 'AMLSim':
        idx = data[:,3] == dates[t]
    else:
        end_time = start_time + time_delta
        idx = (data[:, 3] >= start_time) & (data[:, 3] < end_time)
        start_time = end_time
    
    tensor_idx[idx, 1:3] = (data[idx, 0:2]).type('torch.LongTensor') #Discuss with Venkat for 0-indexing
    tensor_idx[idx, 0] = t
    tensor_labels[idx] = data[idx, 2].type('torch.DoubleTensor')

A =  torch.sparse.DoubleTensor(tensor_idx.transpose(1,0), tensor_val, torch.Size([TT, N, N])).coalesce()
A_labels = torch.sparse.DoubleTensor(tensor_idx.transpose(1,0), tensor_labels, torch.Size([TT, N, N])).coalesce()
#Discuss with Venkat about the descrepency in A_labels
print(A)
print(A_labels)
print(torch.sum(A._values()), torch.sum(A_labels._values()))


# In[4]:


def func_make_symmetric(sparse_tensor, N, TT):  
    count = 0
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)
    A_idx = sparse_tensor._indices()
    A_val = sparse_tensor._values()
    for j in range(TT):
        idx = A_idx[0] == j
        mat = torch.sparse.DoubleTensor(A_idx[1:3,idx], A_val[idx], torch.Size([N,N]))
        mat_t = mat.transpose(1,0)
        sym_mat = mat + mat_t
        sym_mat = sym_mat/2
        count = count + sym_mat._nnz()
        vertices = torch.tensor(sym_mat._indices())
        time = torch.ones(sym_mat._nnz(), dtype=torch.long)* j
        time = time.unsqueeze(0)
        full = torch.cat((time,vertices),0)
        tensor_idx = torch.cat((tensor_idx,full),1)
        tensor_val = torch.cat((tensor_val, sym_mat._values().unsqueeze(1)),0)        
    tensor_val.squeeze_(1)
    A =  torch.sparse.DoubleTensor(tensor_idx, tensor_val, torch.Size([TT, N, N])).coalesce()
    return A

if make_symmetric:
    B = func_make_symmetric(A, N, TT)
else:
    B = A
print(B)
print(torch.sum(B._values()))


# In[5]:


def func_edge_life(A, N, TT):
    A_new = A.clone()
    A_new._values()[:] = 0
    idx = A._indices()[0] == 0
    #A_new._values()[idx] = A._values()[idx]
    print(torch.sum(A_new._values()))
    for t in range(TT):
        idx =  (A._indices()[0] >= max(0, t-edge_life_window+1)) & (A._indices()[0] <= t)  
        block = torch.sparse.DoubleTensor(A._indices()[0:3,idx], A._values()[idx], torch.Size([TT, N, N]))
        block._indices()[0] = t
        A_new = A_new + block
    return A_new.coalesce()

if edge_life:
    B = func_edge_life(B,N,TT)
print(B)
print(torch.sum(B._values()))


# In[6]:


def func_laplacian_transformation(B, N, TT):
    vertices = torch.LongTensor([range(N), range(N)])
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)
    for j in range(TT):
        time = torch.ones(N, dtype=torch.long) * j
        time = time.unsqueeze(0)
        full = torch.cat((time,vertices),0)
        tensor_idx = torch.cat((tensor_idx,full),1)
        val = torch.ones(N, dtype=torch.double)
        tensor_val = torch.cat((tensor_val, val.unsqueeze(1)),0) 
    tensor_val.squeeze_(1)
    I = torch.sparse.DoubleTensor(tensor_idx, tensor_val , torch.Size([TT,N,N]))
    C = B + I
    tensor_idx = torch.LongTensor([])
    tensor_val = torch.DoubleTensor([]).unsqueeze(1)
    for j in range(TT):
        idx = C._indices()[0] == j
        mat = torch.sparse.DoubleTensor(C._indices()[1:3,idx], C._values()[idx], torch.Size([N,N]))
        vec = torch.ones([N,1], dtype=torch.double)
        degree =  1/torch.sqrt(torch.sparse.mm(mat, vec))        
        index = torch.LongTensor(C._indices()[0:3,idx].size())
        values = torch.DoubleTensor(C._values()[idx].size())
        index[0] = j
        index[1:3] = mat._indices()
        values = mat._values()
        count = 0
        for i,j in index[1:3].transpose(1,0):
            values[count] = values[count] * degree[i] * degree[j]
            count = count + 1
        tensor_idx = torch.cat((tensor_idx,index), 1)
        tensor_val = torch.cat((tensor_val,values.unsqueeze(1)),0)
    tensor_val.squeeze_(1)
    C = torch.sparse.DoubleTensor(tensor_idx, tensor_val , torch.Size([TT,N,N]))
    return C.coalesce()

C = func_laplacian_transformation(B, N, TT)
Ct = C.clone().coalesce() 
if TT < (T + no_val_samples + no_test_samples):
    TTT= (T + no_val_samples + no_test_samples)
    Ct = torch.sparse.DoubleTensor(Ct._indices(), Ct._values() , torch.Size([TTT,N,N])).coalesce()
else:
    TTT = TT
print(Ct)
print(torch.sum(Ct._values()))
print(Ct.is_coalesced())


# In[7]:


def func_create_sparse(A, N, TTT, T, start, end):
    assert (end-start) == T
    idx = (A._indices()[0] >= start) & (A._indices()[0] < end)        
    index = torch.LongTensor(A._indices()[0:3,idx].size())
    values = torch.DoubleTensor(A._values()[idx].size())    
    index[0:3] = A._indices()[0:3,idx]
    index[0] = index[0] - start
    values = A._values()[idx]
    sub = torch.sparse.DoubleTensor(index, values , torch.Size([T,N,N]))
    return sub.coalesce()


C_train = func_create_sparse(Ct, N, TTT, T, 0, T)#There is a bug in matlab Bitcoin Alpha
C_val = func_create_sparse(Ct, N, TTT, T, no_val_samples, T+no_val_samples)
C_test = func_create_sparse(Ct, N, TTT, T, no_val_samples+no_test_samples, TTT)
print_tensor(C_train, 'C_train')
print_tensor(C_val, 'C_val')
print_tensor(C_test, 'C_test')
print(C_train.is_coalesced(), C_val.is_coalesced(), C_test.is_coalesced())


# In[8]:


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

dense = torch.randn(3,3)
dense[[0,0,1], [1,2,0]] = 0 # make sparse
print(dense)
sparse = to_sparse(dense)
print(sparse)


# In[9]:


def func_MProduct(C, M):
    assert C.size()[0] == M.size()[0]
    Tr = C.size()[0]
    N = C.size()[1]
    C_new = torch.sparse.DoubleTensor(C.size())
    #C_new = C.clone()
    for j in range(Tr):
        idx = C._indices()[0] == j
        mat = torch.sparse.DoubleTensor(C._indices()[1:3,idx], C._values()[idx], torch.Size([N,N]))
        tensor_idx = torch.zeros([3, mat._nnz()], dtype=torch.long)
        tensor_val = torch.zeros([mat._nnz()], dtype=torch.double)
        tensor_idx[1:3] = mat._indices()[0:2]
        indices = torch.nonzero(M[:,j])
        assert indices.size()[0] <= no_diag
        for i in range(indices.size()[0]):
            tensor_idx[0] = indices[i]
            tensor_val = M[indices[i], j] * mat._values()
            C_new = C_new + torch.sparse.DoubleTensor(tensor_idx, tensor_val , C.size())
        C_new.coalesce()                      
    return C_new.coalesce()  
        
Ct_train = func_MProduct(C_train, M)#There is a bug in matlab Bitcoin Alpha
Ct_val = func_MProduct(C_val, M)
Ct_test = func_MProduct(C_test, M)
print_tensor(Ct_train, 'Ct_train')
print_tensor(Ct_val, 'Ct_val')
print_tensor(Ct_test, 'Ct_test')
print(Ct_train.is_coalesced(), Ct_val.is_coalesced(), Ct_test.is_coalesced())


# In[10]:


A_subs = A._indices()
A_vals = A._values()
A_labels_subs = A_labels._indices()
A_labels_vals = A_labels._values()
C_subs = C._indices()
C_vals = C.values()
C_train_subs = C_train._indices()
C_train_vals = C_train.values()
C_val_subs = C_val._indices()
C_val_vals = C_val.values()
C_test_subs = C_test._indices()
C_test_vals = C_test.values()
Ct_train_subs = Ct_train._indices()
Ct_train_vals = Ct_train.values()
Ct_val_subs = Ct_val._indices()
Ct_val_vals = Ct_val.values()
Ct_test_subs = Ct_test._indices()
Ct_test_vals = Ct_test.values()

print(save_file_name)
print(tensor_idx)
print(save_file_location + save_file_name)

sio.savemat(save_file_location + save_file_name, {
        'tensor_idx': np.array(tensor_idx),
        'tensor_labels': np.array(tensor_labels),
        'A_labels_subs': np.array(A_labels_subs),
        'A_labels_vals': np.array(A_labels_vals),
        'A_subs': np.array(A_subs),
        'A_vals': np.array(A_vals),
        'C_subs': np.array(C_subs),
        'C_vals': np.array(C_vals),
        'C_train_subs': np.array(C_train_subs),
        'C_train_vals': np.array(C_train_vals),
        'C_val_subs': np.array(C_val_subs),
        'C_val_vals': np.array(C_val_vals),
        'C_test_subs': np.array(C_test_subs),
        'C_test_vals': np.array(C_test_vals),
        'Ct_train_subs': np.array(Ct_train_subs),
        'Ct_train_vals': np.array(Ct_train_vals),
        'Ct_val_subs': np.array(Ct_val_subs),
        'Ct_val_vals': np.array(Ct_val_vals),
        'Ct_test_subs': np.array(Ct_test_subs),
        'Ct_test_vals': np.array(Ct_test_vals),
        'M': np.array(M)
    })


# In[ ]:





# In[ ]:




