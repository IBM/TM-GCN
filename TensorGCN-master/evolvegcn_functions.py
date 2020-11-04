# This file contains all the things needed for EvolveGCN

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
import math
import os.path
unsq = t.unsqueeze
sq = t.squeeze

# This is a 1-layer version of EvolveGCN
class EvolveGCN_1_layer(nn.Module):
	def __init__(self, A, X, edges, hidden_feat=[2,2]):
		super(EvolveGCN_1_layer, self).__init__()
		self.A = A
		self.X = X
		self.T, self.N, _ = X.shape
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
		self.edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		self.tanh = t.nn.Tanh()
		self.sigmoid = t.nn.Sigmoid()
		self.relu = nn.ReLU(inplace=False)
		self.F = [X.shape[-1]] + hidden_feat

		i = 0
		self.p = nn.Parameter(t.randn(self.F[i]).double())
		self.W_Z = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_Z = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_Z = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_R = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_R = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_R = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_H = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_H = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_H = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_init = t.randn(self.F[i], self.F[i+1]).double()
		self.U = nn.Parameter(t.randn(self.F[-2]*2, self.F[-1]))

	def __call__(self, A=None, X=None, edges=None, W_init=None):
		output, W = self.forward(A, X, edges, W_init)
		return output, W

	def forward(self, A=None, X=None, edges=None, W_init=None):
		if type(A)==list and type(X)==t.Tensor and type(edges)==t.Tensor and type(W_init)==t.Tensor:
			edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
			edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
			W = W_init
		else:
			A = self.A
			X = self.X
			edge_src_nodes = self.edge_src_nodes
			edge_trg_nodes = self.edge_trg_nodes
			W = self.W_init

		Y = t.zeros(self.T, self.N, self.F[-2])
		for time in range(X.shape[0]):
			X_slice = X[time]
			W = self.GRU(X_slice, W)
			X_slice = self.GCONV(A[time], X_slice, W, nonlin=False)
			Y[time] = X_slice
	
		Y_mat_edge_src_nodes = Y.reshape(-1, Y.shape[-1])[edge_src_nodes]
		Y_mat_edge_trg_nodes = Y.reshape(-1, Y.shape[-1])[edge_trg_nodes]
		Y_mat = t.cat((Y_mat_edge_src_nodes, Y_mat_edge_trg_nodes), dim=1)
		output = t.matmul(Y_mat, self.U)

		return output, W

	def summarize(self, X, k):
		y = t.matmul(X, self.p)/t.norm(self.p, 2)
		_, idx = t.topk(y, k)
		Z = X[idx,:] * y[idx].repeat(X.shape[1], 1).transpose(0,1)
		return Z

	def g(self, X, H):
		Z = self.sigmoid(t.matmul(self.W_Z, X) + t.matmul(self.U_Z, H) + self.B_Z)
		R = self.sigmoid(t.matmul(self.W_R, X) + t.matmul(self.U_R, H) + self.B_R)
		Ht = self.tanh(t.matmul(self.W_H, X) + t.matmul(self.U_H, R*H) + self.B_H)
		H = (1-Z)*H + Z*Ht
		return H

	def GRU(self, H, W_old):
		W_new = self.g(self.summarize(H, W_old.shape[1]).transpose(0,1), W_old)
		return W_new

	def GCONV(self, A, H, W, nonlin):
		H_new = t.matmul(t.sparse.mm(A, H), W)
		if nonlin:
			H_new = self.relu(H_new)
		return H_new

# This is a 2-layer version of EvolveGCN
class EvolveGCN_2_layer(nn.Module):
	def __init__(self, A, X, edges, hidden_feat=[2,2,2]):
		super(EvolveGCN_2_layer, self).__init__()
		self.A = A
		self.X = X
		self.T, self.N, _ = X.shape
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
		self.edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		self.tanh = t.nn.Tanh()
		self.sigmoid = t.nn.Sigmoid()
		self.relu = nn.ReLU(inplace=False)
		self.F = [X.shape[-1]] + hidden_feat

		i = 0
		self.p = nn.Parameter(t.randn(self.F[i]).double())
		self.W_Z = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_Z = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_Z = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_R = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_R = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_R = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_H = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_H = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_H = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_init = t.randn(self.F[i], self.F[i+1]).double()

		i = 1
		self.p2 = nn.Parameter(t.randn(self.F[i]).double())
		self.W_Z2 = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_Z2 = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_Z2 = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_R2 = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_R2 = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_R2 = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_H2 = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_H2 = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_H2 = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_init2 = t.randn(self.F[i], self.F[i+1]).double()

		self.U = nn.Parameter(t.randn(self.F[-2]*2, self.F[-1]))

	def __call__(self, A=None, X=None, edges=None, W_init=None, W_init2=None):
		output, W, W2 = self.forward(A, X, edges, W_init, W_init2)
		return output, W, W2

	def forward(self, A=None, X=None, edges=None, W_init=None, W_init2=None):
		if type(A)==list and type(X) ==t.Tensor and type(edges)==t.Tensor:
			edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
			edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
			W = W_init
			W2 = W_init2
		else:
			A = self.A
			X = self.X
			edge_src_nodes = self.edge_src_nodes
			edge_trg_nodes = self.edge_trg_nodes
			W = self.W_init
			W2 = self.W_init2

		Y = t.zeros(self.T, self.N, self.F[-2])
		for time in range(X.shape[0]):
			X_slice = X[time]
			W = self.GRU(X_slice, W, 1)
			X_slice = self.GCONV(A[time], X_slice, W, nonlin=True)
			W2 = self.GRU(X_slice, W2, 2)
			X_slice = self.GCONV(A[time], X_slice, W2, nonlin=False)
			Y[time] = X_slice
	
		Y_mat_edge_src_nodes = Y.reshape(-1, Y.shape[-1])[edge_src_nodes]
		Y_mat_edge_trg_nodes = Y.reshape(-1, Y.shape[-1])[edge_trg_nodes]
		Y_mat = t.cat((Y_mat_edge_src_nodes, Y_mat_edge_trg_nodes), dim=1)
		output = t.matmul(Y_mat, self.U)

		return output, W, W2

	def summarize(self, X, k, l):
		if l == 1:
			p = self.p
		elif l == 2:
			p = self.p2
		y = t.matmul(X, p)/t.norm(p, 2)
		_, idx = t.topk(y, k)
		Z = X[idx,:] * y[idx].repeat(X.shape[1], 1).transpose(0,1)
		return Z

	def g(self, X, H, l):
		if l == 1:
			W_Z, U_Z, B_Z = self.W_Z, self.U_Z, self.B_Z
			W_R, U_R, B_R = self.W_R, self.U_R, self.B_R
			W_H, U_H, B_H = self.W_H, self.U_H, self.B_H
		elif l == 2:
			W_Z, U_Z, B_Z = self.W_Z2, self.U_Z2, self.B_Z2
			W_R, U_R, B_R = self.W_R2, self.U_R2, self.B_R2
			W_H, U_H, B_H = self.W_H2, self.U_H2, self.B_H2
		Z = self.sigmoid(t.matmul(W_Z, X) + t.matmul(U_Z, H) + B_Z)
		R = self.sigmoid(t.matmul(W_R, X) + t.matmul(U_R, H) + B_R)
		Ht = self.tanh(t.matmul(W_H, X) + t.matmul(U_H, R*H) + B_H)
		H = (1-Z)*H + Z*Ht
		return H

	def GRU(self, H, W_old, l):
		W_new = self.g(self.summarize(H, W_old.shape[1], l).transpose(0,1), W_old, l)
		return W_new

	def GCONV(self, A, H, W, nonlin):
		H_new = t.matmul(t.sparse.mm(A, H), W)
		if nonlin:
			H_new = self.relu(H_new)
		return H_new

# This class, which is meant to be flexible and allow creation of an EvolveGCN with an arbitrary number of layers, does not work with backward properly--for some reason...
class EvolveGCN(nn.Module):
	def __init__(self, A, X, edges, F=[2,2,2]):
		super(EvolveGCN, self).__init__()
		self.A = A
		self.X = X
		self.T, self.N, _ = X.shape
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
		self.edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		self.tanh = t.nn.Tanh()
		self.sigmoid = t.nn.Sigmoid()
		self.relu = nn.ReLU(inplace=False)
		self.F = F
		self.p = nn.ParameterList()
		self.W_Z = nn.ParameterList()
		self.U_Z = nn.ParameterList()
		self.B_Z = nn.ParameterList()
		self.W_R = nn.ParameterList()
		self.U_R = nn.ParameterList()
		self.B_R = nn.ParameterList()
		self.W_H = nn.ParameterList()
		self.U_H = nn.ParameterList()
		self.B_H = nn.ParameterList()
		self.W_init = []
		self.U = nn.Parameter(t.randn(F[-2]*2, F[-1]))
		for i in range(len(self.F)-2):
			self.p.append(nn.Parameter(t.randn(F[i]).double()))
			self.W_Z.append(nn.Parameter(t.randn(F[i], F[i]).double()))
			self.U_Z.append(nn.Parameter(t.randn(F[i], F[i]).double()))
			self.B_Z.append(nn.Parameter(t.randn(F[i], F[i+1]).double()))
			self.W_R.append(nn.Parameter(t.randn(F[i], F[i]).double()))
			self.U_R.append(nn.Parameter(t.randn(F[i], F[i]).double()))
			self.B_R.append(nn.Parameter(t.randn(F[i], F[i+1]).double()))
			self.W_H.append(nn.Parameter(t.randn(F[i], F[i]).double()))
			self.U_H.append(nn.Parameter(t.randn(F[i], F[i]).double()))
			self.B_H.append(nn.Parameter(t.randn(F[i], F[i+1]).double()))
			stdv = 1/np.sqrt(F[i+1])
			#self.W_init.append(t.tensor(np.random.uniform(-stdv, stdv, (F[i], F[i+1]))))
			self.W_init.append(t.randn(F[i], F[i+1]).double())
			#self.W_init.append(nn.Parameter(t.randn(F[i], F[i+1]).double()))

	def __call__(self, A=None, X=None, edges=None):
		return self.forward(A, X, edges)

	def forward(self, A=None, X=None, edges=None):
		if type(A)==list and type(X) ==t.Tensor and type(edges)==t.Tensor:
			edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
			edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		else:
			A = self.A
			X = self.X
			edge_src_nodes = self.edge_src_nodes
			edge_trg_nodes = self.edge_trg_nodes

		Y = t.zeros(self.T, self.N, self.F[-2])
		W = self.W_init
		for time in range(X.shape[0]):
			X_slice = X[time]
			for i in range(len(self.F)-2):
				W[i] = self.GRU(X_slice, W[i], i)
				nonlin = not (i == len(self.F) - 3)
				X_slice = self.GCONV(A[time], X_slice, W[i], nonlin)
			Y[time] = X_slice
	
		Y_mat_edge_src_nodes = Y.reshape(-1, Y.shape[-1])[edge_src_nodes]
		Y_mat_edge_trg_nodes = Y.reshape(-1, Y.shape[-1])[edge_trg_nodes]
		Y_mat = t.cat((Y_mat_edge_src_nodes, Y_mat_edge_trg_nodes), dim=1)
		output = t.matmul(Y_mat, self.U)

		return output

	def summarize(self, X, k, l):
		y = t.matmul(X, self.p[l])/t.norm(self.p[l], 2)
		_, idx = t.topk(y, k)
		Z = X[idx,:] * y[idx].repeat(X.shape[1], 1).transpose(0,1)
		return Z

	def g(self, X, H, l):
		Z = self.sigmoid(t.matmul(self.W_Z[l], X) + t.matmul(self.U_Z[l], H) + self.B_Z[l])
		R = self.sigmoid(t.matmul(self.W_R[l], X) + t.matmul(self.U_R[l], H) + self.B_R[l])
		Ht = self.tanh(t.matmul(self.W_H[l], X) + t.matmul(self.U_H[l], R*H) + self.B_H[l])
		H = (1-Z)*H + Z*Ht
		return H

	def GRU(self, H, W_old, l):
		W_new = self.g(self.summarize(H, W_old.shape[1], l).transpose(0,1), W_old, l)
		return W_new

	def GCONV(self, A, H, W, nonlin):
		H_new = t.matmul(t.sparse.mm(A, H), W)
		if nonlin:
			H_new = self.relu(H_new)
		return H_new
    
class EvolveGCN_reg(nn.Module):
	def __init__(self, A, X, hidden_feat=[2,2]):
		super(EvolveGCN_reg, self).__init__()
		self.A = A
		self.X = X
		self.T, self.N, _ = X.shape
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.tanh = t.nn.Tanh()
		self.sigmoid = t.nn.Sigmoid()
		self.relu = nn.ReLU(inplace=False)
		self.F = [X.shape[-1]] + hidden_feat

		i = 0
		self.p = nn.Parameter(t.randn(self.F[i]).double())
		self.W_Z = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_Z = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_Z = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_R = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_R = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_R = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_H = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.U_H = nn.Parameter(t.randn(self.F[i], self.F[i]).double())
		self.B_H = nn.Parameter(t.randn(self.F[i], self.F[i+1]).double())
		self.W_init = t.randn(self.F[i], self.F[i+1]).double()
		self.U = nn.Parameter(t.randn(self.F[-2]*2, self.F[-1]))
		self.lin1 = nn.Linear(self.F[1], 1)

	def __call__(self, A=None, X=None, W_init=None):
		output = self.forward(A, X, W_init)
		return output

	def forward(self, A=None, X=None, W_init=None):
		if type(A)==list and type(X)==t.Tensor and type(W_init)==t.Tensor:
			W = W_init
		else:
			A = self.A
			X = self.X
			W = self.W_init

		Y = t.zeros(self.T, self.N, self.F[-2])
		for time in range(X.shape[0]):
			X_slice = X[time]
			W = self.GRU(X_slice, W)
			X_slice = self.GCONV(A[time], X_slice, W, nonlin=False)
			Y[time] = X_slice
            
		output = self.lin1(Y)

		return output.squeeze(2)

	def summarize(self, X, k):
		y = t.matmul(X, self.p)/t.norm(self.p, 2)
		_, idx = t.topk(y, k)
		Z = X[idx,:] * y[idx].repeat(X.shape[1], 1).transpose(0,1)
		return Z

	def g(self, X, H):
		Z = self.sigmoid(t.matmul(self.W_Z, X) + t.matmul(self.U_Z, H) + self.B_Z)
		R = self.sigmoid(t.matmul(self.W_R, X) + t.matmul(self.U_R, H) + self.B_R)
		Ht = self.tanh(t.matmul(self.W_H, X) + t.matmul(self.U_H, R*H) + self.B_H)
		H = (1-Z)*H + Z*Ht
		return H

	def GRU(self, H, W_old):
		W_new = self.g(self.summarize(H, W_old.shape[1]).transpose(0,1), W_old)
		return W_new

	def GCONV(self, A, H, W, nonlin):
		H_new = t.matmul(t.sparse.mm(A, H), W)
		if nonlin:
			H_new = self.relu(H_new)
		return H_new