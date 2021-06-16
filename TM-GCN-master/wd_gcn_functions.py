# Here I implement all the things needed for WD-GCN

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

class WD_GCN(nn.Module):
	def __init__(self, A, X, edges, hidden_feat=[2,2]):
		super(WD_GCN, self).__init__()
		self.A = A
		self.X = X
		self.T, self.N, _ = X.shape
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
		self.edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		self.tanh = t.nn.Tanh()
		self.sigmoid = t.nn.Sigmoid()
		self.relu = nn.ReLU(inplace=False)
		self.AX = self.compute_AX(A, X)

		# GCN parameters
		self.W = nn.Parameter(t.randn(X.shape[-1], hidden_feat[0]))

		# LSTM parameters
		self.Wf = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Wj = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Wc = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Wo = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uf = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uj = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uc = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uo = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.bf = nn.Parameter(t.randn(hidden_feat[0]))
		self.bj = nn.Parameter(t.randn(hidden_feat[0]))
		self.bc = nn.Parameter(t.randn(hidden_feat[0]))
		self.bo = nn.Parameter(t.randn(hidden_feat[0]))
		self.h_init = t.randn(hidden_feat[0])
		self.c_init = t.randn(hidden_feat[0])

		# Edge classification parameters
		self.U = t.randn(2*hidden_feat[0], hidden_feat[1])

	def __call__(self, A=None, X=None, edges=None):
		return self.forward(A, X, edges)

	def forward(self, A=None, X=None, edges=None):
		if type(A)==list and type(X)==t.Tensor and type(edges)==t.Tensor:
			AX = self.compute_AX(A, X)
			edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
			edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		else:
			AX = self.AX
			edge_src_nodes = self.edge_src_nodes
			edge_trg_nodes = self.edge_trg_nodes

		Y = self.relu(t.matmul(AX, self.W))
		Z = self.LSTM(Y)

		Z_mat_edge_src_nodes = Z.reshape(-1, Z.shape[-1])[edge_src_nodes]
		Z_mat_edge_trg_nodes = Z.reshape(-1, Z.shape[-1])[edge_trg_nodes]
		Z_mat = t.cat((Z_mat_edge_src_nodes, Z_mat_edge_trg_nodes), dim=1)
		output = t.matmul(Z_mat, self.U)

		return output

	def compute_AX(self, A, X):
		AX = t.zeros(self.T, self.N, X.shape[-1])
		for k in range(len(A)):
			AX[k] = t.sparse.mm(A[k], X[k])
		return AX

	def LSTM(self, Y):
		c = self.c_init.repeat(self.N, 1)
		h = self.h_init.repeat(self.N, 1)
		Z = t.zeros(Y.shape)
		for time in range(Y.shape[0]):
			f = self.sigmoid(t.matmul(Y[time], self.Wf) + t.matmul(h, self.Uf) + self.bf.repeat(self.N, 1))
			j = self.sigmoid(t.matmul(Y[time], self.Wj) + t.matmul(h, self.Uj) + self.bj.repeat(self.N, 1))
			o = self.sigmoid(t.matmul(Y[time], self.Wo) + t.matmul(h, self.Uo) + self.bo.repeat(self.N, 1))
			ct = self.sigmoid(t.matmul(Y[time], self.Wc) + t.matmul(h, self.Uc) + self.bc.repeat(self.N, 1))
			c = j*ct + f*c
			h = o*self.tanh(c)
			Z[time] = h
		return Z
    
class WD_GCN_reg(nn.Module):
	def __init__(self, A, X, hidden_feat=[2,2]):
		super(WD_GCN_reg, self).__init__()
		self.A = A
		self.X = X
		self.T, self.N, _ = X.shape
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.tanh = t.nn.Tanh()
		self.sigmoid = t.nn.Sigmoid()
		self.relu = nn.ReLU(inplace=False)
		self.AX = self.compute_AX(A, X)
		self.lin1 = nn.Linear(hidden_feat[0], 1)

		# GCN parameters
		self.W = nn.Parameter(t.randn(X.shape[-1], hidden_feat[0]))

		# LSTM parameters
		self.Wf = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Wj = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Wc = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Wo = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uf = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uj = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uc = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.Uo = nn.Parameter(t.randn(hidden_feat[0], hidden_feat[0]))
		self.bf = nn.Parameter(t.randn(hidden_feat[0]))
		self.bj = nn.Parameter(t.randn(hidden_feat[0]))
		self.bc = nn.Parameter(t.randn(hidden_feat[0]))
		self.bo = nn.Parameter(t.randn(hidden_feat[0]))
		self.h_init = t.randn(hidden_feat[0])
		self.c_init = t.randn(hidden_feat[0])

		# Edge classification parameters
		self.U = t.randn(2*hidden_feat[0], hidden_feat[1])

	def __call__(self, A=None, X=None):
		return self.forward(A, X)

	def forward(self, A=None, X=None, edges=None):
		if type(A)==list and type(X)==t.Tensor and type(edges)==t.Tensor:
			AX = self.compute_AX(A, X)
		else:
			AX = self.AX

		Y = self.relu(t.matmul(AX, self.W))
		Z = self.LSTM(Y)

		output = self.lin1(Z)

		return output.squeeze(2)

	def compute_AX(self, A, X):
		AX = t.zeros(self.T, self.N, X.shape[-1])
		for k in range(len(A)):
			AX[k] = t.sparse.mm(A[k], X[k])
		return AX

	def LSTM(self, Y):
		c = self.c_init.repeat(self.N, 1)
		h = self.h_init.repeat(self.N, 1)
		Z = t.zeros(Y.shape)
		for time in range(Y.shape[0]):
			f = self.sigmoid(t.matmul(Y[time], self.Wf) + t.matmul(h, self.Uf) + self.bf.repeat(self.N, 1))
			j = self.sigmoid(t.matmul(Y[time], self.Wj) + t.matmul(h, self.Uj) + self.bj.repeat(self.N, 1))
			o = self.sigmoid(t.matmul(Y[time], self.Wo) + t.matmul(h, self.Uo) + self.bo.repeat(self.N, 1))
			ct = self.sigmoid(t.matmul(Y[time], self.Wc) + t.matmul(h, self.Uc) + self.bc.repeat(self.N, 1))
			c = j*ct + f*c
			h = o*self.tanh(c)
			Z[time] = h
		return Z