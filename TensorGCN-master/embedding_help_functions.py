# This file contains help functions for TensorGCN

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

# Function for computing At
# Presently, I don't think this function is used. Can probably be removed in the future.
def compute_At(fname_At_mat, fname_ij_matT, A, M, normalization_type=0):
	T = A.shape[0]
	N = A.shape[1]
	sz = t.Size([T, N, N])
	# START OF IF {
	if (not os.path.isfile(fname_At_mat)) and (not os.path.isfile(fname_ij_matT)):
		print("Files <<" + fname_At_mat + ">> and <<" + fname_ij_matT + ">> do not exist. Creating them... (this will take time)")

		if normalization_type == 0:
			# Normalize adjacency matrices, i.e., add identity to each frontal
			# slice and then divide each column fiber by its 1-norm.
			self_loop_idx = t.zeros(3, T*N).long()
			self_loop_idx[0] = t.tensor(np.repeat(np.arange(0,T), N)).long()
			self_loop_idx[1] = t.tensor(np.tile(np.arange(0,N), T)).long()
			self_loop_idx[2] = self_loop_idx[1]
			self_loop_vals = t.ones(T*N)
			A = A + t.sparse.FloatTensor(self_loop_idx, self_loop_vals, sz)
			A = A.coalesce()
			A_col_sum = t.sparse.sum(A, dim=1)
			AT = A.transpose(1,2).coalesce() # flip row and col, so that row index is last
			norm_idx = 0
			k_old = AT._indices()[0][0]
			j_old = AT._indices()[1][0]
			for nnz_idx in range(AT._nnz()):
				k = AT._indices()[0][nnz_idx]
				j = AT._indices()[1][nnz_idx]
				if not (k == k_old and j == j_old):
					norm_idx += 1
					k_old = k
					j_old = j
				AT._values()[nnz_idx] /= A_col_sum._values()[norm_idx]
			A = AT.transpose(1,2) # Now A has all column sums equal to 1

		elif normalization_type == 1:
			# Normalize adjacency matrices
			A = A + A.transpose(1,2)
			A = A/2
			self_loop_idx = t.zeros(3, T*N).long()
			self_loop_idx[0] = t.tensor(np.repeat(np.arange(0,T), N)).long()
			self_loop_idx[1] = t.tensor(np.tile(np.arange(0,N), T)).long()
			self_loop_idx[2] = self_loop_idx[1]
			self_loop_vals = t.ones(T*N)
			A = A + t.sparse.FloatTensor(self_loop_idx, self_loop_vals, sz)
			A = A.coalesce()
			A_row_sum = t.sparse.sum(A, dim=2)

			# Multiply with D^{-1/2} from the right
			AT = A.transpose(1,2).coalesce()
			norm_idx = 0
			k_old = AT._indices()[0][0]
			j_old = AT._indices()[1][0]
			for nnz_idx in range(AT._nnz()):
				k = AT._indices()[0][nnz_idx]
				j = AT._indices()[1][nnz_idx]
				if not (k == k_old and j == j_old):
					norm_idx += 1
					k_old = k
					j_old = j
				AT._values()[nnz_idx] /= math.sqrt(A_row_sum._values()[norm_idx])
			A = AT.transpose(1,2).coalesce() # Now A has all column sums equal to 1

			# Multiply with D^{-1/2} from the left
			norm_idx = 0
			k_old = A._indices()[0][0]
			i_old = A._indices()[1][0]
			for nnz_idx in range(A._nnz()):
				k = A._indices()[0][nnz_idx]
				i = A._indices()[1][nnz_idx]
				if not (k == k_old and i == i_old):
					norm_idx += 1
					k_old = k
					i_old = i
				A._values()[nnz_idx] /= math.sqrt(A_row_sum._values()[norm_idx])			
		
		# Compute the tubes of the M-transform of A
		nnzcol = t.sparse.sum(A, dim=0)._nnz()
		At_matT = t.zeros(nnzcol, T)
		ij_mat = t.zeros(nnzcol, 2).long()
		AT = A.transpose(0,2).coalesce()
		valT = AT._values()
		idxT = AT._indices()
		i_old = idxT[1][0]
		j_old = idxT[0][0]
		vec = t.zeros(T)
		cnt = 0
		MT = M.transpose(1,0) # Note: This should be transpose of M, NOT inverse of M
		for c in range(idxT.shape[1]):
			j = idxT[0][c]
			i = idxT[1][c]
			k = idxT[2][c]
			if i == i_old and j == j_old:
				vec += valT[c]*MT[k]
			else:
				At_matT[cnt] = vec
				ij_mat[cnt] = t.tensor([i_old, j_old]).long()
				cnt += 1
				vec = valT[c]*MT[k]
				i_old = i
				j_old = j
			if c % 10000 == 0:
				print(c)
		At_matT[cnt] = vec
		ij_mat[cnt] = t.tensor([i_old, j_old]).long()

		# Compute a list containing the frontal matrices of At
		ij_matT = ij_mat.transpose(1,0)
		At_mat = At_matT.transpose(1,0)

		print("Saving to files <<" + fname_ij_matT + ">> and <<" + fname_At_mat + ">>.")
		pickle.dump(At_mat, open(fname_At_mat, "wb"))
		pickle.dump(ij_matT, open(fname_ij_matT, "wb"))

	# } END OF IF
	# START OF ELSE {
	else:
		print("Loading files <<" + fname_ij_matT + ">> and <<" + fname_At_mat + ">> from file...")
		ij_matT = pickle.load(open(fname_ij_matT, "rb"))
		At_mat = pickle.load(open(fname_At_mat, "rb"))
	# } END OF ELSE

	At = []
	for vl in At_mat:
		At.append(t.sparse.FloatTensor(ij_matT, vl))

	return At
# END OF FUNCTION compute_At
# I have verified this function, and made some fixes, on 20-July-2019 

class EmbeddingGCN(nn.Module):
	"""
	Our proposed TensorGCN with 1 layer
	"""
	def __init__(self, At, X, edges, M, hidden_feat=[2,2], condensed_W=False, use_Minv=True):
		"""
		Initialize EmbeddingGCN layer

		Parameters:
			At				:	torch.Tensor
				A tensor containing the M-transformed version of the normalized graph Laplacian. It should be of size T x N x N, where T is the number of time steps, and N is the number of nodes.
			X				:	torch.Tensor
				A tensor of size T x N x 2 which contains the node signals. T is time, and N is the number of nodes. The size 2 third dimension comes from the fact that, in this experiment, each node has 2 features.
			edges			:	torch.Tensor, dtype=t.long
				A matrix of size 3 x no_edges which contains information on the edges. Specifically, each column of edges has three entries: time slice, source node and target node.
			M				: 	torch.Tensor
				A matrix of size T x T, where T is the number of time steps. M is assumed to be invertible.
			hidden_feat 	:	list of int
				The number of hidden layers to utilize. More specifically, this is the number of features for each node that the GCN outputs. Should be list of 2 integers.
			condensed_W		:	bool
				Set to True to use condensed weight tensor, i.e., use the same weight matrix for each time point.
			use_Minv		:	bool
				Set to False to avoid every applying the inverse M transform.
		"""
		super(EmbeddingGCN, self).__init__()
		self.M = M
		self.use_Minv = use_Minv
		if use_Minv:
			self.Minv = t.tensor(np.linalg.inv(M))
		self.T = X.shape[0]
		self.N = X.shape[1]
		self.F = [X.shape[-1]] + hidden_feat
		if condensed_W:
			self.W = nn.Parameter(t.randn(self.F[0], self.F[1]))
		else:
			self.W = nn.Parameter(t.randn(self.T, self.F[0], self.F[1]))
		self.U = nn.Parameter(t.randn(2*self.F[1], self.F[2]))
		self.sigmoid = nn.Sigmoid()
		
		self.AtXt = self.compute_AtXt(At, X)
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
		self.edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)

	def __call__(self, At=None, X=None, edges=None):
		return self.forward(At, X, edges)

	def compute_AtXt(self, At, X):
		Xt = t.matmul(self.M, X.reshape(self.T, -1)).reshape(X.size())
		AtXt = t.zeros(self.T, self.N, self.F[0])
		for k in range(self.T):
			AtXt[k] = t.sparse.mm(At[k], Xt[k])
		return AtXt

	def forward(self, At=None, X=None, edges=None):
		# Either use existing AtXt and edges, or compute new
		if type(At)==list and type(X)==t.Tensor and type(edges)==t.Tensor:
			AtXt = self.compute_AtXt(At, X)
			edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
			edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		else:
			AtXt = self.AtXt
			edge_src_nodes = self.edge_src_nodes
			edge_trg_nodes = self.edge_trg_nodes
		
		Wt = self.W # Do not transform W
		AtXtWt = t.matmul(AtXt, Wt)
		if self.use_Minv:
			Y = t.matmul(self.Minv, AtXtWt.reshape(self.T, -1)).reshape(AtXtWt.size()) 
		else:
			Y = AtXtWt

		AXW_mat_edge_src_nodes = Y.reshape(-1, self.F[1])[edge_src_nodes]
		AXW_mat_edge_trg_nodes = Y.reshape(-1, self.F[1])[edge_trg_nodes]
		AXW_mat = t.cat((AXW_mat_edge_src_nodes, AXW_mat_edge_trg_nodes), dim=1)

		output = t.matmul(AXW_mat, self.U)

		return output

class EmbeddingGCN2(nn.Module):
	"""
	Our proposed TensorGCN with 2 layers
	"""
	def __init__(self, At, X, edges, M, hidden_feat=[2,2,2], condensed_W=False, use_Minv=True, nonlin2="relu"):
		"""
		Initialize EmbeddingGCN2 layer

		Parameters:
			At				:	torch.Tensor
				A tensor containing the M-transformed version of the normalized graph Laplacian. It should be of size T x N x N, where T is the number of time steps, and N is the number of nodes.
			X				:	torch.Tensor
				A tensor of size T x N x 2 which contains the node signals. T is time, and N is the number of nodes. The size 2 third dimension comes from the fact that, in this experiment, each node has 2 features.
			edges			:	torch.Tensor, dtype=t.long
				A matrix of size 3 x no_edges which contains information on the edges. Specifically, each column of edges has three entries: time slice, source node and target node.
			M				: 	torch.Tensor
				A matrix of size T x T, where T is the number of time steps. M is assumed to be invertible.
			hidden_feat 	:	list of int
				The number of hidden layers to utilize. More specifically, this is the number of features for each node that the GCN outputs. Should be list of 3 integers.
			condensed_W		:	bool
				Set to True to use condensed weight tensor, i.e., use the same weight matrix for each time point.
			use_Minv		:	bool
				Set to False to avoid every applying the inverse M transform.
			nonlin2			: 	str
				Set to either "relu", "leaky" or "selu" use a ReLU, leaky ReLU, and SELU as the nonlinearity in between layers.
		"""
		super(EmbeddingGCN2, self).__init__()
		self.At = At
		self.M = M
		self.use_Minv = use_Minv
		if use_Minv:
			self.Minv = t.tensor(np.linalg.inv(M))
		self.T = X.shape[0]
		self.N = X.shape[1]
		self.F = [X.shape[-1]] + hidden_feat
		if condensed_W:
			self.W1 = nn.Parameter(t.randn(self.F[0], self.F[1]))
			self.W2 = nn.Parameter(t.randn(self.F[1], self.F[2]))
		else:
			self.W1 = nn.Parameter(t.randn(self.T, self.F[0], self.F[1]))
			self.W2 = nn.Parameter(t.randn(self.T, self.F[1], self.F[2]))
		self.U = nn.Parameter(t.randn(self.F[2]*2,self.F[3]))
		if nonlin2 == "relu":
			self.nonlin_func = nn.ReLU(inplace=False)
		elif nonlin2 == "leaky":
			self.nonlin_func = nn.LeakyReLU(negative_slope=0.01, inplace=False)
		elif nonlin2 == "selu":
			self.nonlin_func = nn.SELU(inplace=False)

		self.sigmoid = nn.Sigmoid()
		
		self.AtXt = self.compute_AtXt(At, X)
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
		self.edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)

	def __call__(self, At=None, X=None, edges=None):
		return self.forward(At, X, edges)

	def compute_AX(self, A, X):
		AX = t.zeros(self.T, self.N, X.shape[-1])
		for k in range(self.T):
			AX[k] = t.sparse.mm(A[k], X[k])
		return AX

	def compute_AtXt(self, At, X):
		Xt = t.matmul(self.M, X.reshape(self.T, -1)).reshape(X.size())
		AtXt = t.zeros(self.T, self.N, X.shape[-1])
		for k in range(self.T):
			AtXt[k] = t.sparse.mm(At[k], Xt[k])
		return AtXt

	def forward(self, At=None, X=None, edges=None):
		# Either use existing AtXt and edges, or compute new
		if type(At)==list and type(X)==t.Tensor and type(edges)==t.Tensor:
			AtXt = self.compute_AtXt(At, X)
			edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
			edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		else:
			AtXt = self.AtXt
			edge_src_nodes = self.edge_src_nodes
			edge_trg_nodes = self.edge_trg_nodes

		# Do not transform weight tensors/matrices
		W1t = self.W1
		W2t = self.W2

		# First layer
		AtXtW1t = t.matmul(AtXt, W1t)
		if self.use_Minv:
			Y = self.nonlin_func(t.matmul(self.Minv, AtXtW1t.reshape(self.T, -1)).reshape(AtXtW1t.size())) 
		else:
			Y = self.nonlin_func(AtXtW1t)
		Y = Y.double()

		# Second layer
		if self.use_Minv:
			AtYt = self.compute_AtXt(self.At, Y)
			AtYtW2t = t.matmul(AtYt, W2t)
			Z = t.matmul(self.Minv, AtYtW2t.reshape(self.T, -1)).reshape(AtYtW2t.size())
		else:
			AY = self.compute_AX(self.At, Y)
			Z = t.matmul(AY, W2t)

		Z_mat_edge_src_nodes = Z.reshape(-1, self.F[2])[edge_src_nodes]
		Z_mat_edge_trg_nodes = Z.reshape(-1, self.F[2])[edge_trg_nodes]
		Z_mat = t.cat((Z_mat_edge_src_nodes, Z_mat_edge_trg_nodes), dim=1)

		output = t.matmul(Z_mat, self.U)

		return output

class EmbeddingKWGCN(nn.Module):
	"""
	Embedding implementation of the baseline GCN with 1 or 2 layers
	"""
	def __init__(self, A, X, edges, hidden_feat=[2,2], nonlin2="relu"):
		"""
		Initialize EmbeddingKWGCN layer

		Parameters:
			A				:	torch.Tensor
				A tensor containing the the normalized graph Laplacian. It should be of size T x N x N, where T is the number of time steps, and N is the number of nodes.
			X				:	torch.Tensor
				A tensor of size T x N x F which contains the node signals. T is time, N is the number of nodes, and F is the number of features.
			edges			:	torch.Tensor, dtype=t.long
				A matrix of size 3 x no_edges which contains information on the edges. Specifically, each column of edges has three entries: time slice, source node and target node.
			hidden_feat 	:	list of int
				The number of hidden layers to utilize. More specifically, this is the number of features for each node that the GCN outputs. Should be list of 2 or 3 integers.
			nonlin2			: 	str
				Set to either "relu", "leaky" or "selu" use a ReLU, leaky ReLU, and SELU as the nonlinearity in between layers.
		"""
		super(EmbeddingKWGCN, self).__init__()
		self.no_layers = len(hidden_feat)-1
		self.T = len(A)
		self.N = X.shape[1]
		self.A = A
		self.F = [X.shape[-1]] + hidden_feat
		if self.no_layers == 2:
			self.W2 = nn.Parameter(t.randn(self.F[1], self.F[2]))
		self.W1 = nn.Parameter(t.randn(self.F[0], self.F[1]))
		self.U = nn.Parameter(t.randn(self.F[-2]*2, self.F[-1]))
		if nonlin2 == "relu":
			self.nonlin_func = nn.ReLU(inplace=False)
		elif nonlin2 == "leaky":
			self.nonlin_func = nn.LeakyReLU(negative_slope=0.01, inplace=False)
		elif nonlin2 == "selu":
			self.nonlin_func = nn.SELU(inplace=False)
		self.v = t.tensor([self.N, 1], dtype=t.long)
		self.edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
		self.edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		self.AX = self.compute_AX(A, X)

	def __call__(self, A=None, X=None, edges=None):
		return self.forward(A, X, edges)

	def compute_AX(self, A, X):
		AX = t.zeros(self.T, self.N, X.shape[-1])
		for k in range(len(A)):
			AX[k] = t.sparse.mm(A[k], X[k])
		return AX

	def forward(self, A=None, X=None, edges=None):
		if type(A)==list and type(X)==t.Tensor and type(edges)==t.Tensor:
			AX = self.compute_AX(A, X)
			edge_src_nodes = t.matmul(edges[[0,1]].transpose(1,0), self.v)
			edge_trg_nodes = t.matmul(edges[[0,2]].transpose(1,0), self.v)
		else:
			AX = self.AX
			edge_src_nodes = self.edge_src_nodes
			edge_trg_nodes = self.edge_trg_nodes

		if self.no_layers == 2:
			Y = self.nonlin_func(t.matmul(AX, self.W1)).double()
			Z = t.matmul(self.compute_AX(self.A, Y), self.W2)
		else:
			Z = t.matmul(AX, self.W1)

		Z_mat_edge_src_nodes = Z.reshape(-1, Z.shape[-1])[edge_src_nodes]
		Z_mat_edge_trg_nodes = Z.reshape(-1, Z.shape[-1])[edge_trg_nodes]
		Z_mat = t.cat((Z_mat_edge_src_nodes, Z_mat_edge_trg_nodes), dim=1)

		output = t.matmul(Z_mat, self.U)

		return output
    
class EmbeddingGCN_reg(nn.Module):
	"""
	Our proposed TensorGCN with 1 layer
	"""
	def __init__(self, At, X, M, hidden_feat=[2,2], condensed_W=False, use_Minv=True):
		"""
		Initialize EmbeddingGCN layer

		Parameters:
			At				:	torch.Tensor
				A tensor containing the M-transformed version of the normalized graph Laplacian. It should be of size T x N x N, where T is the number of time steps, and N is the number of nodes.
			X				:	torch.Tensor
				A tensor of size T x N x 2 which contains the node signals. T is time, and N is the number of nodes. The size 2 third dimension comes from the fact that, in this experiment, each node has 2 features.
			edges			:	torch.Tensor, dtype=t.long
				A matrix of size 3 x no_edges which contains information on the edges. Specifically, each column of edges has three entries: time slice, source node and target node.
			M				: 	torch.Tensor
				A matrix of size T x T, where T is the number of time steps. M is assumed to be invertible.
			hidden_feat 	:	list of int
				The number of hidden layers to utilize. More specifically, this is the number of features for each node that the GCN outputs. Should be list of 2 integers.
			condensed_W		:	bool
				Set to True to use condensed weight tensor, i.e., use the same weight matrix for each time point.
			use_Minv		:	bool
				Set to False to avoid every applying the inverse M transform.
		"""
		super(EmbeddingGCN_reg, self).__init__()
		self.M = M
		self.use_Minv = use_Minv
		if use_Minv:
			self.Minv = t.tensor(np.linalg.inv(M))
		self.T = X.shape[0]
		self.N = X.shape[1]
		self.F = [X.shape[-1]] + hidden_feat
		if condensed_W:
			self.W = nn.Parameter(t.randn(self.F[0], self.F[1]))
		else:
			self.W = nn.Parameter(t.randn(self.T, self.F[0], self.F[1]))
		self.lin1 = nn.Linear(self.F[1], 1)
		self.sigmoid = nn.Sigmoid()
		
		self.AtXt = self.compute_AtXt(At, X)

	def __call__(self, At=None, X=None):
		return self.forward(At, X)

	def compute_AtXt(self, At, X):
		Xt = t.matmul(self.M, X.reshape(self.T, -1)).reshape(X.size())
		AtXt = t.zeros(self.T, self.N, self.F[0])
		for k in range(self.T):
			AtXt[k] = t.sparse.mm(At[k], Xt[k])
		return AtXt

	def forward(self, At=None, X=None):
		# Either use existing AtXt and edges, or compute new
		AtXt = self.AtXt
		
		Wt = self.W # Do not transform W
		AtXtWt = t.matmul(AtXt, Wt)
		if self.use_Minv:
			Y = t.matmul(self.Minv, AtXtWt.reshape(self.T, -1)).reshape(AtXtWt.size()) 
		else:
			Y = AtXtWt

		output = self.lin1(Y)

		return output.squeeze(2)