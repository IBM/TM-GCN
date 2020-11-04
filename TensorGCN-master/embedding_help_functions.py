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
import random
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import math
import os.path
import scipy.io as sio
from sklearn.metrics import average_precision_score
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
	def __init__(self, At, X, edges, M, hidden_feat=[2,2,2], condensed_W=False, use_Minv=True, apply_M_twice=False, apply_M_three_times=False, nonlin2="relu"):
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
			apply_M_twice 	: 	bool
				When use_Minv is set to false, this is used to still force the program to apply the M matrix a second time before going into the second layer. The idea here is that this may help improve the performance, since it corresponds to doing a time mixing twice.
			self.apply_M_three_times	:	bool
				If we want to apply M a final time in the 2-layer architechture before putting the signal into the classifier
			nonlin2			: 	str
				Set to either "relu", "leaky" or "selu" use a ReLU, leaky ReLU, and SELU as the nonlinearity in between layers.
		"""
		super(EmbeddingGCN2, self).__init__()
		self.At = At
		self.M = M
		self.use_Minv = use_Minv
		self.apply_M_twice = apply_M_twice
		self.apply_M_three_times = apply_M_three_times
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
		elif self.apply_M_twice:
			AtYt = self.compute_AtXt(self.At, Y)
			Z = t.matmul(AtYt, W2t)
			if self.apply_M_three_times:
				Z = t.matmul(self.M, Z.reshape(self.T, -1).double()).reshape(Z.size())
		else:
			AY = self.compute_AX(self.At, Y)
			Z = t.matmul(AY, W2t)

		Z_mat_edge_src_nodes = Z.reshape(-1, self.F[2])[edge_src_nodes]
		Z_mat_edge_trg_nodes = Z.reshape(-1, self.F[2])[edge_trg_nodes]
		Z_mat = t.cat((Z_mat_edge_src_nodes, Z_mat_edge_trg_nodes), dim=1)

		output = t.matmul(Z_mat.float(), self.U)

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

# This function is used to construct augmented edge dataset for link prediction
def augment_edges(edges, N, beta1, beta2, cutoff):
	edges_t = edges.transpose(1,0)
	edges_new = []
	for j in range(t.max(edges[0])+1):
		if j < cutoff:
			beta = beta1
		else:
			beta = beta2
		to_add = beta*t.sum(edges[0]==j)
		n_added = 0
		edges_subset = edges[1:3, edges[0]==j]
		while n_added < to_add:
			e = [random.randint(0,N-1), random.randint(0,N-1)]
			if t.max(t.sum(edges_subset.transpose(1,0) == t.tensor(e), 1)) < 2:
				edges_new.append([j, e[0], e[1]])
				n_added += 1
		print(j)

	edges_aug = t.cat((edges, t.tensor(edges_new).transpose(1,0)), 1)
	_, sort_id = edges_aug[0].sort()
	edges_aug = edges_aug[:, sort_id]
	edges_aug_t = edges_aug.transpose(1,0)

	labels = t.cat((t.zeros(edges.shape[1], dtype=t.long), t.ones(edges_aug.shape[1]-edges.shape[1], dtype=t.long)), 0)
	labels = labels[sort_id]

	return edges_aug, labels

# This function computes precision, recall and F1
# Makes experiment code more compact by putting this in a separate function rather than repeating in every experiment script. Note 0 is assumed to be minority class, ie positive class we're trying to identify.
def compute_f1(guess, target):
	tp 			= t.sum((guess==0)&(target==0), dtype=t.float64) # true positive
	fp 			= t.sum((guess==0)&(target!=0), dtype=t.float64) # false positive
	fn 			= t.sum((guess!=0)&(target==0), dtype=t.float64) # false negative
	precision 	= tp/(tp+fp)
	recall 		= tp/(tp+fn)
	f1 			= 2*(precision*recall)/(precision + recall)

	return precision, recall, f1

# This function loads the preprocessed data from a mat file, and returns the
# various quantities that will be used later.
def load_data(data_loc, mat_f_name, S_train, S_val, S_test, transformed):
	# Load stuff from mat file
	saved_content = sio.loadmat(data_loc + mat_f_name)
	T = np.max(saved_content["A_labels_subs"][:,0])
	N1 = np.max(saved_content["A_labels_subs"][:,1])
	N2 = np.max(saved_content["A_labels_subs"][:,2])
	N = max(N1, N2)
	A_sz = t.Size([T, N, N])
	C_sz = t.Size([S_train, N, N])
	A_labels = t.sparse.FloatTensor(t.tensor(np.array(saved_content["A_labels_subs"].transpose(1,0), dtype=int) - 1, dtype=t.long), sq(t.tensor(saved_content["A_labels_vals"])), A_sz).coalesce()
	A = t.sparse.FloatTensor(A_labels._indices(), t.ones(A_labels._values().shape), A_sz).coalesce()

	if transformed:
		Ct_train = t.sparse.FloatTensor(t.tensor(np.array(saved_content["Ct_train_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["Ct_train_vals"])), C_sz).coalesce()
		Ct_val = t.sparse.FloatTensor(t.tensor(np.array(saved_content["Ct_val_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["Ct_val_vals"])), C_sz).coalesce()
		Ct_test = t.sparse.FloatTensor(t.tensor(np.array(saved_content["Ct_test_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["Ct_test_vals"])), C_sz).coalesce()
		M = t.tensor(saved_content["M"], dtype=t.float64)

		# Turn each Ct_train, Ct_val and Ct_test into a list of sparse matrices so that we can use them in matrix multiplication...
		Ct_train_2 = []
		for j in range(S_train):
			idx = Ct_train._indices()[0] == j
			Ct_train_2.append(t.sparse.FloatTensor(Ct_train._indices()[1:3,idx], Ct_train._values()[idx]))	
		Ct_val_2 = []
		for j in range(S_train):
			idx = Ct_val._indices()[0] == j
			Ct_val_2.append(t.sparse.FloatTensor(Ct_val._indices()[1:3,idx], Ct_val._values()[idx]))
		Ct_test_2 = []
		for j in range(S_train):
			idx = Ct_test._indices()[0] == j
			Ct_test_2.append(t.sparse.FloatTensor(Ct_test._indices()[1:3,idx], Ct_test._values()[idx]))

		return A, A_labels, Ct_train_2, Ct_val_2, Ct_test_2, N, M

	else:
		C = t.sparse.FloatTensor(t.tensor(np.array(saved_content["C_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["C_vals"])), t.Size([T,N,N])).coalesce()

		# Turn C_train, C_val and C_test into lists of sparse matrices so that we can use them in matrix multiplication...
		C_train = []
		for j in range(S_train):
			idx = C._indices()[0] == j
			C_train.append(t.sparse.FloatTensor(C._indices()[1:3,idx], C._values()[idx]))	
		C_val = []
		for j in range(S_train, S_train+S_val):
			idx = C._indices()[0] == j
			C_val.append(t.sparse.FloatTensor(C._indices()[1:3,idx], C._values()[idx]))
		C_test = []
		for j in range(S_train+S_val, S_train+S_val+S_test):
			idx = C._indices()[0] == j
			C_test.append(t.sparse.FloatTensor(C._indices()[1:3,idx], C._values()[idx]))

		return A, A_labels, C_train, C_val, C_test, N

# Create node features for training, validation and test data.
# Set same_block_size to true for TensorGCN methods, and false for all other methods
def create_node_features(A, S_train, S_val, S_test, same_block_size):
	X = t.zeros(A.shape[0], A.shape[1], 2)
	X[:, :, 0] = t.sparse.sum(A, 1).to_dense()
	X[:, :, 1] = t.sparse.sum(A, 2).to_dense()
	X_train = X[0:S_train].double()
	if same_block_size:
		X_val = X[S_val:S_train+S_val].double()
		X_test = X[S_val+S_test:].double()
	else:
		X_val = X[S_train:S_train+S_val].double()
		X_test = X[S_train+S_val:].double()

	return X_train, X_val, X_test

# Divide adjacency matrices and labels into training, validation and testing sets
def split_data(edges_aug, labels, S_train, S_val, S_test, same_block_size):
	# 	Training
	subs_train = edges_aug[0]<S_train
	edges_train = edges_aug[:, subs_train]
	target_train = labels[subs_train]
	e_train = edges_train[:, edges_train[0]!=0]
	e_train = e_train-t.cat((t.ones(1,e_train.shape[1]), t.zeros(2,e_train.shape[1])),0).long()

	#	Validation
	if same_block_size:
		subs_val = (edges_aug[0]>=S_val) & (edges_aug[0]<S_train+S_val)
	else:
		subs_val = (edges_aug[0]>=S_train) & (edges_aug[0]<S_train+S_val)
	edges_val = edges_aug[:, subs_val]
	if same_block_size:
		edges_val[0] -= S_val
	else:
		edges_val[0] -= S_train
	target_val = labels[subs_val]
	if same_block_size:
		K_val = t.sum(edges_val[0] - (S_train-S_val-1) > 0)
	e_val = edges_val[:, edges_val[0]!=0]
	e_val = e_val-t.cat((t.ones(1,e_val.shape[1]), t.zeros(2,e_val.shape[1])),0).long()

	#	Testing
	if same_block_size:
		subs_test = (edges_aug[0]>=S_test+S_val) 
	else:
		subs_test = (edges_aug[0]>=S_train+S_val)
	edges_test = edges_aug[:, subs_test]
	if same_block_size:
		edges_test[0] -= (S_test+S_val)
	else:
		edges_test[0] -= (S_train+S_val)
	target_test = labels[subs_test]
	if same_block_size:
		K_test = t.sum(edges_test[0] - (S_train-S_test-1) > 0)
	e_test = edges_test[:, edges_test[0]!=0]
	e_test = e_test-t.cat((t.ones(1,e_test.shape[1]), t.zeros(2,e_test.shape[1])),0).long()

	if same_block_size:
		return edges_train, target_train, e_train, edges_val, target_val, e_val, K_val, edges_test, target_test, e_test, K_test
	else:
		return edges_train, target_train, e_train, edges_val, target_val, e_val, edges_test, target_test, e_test

# Used to print results
def print_f1(precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test, alpha=None, tr=None, ep=None,  is_final=False):
	if is_final:
		print("FINAL: Train precision/recall/f1 %.16f/%.16f/%.16f. Train loss %.16f." % (precision_train, recall_train, f1_train, loss_train))
		print("FINAL: Val precision/recall/f1 %.16f/%.16f/%.16f. Val loss %.16f." % (precision_val, recall_val, f1_val, loss_val))
		print("FINAL: Test precision/recall/f1 %.16f/%.16f/%.16f. Test loss %.16f.\n" % (precision_test, recall_test, f1_test, loss_test))
	else:
		print("alpha/Tr/Ep %.2f/%d/%d. Train precision/recall/f1 %.16f/%.16f/%.16f. Train loss %.16f." % (alpha, tr, ep, precision_train, recall_train, f1_train, loss_train))
		print("alpha/Tr/Ep %.2f/%d/%d. Val precision/recall/f1 %.16f/%.16f/%.16f. Val loss %.16f." % (alpha, tr, ep, precision_val, recall_val, f1_val, loss_val))
		print("alpha/Tr/Ep %.2f/%d/%d. Test precision/recall/f1 %.16f/%.16f/%.16f. Test loss %.16f.\n" % (alpha, tr, ep, precision_test, recall_test, f1_test, loss_test))

# get_row_MRR copied from EvolveGCN code
def get_row_MRR(probs,true_classes):
    existing_mask = true_classes == 0 # Since 0 is our minority class, i.e., existing edges
    #descending in probability
    ordered_indices = np.flip(probs.argsort())

    ordered_existing_mask = existing_mask[ordered_indices]

    existing_ranks = np.arange(1,
                               true_classes.shape[0]+1,
                               dtype=np.float)[ordered_existing_mask]

    MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
    return MRR

# get_MRR copied from EvolveGCN code
def get_MRR(predictions,true_classes, adj ,do_softmax=True):
    if do_softmax:
        probs = t.softmax(predictions,dim=1)[:,0] # Send in probabilities for 0 class, i.e., minority class, i.e., existing edges
    else:
        probs = predictions[:,0]
    probs = probs.detach().cpu().numpy()
    true_classes = true_classes.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()
    pred_matrix = coo_matrix((probs,(adj[0],adj[1]))).toarray()
    true_matrix = coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

    row_MRRs = []
    for i,pred_row in enumerate(pred_matrix):
        if np.isin(1,true_matrix[i]):
            row_MRRs.append(get_row_MRR(pred_row,true_matrix[i]))

    avg_MRR = t.tensor(row_MRRs).mean()
    return avg_MRR

# get_MAP copied from EvolveGCN code
def get_MAP(predictions, true_classes, do_softmax=True):
    if do_softmax:
        probs = t.softmax(predictions,dim=1)[:,0] # Send in probabilities for 0 class, i.e., minority class, i.e., existing edges
    else:
        probs = predictions
    predictions_np = probs.detach().cpu().numpy()
    true_classes_np = true_classes.detach().cpu().numpy()
    return average_precision_score(true_classes_np, predictions_np, pos_label=0) # Since 0 is label we care about, i.e., existing edges

# This function compute MAP and MRR, via EvolveGCN support functions below
def compute_MAP_MRR(output, target, edges, do_softmax=True):
    MAP = 0.0
    MRR = 0.0
    # Compute MAP/MRR for each time slice and do weighted average
    for k in edges[0].unique():
        edges_mask = edges[0] == k
        w = t.sum(edges_mask).double()/t.tensor(len(edges_mask)).double()
        adj = edges[1:3, edges_mask]
        predictions = output[edges_mask, :]
        true_classes = target[edges_mask]
        MAP_slice = get_MAP(predictions, true_classes, True)
        MRR_slice = get_MRR(predictions, true_classes, adj, False)
        MAP += MAP_slice*w
        MRR += MRR_slice*w

    return MAP, MRR
