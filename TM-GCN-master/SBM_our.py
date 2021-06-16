# This version of bitcoin experiment imports data preprocessed in Matlab, and uses our TensorGCN
# The point of this script is to do link prediction

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
import random
import networkx as nx
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
unsq = t.unsqueeze
sq = t.squeeze


# Settings
alpha_vec = [ .90]
no_layers = 1
dataset = "OTC" # OTC or Alpha
no_epochs = 100
no_trials = 1
beta1 = 19
beta2 = 19
cutoff = 95
loss_type = "softmax" # "sigmoid" or "softmax"
eval_type = "MAP-MRR" # "MAP-MRR" or "F1"

S_train, S_val, S_test = 35, 5, 10
lr = 0.01
momentum = 0.9
N = 1000
T = 50
no_diag = 20
datagen = 1

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

def load_data(S_train, S_val, S_test, no_diag, N, T):
    # Load stuff from mat file
    # Test with two communities
    community_num = 2
    # At each iteration migrate 10 nodes from one community to the another
    node_change_num = 10
    # Length of total time steps the graph will dynamically change
    dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(N,
                                                               community_num,
                                                               T,
                                                               1,  # comminity ID to perturb
                                                               node_change_num))
    graphs = [g[0] for g in dynamic_sbm_series]
    A_sz = t.Size([N, N,1])
    for tt in range(T):
        G = nx.adjacency_matrix(graphs[tt])
        ij = np.nonzero(G)
        vals = G[ij]
        ij = t.LongTensor(ij)
        # ij = t.cat((t.zeros((1,len(ij[0])),dtype=t.long), ij))
        vals  = t.DoubleTensor(vals).T
        tmp = t.sparse.DoubleTensor(ij, vals, A_sz)
        if tt==0:
            A = tmp
        else:
            A = t.cat((A,tmp),2)
    X = A.to_dense()
    X = X.transpose(2,0)        
    indices = t.nonzero(X).t()
    values = X[indices[0],indices[1],indices[2]] # modify this based on dimensionality
    A = t.sparse.DoubleTensor(indices, values, X.size())
   
    C_train = func_create_sparse(A, N, T, S_train, 0, S_train)#There is a bug in matlab Bitcoin Alpha
    C_val = func_create_sparse(A, N, T, S_train, S_val, S_train+S_val)
    C_test = func_create_sparse(A, N, T, S_train, S_val+S_test, T)
    
    M = create_matrix_M(S_train,no_diag)
    
    Ct_train = func_MProduct_dense(C_train, M)#There is a bug in matlab Bitcoin Alpha
    Ct_val = func_MProduct_dense(C_val, M)
    Ct_test = func_MProduct_dense(C_test, M)
    return A, Ct_train, Ct_val, Ct_test, M


if datagen == 1:
    A, Ct_train, Ct_val, Ct_test, M = load_data(S_train, S_val, S_test, no_diag, N, T)
    with open('SBM_data.pickle','wb') as f:
        pickle.dump([A, Ct_train, Ct_val, Ct_test, M], f)

else:
    with open('SBM_data.pickle','rb') as f:
        A, Ct_train, Ct_val, Ct_test, M = pickle.load(f)
# Load and return relevant data


# Create features for the nodes
X_train, X_val, X_test = ehf.create_node_features(A, S_train, S_val, S_test, same_block_size=True)

# Extract edges and labels from A_labels, and augment with nonexisting edges
# edges, beta
edges = A._indices()
edges_aug, labels = ehf.augment_edges(edges, N, beta1, beta2, cutoff)

# Divide adjacency matrices and labels into training, validation and testing sets
edges_train, target_train, e_train, edges_val, target_val, e_val, K_val, edges_test, target_test, e_test, K_test = ehf.split_data(edges_aug, labels, S_train, S_val, S_test, same_block_size=True)

if no_trials > 1:
	ep_acc_loss_vec = []

for tr in range(no_trials):
	for alpha in alpha_vec:
		class_weights = t.tensor([alpha, 1.0-alpha])
		save_res_fname = "results_OUR_layers" + str(no_layers) + "_w" + str(round(float(class_weights[0])*100)) + "_" + dataset + "_link_prediction"

		# Create gcn for training
		if loss_type == "softmax":
			n_output_feat = 2
		elif loss_type == "sigmoid":
			n_output_feat = 1	
		if no_layers == 2: 
			gcn = ehf.EmbeddingGCN2(Ct_train[:-1], X_train[:-1], e_train, M[:-1, :-1], hidden_feat=[6,6,n_output_feat], condensed_W=True, use_Minv=False, nonlin2="selu")
		elif no_layers == 1:
			gcn = ehf.EmbeddingGCN(Ct_train[:-1], X_train[:-1], e_train, M[:-1, :-1], hidden_feat=[6,n_output_feat], condensed_W=True, use_Minv=False)

		# Train
		optimizer = t.optim.SGD(gcn.parameters(), lr=lr, momentum=momentum)
		criterion = nn.CrossEntropyLoss(weight=class_weights) # Takes arguments (output, target)
		my_sig = nn.Sigmoid()
		if eval_type == "F1":
			ep_acc_loss = np.zeros((no_epochs,12)) # (precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test)
		elif eval_type == "MAP-MRR":
			ep_acc_loss = np.zeros((no_epochs,9)) # (MAP_train, MRR_train, loss_train, MAP_val, MRR_val, loss_val, MAP_test, MRR_test, loss_test)
			
		for ep in range(no_epochs):
			# Compute loss and take step
			optimizer.zero_grad()
			output_train = gcn()
			if loss_type == "sigmoid":
				p = my_sig(output_train)
				output_train = t.cat((p, 1-p), 1)
			loss_train = criterion(output_train, target_train[edges_train[0]!=0])
			loss_train.backward()
			optimizer.step()

			# Things that don't require gradient
			with t.no_grad():
				if ep % 100 == 0:
					# Compute stats for training data; no point in doing more often than this
					guess_train = t.argmax(output_train, dim=1)
					if eval_type == "F1":
						precision_train, recall_train, f1_train = ehf.compute_f1(guess_train, target_train[edges_train[0]!=0])
					elif eval_type == "MAP-MRR":
						MAP_train, MRR_train = ehf.compute_MAP_MRR(output_train, target_train[edges_train[0]!=0], edges_train[:, edges_train[0]!=0])

					# Compute stats for validation data
					output_val = gcn(Ct_val[:-1], X_val[:-1], e_val)
					if loss_type == "sigmoid":
						p = my_sig(output_val)
						output_val = t.cat((p, 1-p), 1)
					guess_val = t.argmax(output_val, dim=1)
					if eval_type == "F1":
						precision_val, recall_val, f1_val = ehf.compute_f1(guess_val[-K_val:], target_val[-K_val:])
					elif eval_type == "MAP-MRR":
						MAP_val, MRR_val = ehf.compute_MAP_MRR(output_val[-K_val:], target_val[-K_val:], edges_val[:, -K_val:])
					loss_val = criterion(output_val[-K_val:], target_val[-K_val:])

					# Compute stats for test data
					output_test = gcn(Ct_test[:-1], X_test[:-1], e_test)
					if loss_type == "sigmoid":
						p = my_sig(output_test)
						output_test = t.cat((p, 1-p), 1)
					guess_test = t.argmax(output_test, dim=1)
					if eval_type == "F1":
						precision_test, recall_test, f1_test = ehf.compute_f1(guess_test[-K_test:], target_test[-K_test:])
					elif eval_type == "MAP-MRR":
						MAP_test, MRR_test = ehf.compute_MAP_MRR(output_test[-K_test:], target_test[-K_test:], edges_test[:, -K_test:])
					loss_test = criterion(output_test[-K_test:], target_test[-K_test:])
					
					# Print
					if eval_type == "F1":
						ehf.print_f1(precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test, alpha, tr, ep)
					elif eval_type == "MAP-MRR":
						print("alpha/Tr/Ep %.2f/%d/%d. Train MAP/MRR %.16f/%.16f. Train loss %.16f." % (alpha, tr, ep, MAP_train, MRR_train, loss_train))
						print("alpha/Tr/Ep %.2f/%d/%d. Val MAP/MRR %.16f/%.16f. Val loss %.16f." % (alpha, tr, ep, MAP_val, MRR_val, loss_val))
						print("alpha/Tr/Ep %.2f/%d/%d. Test MAP/MRR %.16f/%.16f. Test loss %.16f.\n" % (alpha, tr, ep, MAP_test, MRR_test, loss_test))

				# Store values with results
				if eval_type == "F1":
					ep_acc_loss[ep] = [precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test]
				elif eval_type == "MAP-MRR":
					ep_acc_loss[ep] = [MAP_train, MRR_train, loss_train, MAP_val, MRR_val, loss_val, MAP_test, MRR_test, loss_test]

		if eval_type == "F1":
			ehf.print_f1(precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test, is_final=True)
		elif eval_type == "MAP-MRR":
			print("FINAL: Train MAP/MRR %.16f/%.16f. Train loss %.16f." % (MAP_train, MRR_train, loss_train))
			print("FINAL: Val MAP/MRR %.16f/%.16f. Val loss %.16f." % (MAP_val, MRR_val, loss_val))
			print("FINAL: Test MAP/MRR %.16f/%.16f. Test loss %.16f.\n" % (MAP_test, MRR_test, loss_test))

		if no_trials == 1:
			pickle.dump(ep_acc_loss, open(save_res_fname, "wb"))
			print("Results saved for single trial")
		else:
			ep_acc_loss_vec.append(ep_acc_loss)

if no_trials > 1:
	pickle.dump(ep_acc_loss_vec, open(save_res_fname + "_no_trials" + str(no_trials), "wb"))
	print("Results saved for all trials")