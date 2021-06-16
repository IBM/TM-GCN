# This version of bitcoin experiment imports data preprocessed in Matlab, and uses our TensorGCN

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
import scipy.io as sio
unsq = t.unsqueeze
sq = t.squeeze

# Settings
class_weights = t.tensor([.33, .33, .33]) # draw, black win, white win
no_layers = 1
no_epochs = 10000
dataset = "chess"
mat_f_name = "saved_content_chess.mat"
no_trials = 1

data_loc = "data/" + dataset + "/"
save_res_fname = "results_OUR_layers" + str(no_layers) + "_w" + str(round(float(class_weights[0])*100)) + "_" + dataset
S_train, S_val, S_test = 80, 10, 10
lr = 0.01
momentum = 0.9

# Load stuff from mat file
saved_content = sio.loadmat(data_loc + mat_f_name)
T = np.max(saved_content["tensor_idx"][:,0])
N1 = np.max(saved_content["tensor_idx"][:,1])
N2 = np.max(saved_content["tensor_idx"][:,2])
N = max(N1, N2)
A_sz = t.Size([T, N, N])
C_sz = t.Size([S_train, N, N])
A_labels = t.sparse.FloatTensor(t.tensor(np.array(saved_content["tensor_idx"].transpose(1,0), dtype=int) - 1, dtype=t.long), sq(t.tensor(saved_content["tensor_labels"])), A_sz).coalesce()
Ct_train = t.sparse.FloatTensor(t.tensor(np.array(saved_content["Ct_train_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["Ct_train_vals"])), C_sz).coalesce()
Ct_val = t.sparse.FloatTensor(t.tensor(np.array(saved_content["Ct_val_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["Ct_val_vals"])), C_sz).coalesce()
Ct_test = t.sparse.FloatTensor(t.tensor(np.array(saved_content["Ct_test_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["Ct_test_vals"])), C_sz).coalesce()
M = t.tensor(saved_content["M"], dtype=t.float64)

A = t.sparse.FloatTensor(A_labels._indices(), t.ones(A_labels._values().shape), A_sz).coalesce()

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

# Create features for the nodes
X = t.zeros(A.shape[0], A.shape[1], 2)
X[:, :, 0] = t.sparse.sum(A, 1).to_dense()
X[:, :, 1] = t.sparse.sum(A, 2).to_dense()
X_train = X[0:S_train].double()
X_val = X[S_val:S_train+S_val].double()
X_test = X[S_val+S_test:].double()

# Divide adjacency matrices and labels into training, validation and testing sets
# 	Training
subs_train = A_labels._indices()[0]<S_train
edges_train = A_labels._indices()[:, subs_train]
labels_train = A_labels._values()[subs_train]
target_train = (t.sign(labels_train)+1).long() # 0 black win, 1 draw, 2 white win

#	Validation
subs_val = (A_labels._indices()[0]>=S_val) & (A_labels._indices()[0]<S_train+S_val)
edges_val = A_labels._indices()[:, subs_val]
edges_val[0] -= S_val
labels_val = A_labels._values()[subs_val]
target_val = (t.sign(labels_val)+1).long()
evaluation_idx_val = edges_val[0] >= S_train-S_val

#	Testing
subs_test = (A_labels._indices()[0]>=S_test+S_val) 
edges_test = A_labels._indices()[:, subs_test]
edges_test[0] -= (S_test+S_val)
labels_test = A_labels._values()[subs_test]
target_test = (t.sign(labels_test)+1).long()
evaluation_idx_test = edges_test[0] >= S_train-S_test

if no_trials > 1:
	ep_acc_loss_vec = []

for tr in range(no_trials):
	# Create gcn for training
	if no_layers == 2: 
		gcn = ehf.EmbeddingGCN2(Ct_train_2, X_train, edges_train, M, hidden_feat=[6,6,3], condensed_W=True, use_Minv=False, nonlin2="selu")
	elif no_layers == 1:
		gcn = ehf.EmbeddingGCN(Ct_train_2, X_train, edges_train, M, hidden_feat=[6,3], condensed_W=True, use_Minv=False)

	# Train
	optimizer = t.optim.SGD(gcn.parameters(), lr=lr, momentum=momentum)
	criterion = nn.CrossEntropyLoss(weight=class_weights) # Takes arguments (output, target)
	ep_acc_loss = np.zeros((no_epochs,6)) # (accuracy_train, loss_train, accuracy_val, loss_val, accuracy_test, loss_test)
	for ep in range(no_epochs):
		# Compute loss and take step
		optimizer.zero_grad()
		output_train = gcn()
		loss_train = criterion(output_train, target_train)
		loss_train.backward()
		optimizer.step()

		# Things that don't require gradient
		with t.no_grad():
			guess_train = t.argmax(output_train, dim=1)
			accuracy_train = int(t.sum(guess_train==target_train, dtype=t.float64))/len(guess_train)
			if ep % 100 == 0:
				# Compute stats for validation data
				output_val = gcn(Ct_val_2, X_val, edges_val)
				guess_val = t.argmax(output_val, dim=1)
				accuracy_val = int(t.sum(guess_val[evaluation_idx_val]==target_val[evaluation_idx_val], dtype=t.float64))/len(guess_val[evaluation_idx_val])
				loss_val = criterion(output_val[evaluation_idx_val], target_val[evaluation_idx_val])

				# Compute stats for test data
				output_test = gcn(Ct_test_2, X_test, edges_test)
				guess_test = t.argmax(output_test, dim=1)
				accuracy_test = int(t.sum(guess_test[evaluation_idx_test]==target_test[evaluation_idx_test], dtype=t.float64))/len(guess_test[evaluation_idx_test])
				loss_test = criterion(output_test[evaluation_idx_test], target_test[evaluation_idx_test])
				
				# Print
				print("Tr/Ep %d/%d. Train accuracy %.16f. Train loss %.16f." % (tr, ep, accuracy_train, loss_train))
				print("Tr/Ep %d/%d. Val accuracy %.16f. Val loss %.16f." % (tr, ep, accuracy_val, loss_val))
				print("Tr/Ep %d/%d. Test accuracy %.16f. Test loss %.16f.\n" % (tr, ep, accuracy_test, loss_test))
			ep_acc_loss[ep] = [accuracy_train, loss_train, accuracy_val, loss_val, accuracy_test, loss_test]

	print("FINAL: Train accuracy %.16f. Train loss %.16f." % (accuracy_train, loss_train))
	print("FINAL: Val accuracy %.16f. Val loss %.16f." % (accuracy_val, loss_val))
	print("FINAL: Test accuracy %.16f. Test loss %.16f.\n" % (accuracy_test, loss_test))

	if no_trials == 1:
		pickle.dump(ep_acc_loss, open(save_res_fname, "wb"))
		print("Results saved for single trial")
	else:
		ep_acc_loss_vec.append(ep_acc_loss)

if no_trials > 1:
	pickle.dump(ep_acc_loss_vec, open(save_res_fname + "_no_trials" + str(no_trials), "wb"))
	print("Results saved for all trials")
