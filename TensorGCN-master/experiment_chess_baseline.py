# This version of bitcoin experiment imports data preprocessed in Matlab, and uses the GCN baseline

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
dataset = "chess"
no_epochs = 10000
mat_f_name = "saved_content_chess.mat"
no_trials = 1

data_loc = "data/" + dataset + "/"
save_res_fname = "results_BASELINE_layers" + str(no_layers) + "_w" + str(round(float(class_weights[0])*100)) + "_" + dataset
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
C = t.sparse.FloatTensor(t.tensor(np.array(saved_content["C_subs"].transpose(1,0), dtype=int), dtype=t.long) - 1, sq(t.tensor(saved_content["C_vals"])), t.Size([T,N,N])).coalesce()

A = t.sparse.FloatTensor(A_labels._indices(), t.ones(A_labels._values().shape), A_sz).coalesce()

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

# Create features for the nodes
X = t.zeros(A.shape[0], A.shape[1], 2)
X[:, :, 0] = t.sparse.sum(A, 1).to_dense()
X[:, :, 1] = t.sparse.sum(A, 2).to_dense()
X_train = X[0:S_train].double()
X_val = X[S_train:S_train+S_val].double()
X_test = X[S_train+S_val:].double()

# Divide adjacency matrices and labels into training, validation and testing sets
# 	Training
subs_train = A_labels._indices()[0]<S_train
edges_train = A_labels._indices()[:, subs_train]
labels_train = A_labels._values()[subs_train]
target_train = (t.sign(labels_train)+1).long() # 0 black win, 1 draw, 2 white win

#	Validation
subs_val = (A_labels._indices()[0]>=S_train) & (A_labels._indices()[0]<S_train+S_val)
edges_val = A_labels._indices()[:, subs_val]
edges_val[0] -= S_train
labels_val = A_labels._values()[subs_val]
target_val = (t.sign(labels_val)+1).long()

#	Testing
subs_test = (A_labels._indices()[0]>=S_train+S_val) 
edges_test = A_labels._indices()[:, subs_test]
edges_test[0] -= (S_train+S_val)
labels_test = A_labels._values()[subs_test]
target_test = (t.sign(labels_test)+1).long()

if no_trials > 1:
	ep_acc_loss_vec = []

for tr in range(no_trials):
	# Create gcn for training
	if no_layers == 2:
		gcn = ehf.EmbeddingKWGCN(C_train, X_train, edges_train, [6,6,3], nonlin2="selu")
	elif no_layers == 1:
		gcn = ehf.EmbeddingKWGCN(C_train, X_train, edges_train, [6,3])

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
				output_val = gcn(C_val, X_val, edges_val)
				guess_val = t.argmax(output_val, dim=1)
				accuracy_val = int(t.sum(guess_val==target_val, dtype=t.float64))/len(guess_val)
				loss_val = criterion(output_val, target_val)
				
				# Compute stats for test data
				output_test = gcn(C_test, X_test, edges_test)
				guess_test = t.argmax(output_test, dim=1)
				accuracy_test = int(t.sum(guess_test==target_test, dtype=t.float64))/len(guess_test)
				loss_test = criterion(output_test, target_test)

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