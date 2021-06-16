# This version of uci experiment imports data preprocessed in Matlab, and uses EvolveGCN
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
import evolvegcn_functions as ef
import embedding_help_functions as ehf
import scipy.io as sio
unsq = t.unsqueeze
sq = t.squeeze

# Settings
alpha_vec = [.75, .76, .77, .78, .79, .80, .81, .82, .83, .84, .85, .86, .87, .88, .89, .90, .91, .92, .93, .94, .95]
no_layers = 1
dataset = "uci"
no_epochs = 1000
mat_f_name = "saved_content_uci.mat"
no_trials = 1
beta1 = 19
beta2 = 19
cutoff = 62
eval_type = "MAP-MRR" # "MAP-MRR" or "F1"

data_loc = "data/" + dataset + "/"
S_train, S_val, S_test = 62, 13, 13
lr = 0.01
momentum = 0.9

# Load and return relevant data
A, A_labels, C_train, C_val, C_test, N = ehf.load_data(data_loc, mat_f_name, S_train, S_val, S_test, transformed=False)

# Create features for the nodes
X_train, X_val, X_test = ehf.create_node_features(A, S_train, S_val, S_test, same_block_size=False)

# Extract edges and labels from A_labels, and augment with nonexisting edges
edges = A_labels._indices()
edges_aug, labels = ehf.augment_edges(edges, N, beta1, beta2, cutoff)

# Divide adjacency matrices and labels into training, validation and testing sets
edges_train, target_train, e_train, edges_val, target_val, e_val, edges_test, target_test, e_test = ehf.split_data(edges_aug, labels, S_train, S_val, S_test, same_block_size=False)

if no_trials > 1:
	ep_acc_loss_vec = []

for tr in range(no_trials):
	for alpha in alpha_vec:
		class_weights = t.tensor([alpha, 1.0-alpha])
		save_res_fname = "results_EVOLVEGCN_layers" + str(no_layers) + "_w" + str(round(float(class_weights[0])*100)) + "_" + dataset + "_link_prediction"

		# Create gcn for training
		if no_layers == 2:
			gcn = ef.EvolveGCN_2_layer(C_train[:-1], X_train[:-1], e_train, [6,6,2])
		elif no_layers == 1:
			gcn = ef.EvolveGCN_1_layer(C_train[:-1], X_train[:-1], e_train, [6,2])

		# Train
		optimizer = t.optim.SGD(gcn.parameters(), lr=lr, momentum=momentum)
		criterion = nn.CrossEntropyLoss(weight=class_weights) # Takes arguments (output, target)
		if eval_type == "F1":
			ep_acc_loss = np.zeros((no_epochs,12)) # (precision_train, recall_train, f1_train, loss_train, precision_val, recall_val, f1_val, loss_val, precision_test, recall_test, f1_test, loss_test)
		elif eval_type == "MAP-MRR":
			ep_acc_loss = np.zeros((no_epochs,9)) # (MAP_train, MRR_train, loss_train, MAP_val, MRR_val, loss_val, MAP_test, MRR_test, loss_test)

		for ep in range(no_epochs):
			# Compute loss and take step
			optimizer.zero_grad()
			if no_layers == 2:
				output_train, W_val, W2_val = gcn()
			elif no_layers == 1:
				output_train, W_val = gcn()
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
					if no_layers == 2:
						output_val, W_test, W2_test = gcn(C_val[:-1], X_val[:-1], e_val, W_val, W2_val)
					elif no_layers == 1:
						output_val, W_test = gcn(C_val[:-1], X_val[:-1], e_val, W_val)
					guess_val = t.argmax(output_val, dim=1)
					if eval_type == "F1":
						precision_val, recall_val, f1_val = ehf.compute_f1(guess_val, target_val[edges_val[0]!=0])
					elif eval_type == "MAP-MRR":
						MAP_val, MRR_val = ehf.compute_MAP_MRR(output_val, target_val[edges_val[0]!=0], edges_val[:, edges_val[0]!=0])
					loss_val = criterion(output_val, target_val[edges_val[0]!=0])
					
					# Compute stats for test data
					if no_layers == 2:
						output_test, _, _ = gcn(C_test[:-1], X_test[:-1], e_test, W_test, W2_test)
					elif no_layers == 1:
						output_test, _ = gcn(C_test[:-1], X_test[:-1], e_test, W_test)
					guess_test = t.argmax(output_test, dim=1)
					if eval_type == "F1":
						precision_test, recall_test, f1_test = ehf.compute_f1(guess_test, target_test[edges_test[0]!=0])
					elif eval_type == "MAP-MRR":
						MAP_test, MRR_test = ehf.compute_MAP_MRR(output_test, target_test[edges_test[0]!=0], edges_test[:, edges_test[0]!=0])
					loss_test = criterion(output_test, target_test[edges_test[0]!=0])

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
	