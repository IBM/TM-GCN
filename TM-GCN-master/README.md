# TensorGCN
PyTorch implementation of TensorGCN

## Files and what they do:
- **embedding_help_functions.py**: Implementation of our proposed TensorGCN as well as the GCN baseline.
- **evolvegcn_functions.py**: Implementation of EvolveGCN-H.
- **wd_gcn_functions.py**: Implementation of WD-GCN.
- **read_data.m**: Read data from CSV files, preprocess it, and save in a file format that later can be imported into Python.
- **experiment_xyz_baseline.py**: Used to run baseline GCN experiment on dataset xyz.
- **experiment_xyz_evolvegcn.py**: Used to run EvolveGCN-H experiment on dataset xyz.
- **experiment_xyz_our.py**: Used to run our proposed TensorGCN method on dataset xyz.
- **experiment_xyz_wd-gcn.py**: Used to run WD-GCN on dataset xyz.