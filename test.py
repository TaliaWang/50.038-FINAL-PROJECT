import h5py
import numpy as np

### context features ###
f = h5py.File('./MUStARD/data/features/context_final/resnet_pool5.hdf5', 'r')

# each key represents an instance (example) corresponding to the same entry(?) in sarcasm_data.json
# print(list(f.keys()))

# # hdf5 dataset object
# print(f['1_507'])

# numpy array for dataset 1_507
# print(f['1_507'][()])

# array shape for dataset 1_507
# print(np.shape(f['1_507'][()]))print(np.shape(f['1_507'][()]))

# #### utterance features - similar to above ###
# f = h5py.File('./MUStARD/data/features/context_final/resnet_pool5.hdf5', 'r')
# print(f['1_507'])
# print(f['1_507'][()])
# print(np.shape(f['1_507'][()]))

### MNIST DATASET ###
f = h5py.File('./MUStARD/data/features/context_final/resnet_pool5.hdf5', 'r')
# print(list(f.keys()))

# TODO: QUESTIONS
# What does each entry in the numpy array represent?
# Is each hdf5 object an example?