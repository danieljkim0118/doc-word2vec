from scipy.io import loadmat
import glob
import h5py
import numpy as np

# Check number of files in directory
path = './baseline'
num = len(glob.glob1(path, "*.mat"))

# Create new HDF file for storing baseline embeddings
baseline_file = h5py.File('baseline_embeddings.h5', 'w')
stacked_embeddings = np.zeros((num, 20936, 50))
stacked_embeddings = baseline_file.create_dataset('U', data=stacked_embeddings, chunks=True)

# Iterate over the directory and extract all embeddings
for idx in range(num):
    embedding = loadmat(path + '/embeddings_%d.mat' % idx)['U']
    embedding = np.array(embedding)
    stacked_embeddings[idx, :, :] = embedding

# Indicate that the baseline embedding has been extracted
baseline_file.close()
print('Baseline embedding extraction complete')
