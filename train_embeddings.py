import copy
import h5py
import numpy as np

# Load the PPMI matrix
input_path = 'blog_dataset.h5'
input_file = h5py.File(input_path, 'r')
ppmi = input_file['ppmi']

# Unmodifiable constants
vocab_size = ppmi.shape[0]
timepoints = range(ppmi.shape[-1])

# Modifiable hyper-parameters
num_iter = 5
lmb = 10
gam = 100
tau = 50
dim = 50
batch_size = int(vocab_size / 4)

# Initialize output file name
file_path = 'embeddings-' + str(num_iter) + 'iter'


# Returns batch information for training the vocabulary set
# Inputs
#   num_vocabs: size of the vocabulary set
#   num_batch: size of each batch
# Outputs
#   batch_output: a list of tuples that contains inclusive start and exclusive end points
#                 for every batch
def return_batch(num_vocabs, num_batch):
    if num_batch == num_vocabs:
        return [(0, num_vocabs)]
    batch_output = [(idx * num_batch, (idx + 1) * num_batch) for idx in range(int(num_vocabs / num_batch))]
    return batch_output


# Modifies word embeddings using BCD (Batch Coordinate Descent)
# Inputs
#   ppmi_mat: a batch of PPMI matrix of size B x V where B is the batch size and V is
#             the size of the vocabulary
#   w_mat: co-optimizing matrix of size V x D
#   u_prev_: optimizing matrix in the previous time step, of size B x D
#   u_next_: optimizing matrix in the next time step, of size B x D
#   start_idx: beginning index of the batch
#   finish_idx: ending index of the batch
#   timestep: the time point at which the batch embedding is trained
# Outputs
#   u_mat: optimizing matrix batch of size B x D
def train(ppmi_mat, w_mat, u_prev_, u_next_, start_idx, finish_idx, timestep):
    w_square = np.dot(w_mat.T, w_mat)
    const = 1 if timestep == 0 or timestep == len(timepoints) - 1 else 2
    a_mat = w_square + (gam + lmb + const * tau) * np.eye(dim)
    b_mat = np.dot(ppmi_mat, w_mat) + gam * w_mat[start_idx:finish_idx, :] + tau * (u_prev_ + u_next_)
    u_mat = np.linalg.lstsq(a_mat.T, b_mat.T, rcond=None)[0]
    return u_mat.T


# Main method for the file
if __name__ == '__main__':

    # Initialize arrays
    output_file = h5py.File(file_path + '.h5', 'w')
    u_array_slice = np.random.randn(vocab_size, dim) / np.sqrt(dim)
    u_array = np.array([copy.deepcopy(u_array_slice) for _ in timepoints])
    u_array = output_file.create_dataset('U', data=u_array, chunks=True)
    w_array_slice = np.random.randn(vocab_size, dim) / np.sqrt(dim)
    w_array = np.array([copy.deepcopy(w_array_slice) for _ in timepoints])
    w_array = output_file.create_dataset('W', data=w_array, chunks=True)

    # Obtain batch indices
    batches = return_batch(vocab_size, batch_size)

    # Iterate and train word embeddings
    for iter_idx in range(num_iter):
        print('==========ITERATION %d==========' % (iter_idx + 1))
        timepoints_train = np.random.permutation(timepoints) if iter_idx > 0 else timepoints
        # Iterate over shuffled timepoints
        for enum, t in enumerate(timepoints_train):
            print('Timepoint: %d' % (enum + 1))
            # Iterate over separate batches
            for batch in batches:
                start, end = batch[0], batch[1]
                # Extract the current batch of PPMI matrix
                ppmi_batch = None
                try:
                    ppmi_batch = ppmi[start:end, :, t]
                except MemoryError:
                    print('Please use a smaller batch size than %d' % batch_size)
                # Extract U and W matrices in current time step
                u_curr = np.array(u_array[t, :, :])
                w_curr = np.array(w_array[t, :, :])
                # Obtain U and W matrices in previous time step
                if t == 0:
                    u_prev = np.zeros((batch_size, dim))
                    w_prev = np.zeros((batch_size, dim))
                else:
                    u_prev = u_array[t - 1, start:end, :]
                    w_prev = w_array[t - 1, start:end, :]
                # Obtain U and W matrices in subsequent time step
                if t == len(timepoints) - 1:
                    u_next = np.zeros((batch_size, dim))
                    w_next = np.zeros((batch_size, dim))
                else:
                    u_next = u_array[t + 1, start:end, :]
                    w_next = w_array[t + 1, start:end, :]
                # Train the U and W matrices
                w_array[t, start:end, :] = train(ppmi_batch, u_curr, w_prev, w_next, start, end, t)
                u_array[t, start:end, :] = train(ppmi_batch, w_curr, u_prev, u_next, start, end, t)

    # Indicate that the training step has completed
    print('Training completed')
