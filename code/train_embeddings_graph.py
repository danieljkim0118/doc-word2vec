import copy
import h5py
import math
import numpy as np

# Load the PPMI matrix
category = 'topic'
input_path = 'blog_dataset_%s.h5' % category
input_file = h5py.File(input_path, 'r')
ppmi = input_file['ppmi']
labels = input_file['label']
labels_copy = copy.deepcopy(np.asarray(labels))
use_human = False

# Load the topic adjacency matrix
path = 'topics_human.npy' if use_human else 'topics_ppmi_cosine.npy'
topic_matrix = np.load('data/%s.npy' % path)

# Unmodifiable constants
vocab_size = ppmi.shape[0]
indices = range(ppmi.shape[-1])

# Modifiable hyper-parameters
num_iter = 5
lmb = 10
gam = 100
tau = 30
dim = 50
batch_size = int(vocab_size)

# Initialize output file name
file_path = 'embeddings-%s-human-' % category + str(num_iter) + 'iter'


# Inverse sigmoid function
# Inputs
#   x: input value, can be a vector
#   scale: the scaling constant that decides the steepness of the output
# Outputs
#   y: the output value, can be broadcasted from a vector
def inverse_sigmoid(x, scale):
    alpha = 1 + math.exp(-1 * scale)
    y = 1/scale * np.log(x / (alpha - x))
    return y


# Inverse tanh function
# Inputs
#   x: input value, can be a vector
#   scale: the scaling constant that decides the steepness of the output
# Outputs
#   y: the output value, can be broadcasted from a vector
def inverse_tanh(x, scale):
    alpha = (math.exp(scale) + 1) / (math.exp(scale) - 1)
    y = 1/scale * np.log((alpha + x) / (alpha-x))
    return y


# Modifies the weights for training embeddings
# Inputs
#   weight_vector: a 1D array that contains weights for the given topic
#   option: the rescaling function to use
#   scale: the scaling constant that decides the steepness of the function
# Outputs
#   output_vector: a modified weight vector for the training process
def modify_weights(weight_vector, option='tanh', scale=4):
    if option == 'tanh':
        output_vector = inverse_tanh(weight_vector, scale=scale)
    elif option == 'sigmoid':
        output_vector = inverse_sigmoid(weight_vector, scale=scale)
    else:
        output_vector = weight_vector
    return output_vector


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


# Modifies word embeddings using BCD (Block Coordinate Descent)
# Inputs
#   ppmi_mat: a batch of PPMI matrix of size B x V where B is the batch size and V is
#             the size of the vocabulary
#   w_mat: co-optimizing array of size T x V x D
#   weights: the topic adjacency matrix
#   start_idx: beginning index of the batch
#   finish_idx: ending index of the batch
#   topic_idx: the index at which the batch embedding is trained
# Outputs
#   u_mat: optimizing matrix batch of size B x D
def train(ppmi_mat, u_input, w_input, weights, start_idx, finish_idx, topic_idx):
    w_mat = w_input[topic_idx, :, :]
    w_square = np.dot(w_mat.T, w_mat)
    a_mat = w_square + (gam + lmb + tau) * np.eye(dim)
    weights_slice = copy.deepcopy(weights[topic_idx, :])
    weights_slice = modify_weights(weights_slice, option='tanh', scale=4)
    weights_slice[topic_idx] = 0
    d_mat = np.tensordot(weights_slice, u_input[:, start_idx:finish_idx, :], axes=([0], [0]))
    b_mat = np.dot(ppmi_mat, w_mat) + gam * w_mat[start_idx:finish_idx, :] + tau * d_mat
    u_mat = np.linalg.lstsq(a_mat.T, b_mat.T, rcond=None)[0]
    return u_mat.T


# Main method for the file
if __name__ == '__main__':

    # Initialize arrays
    output_file = h5py.File(file_path + '.h5', 'w')
    u_array_slice = np.random.randn(vocab_size, dim) / np.sqrt(dim)
    u_array = np.array([copy.deepcopy(u_array_slice) for _ in indices])
    u_array = output_file.create_dataset('U', data=u_array, chunks=True)
    w_array_slice = np.random.randn(vocab_size, dim) / np.sqrt(dim)
    w_array = np.array([copy.deepcopy(w_array_slice) for _ in indices])
    w_array = output_file.create_dataset('W', data=w_array, chunks=True)

    # Add labels to the output dataset
    output_file.create_dataset('labels', data=labels)

    # Obtain batch indices
    batches = return_batch(vocab_size, batch_size)

    # Iterate and train word embeddings
    for iter_idx in range(num_iter):
        print('==========ITERATION %d==========' % (iter_idx + 1))
        indices_train = np.random.permutation(indices)
        # Iterate over shuffled indices
        for enum, idx in enumerate(indices_train):
            print('Group: %d' % (enum + 1))
            # Iterate over separate batches
            for batch in batches:
                start, end = batch[0], batch[1]
                # Extract the current batch of PPMI matrix
                ppmi_batch = None
                try:
                    ppmi_batch = ppmi[start:end, :, idx]
                except MemoryError:
                    print('Please use a smaller batch size than %d' % batch_size)
                # Train the U and W matrices
                w_array[idx, start:end, :] = train(ppmi_batch, w_array, u_array, topic_matrix, start, end, idx)
                u_array[idx, start:end, :] = train(ppmi_batch, u_array, w_array, topic_matrix, start, end, idx)

    # Indicate that the training step has completed
    output_file.close()
    print('Training completed')
