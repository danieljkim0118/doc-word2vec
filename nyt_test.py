import h5py
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Open the sample dataset
file = h5py.File('embeddings/embeddings-5iter.h5', 'r')

for key in file.keys():
    print(key)

# Access the time-based embedding matrix (can also call 'context' and 'label')
print("____TIME-BASED EMBEDDING____")
embedding = file['U']
print(embedding.shape)
print(embedding[:, :, 1])
print(embedding.shape[1])


def nearest_neighbors(index, time, num_neighbors):
    index_word = embedding[time, index, :]

    top_indices = [0] * num_neighbors
    top_similarities = [0] * num_neighbors

    for i in range(20936): # Size of NYT vocabulary
        # Get vector and calculate similarity
        vec = embedding[time, i, :]
        similarity = cosine_similarity([index_word, vec])[0][1]

        # Compare vector to most similar vectors
        top_found = False
        for j in range(num_neighbors):
            if j != 0 and not top_found:
                if top_similarities[j] <= similarity: # If more similar, shift prior tops to the left
                    top_similarities[j-1] = top_similarities[j]
                    top_indices[j-1] = top_indices[j]
                    if j == num_neighbors - 1: # If the most similar, insert at rightmost position
                        top_similarities[j] = similarity
                        top_indices[j] = i
                else: # If less similar and greater than some, insert at previous position
                    top_similarities[j-1] = similarity
                    top_indices[j-1] = i
                    top_found = True

    return top_indices


file = open("baseline/baseline_vocab.pkl", "rb")
index_to_word = pickle.load(file)
word_to_index = {}
for i in range(len(index_to_word)):
    word_to_index[index_to_word[i]] = i

word = word_to_index["apple"]

print("The index for apple is: " + str(word))

similar_words = []
for t in range(17):
    similar_words += nearest_neighbors(word, t, 10)

print(similar_words)


xs = set(similar_words)
for x in xs:
    try:
        print(index_to_word[x])
    except Exception as e:
        print(e)
