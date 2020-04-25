import h5py
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Open the sample dataset
file = h5py.File('baseline/embeddings-5iter.h5', 'r')

for key in file.keys():
    print(key)

# Access the time-based embedding matrix (can also call 'context' and 'label')
print("____TIME-BASED EMBEDDING____")
embedding = file['U']
print(embedding.shape)
print(embedding[:, :, 1])
print("The number of words in the embedding is: " + str(embedding.shape[1]))


def nearest_neighbors(index, time, num_neighbors):
    index_word = embedding[time, index, :]

    top_indices = [-2] * num_neighbors
    top_similarities = [-2] * num_neighbors

    for i in range(embedding.shape[1]): # Size of NYT vocabulary dict is 20936, which is smaller than embedding.shape[1] (23086)
    # for i in range(100):
        # Get vector and calculate similarity
        vec = embedding[time, i, :]
        similarity = cosine_similarity([index_word, vec])[0][1]
        # print(index_to_word[i] + ": " + str(similarity))

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
            if j == 0 and top_similarities[j] > similarity:
                top_found = True

        # print(top_similarities)

    return top_indices


file = open("baseline/baseline_vocab.pkl", "rb")
index_to_word = pickle.load(file)
word_to_index = {}
for i in range(len(index_to_word)):
    word_to_index[index_to_word[i]] = i

word = word_to_index["apple"]

print("The number of words in the conversion dictionary is: " + str(len(word_to_index)))

print("The index for apple in the dictionary is: " + str(word))

print("\nMOST SIMILAR WORDS TO APPLE:")

similar_words = []
for t in range(1):
    similar_words += nearest_neighbors(word, t, 10)

print(similar_words)


xs = set(similar_words)
for x in similar_words:
    try:
        print(index_to_word[x])
    except Exception as e:
        print(e)
