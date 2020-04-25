import h5py
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Open the sample dataset
file = h5py.File('baseline/baseline_embeddings.h5', 'r')

# for key in file.keys():
#     print(key)

# Access the time-based embedding matrix (can also call 'context' and 'label')
# print("____TIME-BASED EMBEDDING____")
embedding = file['U']
# print(embedding.shape)
# print(embedding[:, :, 1])
# print("The number of words in the embedding is: " + str(embedding.shape[1]))


# Utility script to find the <num_neighbors> nearest neighbors to the word <index> at a given <time> slice
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

# Load NYT embedding
file = open("baseline/baseline_vocab.pkl", "rb")
index_to_word = pickle.load(file)
word_to_index = {}
for i in range(len(index_to_word)):
    word_to_index[index_to_word[i]] = i
# print("The number of words in the conversion dictionary is: " + str(len(word_to_index)))


# Take user input for word to plot
valid_word = False
word = ""
word_str = ""
while not valid_word:
    word_str = input("Which word would you like to visualize the trajectory for?: ")
    try:
        word = word_to_index[word_str]
        valid_word = True
    except:
        print("The word you entered is not in the NYT dataset")
        valid_word = False

# print("The index for " + word_str + " in the NYT dataset is: " + str(word))

print("\nCalculating the most similar words to " + word_str + "... (this may take a while)")

# similar_words = [15830, 8763, 9019, 8952, 7202, 17883, 8602, 18692, 15935, 19277, 10544, 17883, 8602, 4632, 8952, 15935, 12913, 973, 18692, 19277, 4632, 7202, 13744, 8602, 12913, 17883, 18692, 973, 15935, 19277, 8503, 10837, 18692, 10074, 8895, 1898, 9019, 11187, 13585, 19277, 15935, 444, 17883, 11179, 8952, 15830, 11613, 6173, 16229, 19277, 8763, 5328, 11179, 8538, 6914, 17883, 15830, 8952, 16229, 19277, 18670, 8763, 17115, 13585, 3773, 15935, 18948, 8952, 17883, 19277, 14386, 13744, 17883, 18948, 8952, 6039, 973, 15935, 16229, 19277, 13744, 8503, 13585, 12832, 15935, 6173, 17883, 18948, 9019, 19277, 8895, 1575, 8763, 12832, 6039, 13585, 15830, 11179, 17883, 19277, 10375, 8895, 13585, 8763, 7817, 15935, 8952, 17883, 15830, 19277, 5818, 8895, 12832, 8952, 18948, 6039, 17883, 15935, 14847, 19277, 18948, 5818, 9019, 15935, 14847, 17883, 13903, 8895, 15830, 19277, 15830, 18948, 18692, 13585, 2746, 17883, 12913, 16229, 15935, 19277, 11432, 19021, 14928, 9907, 6510, 7817, 15913, 10097, 16229, 19277, 9693, 9907, 14928, 13084, 6510, 19147, 7269, 10097, 16229, 19277, 5379, 18692, 10689, 6276, 12913, 9907, 17488, 18948, 16229, 19277]
similar_words = []
similar_words_times = {}
n = 10
for t in range(26):
    similar_words += nearest_neighbors(word, t, n)

# print(similar_words)

similar_words = list(set(similar_words))

perplexity = "" # A perplexity of 9 tends to work well
while perplexity != "n":
    perplexity = input("Enter the perplexity value (5-50) for the tSNE projection or 'n' to quit: ")

    if perplexity == "n":
        break
    try:
        perplexity = int(perplexity)

        initial_length = len(similar_words)

        similar_words_strs = [index_to_word[index] for index in similar_words]
        similar_words_vecs = [embedding[8, index, :] for index in similar_words]

        for t in range(0, 26, 4):
            similar_words_strs += [word_str + "-" + str(1990 + t)]
            similar_words_vecs += [embedding[t, word, :]]

        t = 25
        similar_words_strs += [word_str + "-" + str(1990 + t)]
        similar_words_vecs += [embedding[t, word, :]]

        # print(similar_words_strs)

        tsne = TSNE(perplexity=perplexity)
        Y = tsne.fit_transform(similar_words_vecs)

        plt.scatter(Y[:initial_length, 0], Y[:initial_length, 1], color='lightblue')
        plt.scatter(Y[initial_length:, 0], Y[initial_length:, 1], color='coral')

        for i in range(len(Y[:, 0])):
            plt.annotate(
                similar_words_strs[i],
                (Y[:, 0][i], Y[:, 1][i])
            )

        plt.show()
    except:
        print("Invalid perplexity value")