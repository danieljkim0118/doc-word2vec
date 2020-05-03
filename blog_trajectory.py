import h5py
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Allow user to choose which embedding to use
print("__BLOG EMBEDDINGS__")
print("1: Time")
print("2: Age")
print("3: Topic (Computer Similarity)")
print("4: Topic (Human Similarity)")
choice = input("To select an embedding, enter a number 0 through 4: ")
choice = int(choice)
print()

# Based on which embedding the user chooses, define the necessary settings
if choice == 1:
    file_str = 'embeddings/embeddings-time-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_age.pkl'
    categories = [
        'APR03', 'MAY03', 'JUN03', 'JUL03', 'AUG03', 'SEP03', 'OCT03', 'NOV03', 'DEC03',
        'JAN04', 'FEB04', 'MAR04', 'APR04', 'MAY04', 'JUN04', 'JUL04', 'AUG04'
    ]
elif choice == 2:
    file_str = 'embeddings/embeddings-age-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_age.pkl'
    categories = ['13to15', '16to18', '19to21', '22to24', '25to27', '28to30',
                  '31to33', '34to36', '37to39', '40to42', '42to47']
elif choice == 3:
    file_str = 'embeddings/embeddings-topic-human-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_topic.pkl'
    categories = ['Accounting', 'Advertising', 'Arts', 'Banking', 'BusinessServices', 'Chemicals',
                  'Communications-Media', 'Consulting', 'Education', 'Engineering', 'Fashion', 'Government', 'Internet',
                  'Law', 'Marketing', 'Non-Profit', 'Publishing', 'Religion', 'Science', 'Student', 'Technology']
elif choice == 4:
    file_str = 'embeddings/embeddings-topic-ppmi_cosine-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_topic.pkl'
    categories = ['Accounting', 'Advertising', 'Arts', 'Banking', 'BusinessServices', 'Chemicals',
                  'Communications-Media', 'Consulting', 'Education', 'Engineering', 'Fashion', 'Government', 'Internet',
                  'Law', 'Marketing', 'Non-Profit', 'Publishing', 'Religion', 'Science', 'Student', 'Technology']
else:
    print("The input you gave was invalid")

# Load the dataset
print("Loading dataset...")
file = h5py.File(file_str, 'r')

for key in file.keys():
    print(key)

embedding = file['U']

file = h5py.File('embeddings/blog_dataset_sample.h5', 'r')
topics = file['label'][:, 0]

print(embedding.shape)
print(embedding[:, :, 1])


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

# Load the dictionaries
file = open(vocab_file_str, 'rb')
index_to_word = pickle.load(file)
word_to_index = {}
for i in range(len(index_to_word)):
    word_to_index[index_to_word[i]] = i
print("The number of words in the conversion dictionary is: " + str(len(word_to_index)))


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
        print("The word you entered is not in this embedding")
        valid_word = False

# print("The index for " + word_str + " in the NYT dataset is: " + str(word))

print("\nCalculating the most similar words to " + word_str + "... (this may take a while)")

similar_words = []
similar_words_times = {}
n = 10
for t in range(embedding.shape[0]):
    similar_words += nearest_neighbors(word, t, n)

print(similar_words)

similar_words = list(set(similar_words))

perplexity = "" # A perplexity of 9 tends to work well
while perplexity != "n":
    perplexity = input("Enter the perplexity value (5-50) for the tSNE projection or 'n' to quit: ")

    if perplexity == "n":
        break
    # try:
    perplexity = int(perplexity)

    initial_length = len(similar_words)

    similar_words_strs = [index_to_word[index] for index in similar_words]
    similar_words_vecs = [embedding[8, index, :] for index in similar_words]

    for t in range(0, embedding.shape[0]):
        similar_words_strs += [word_str + "-" + categories[t]]
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
    # except:
    #     print("Invalid perplexity value")