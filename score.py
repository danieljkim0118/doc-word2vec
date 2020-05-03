from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Script that computes NMF and F_beta using an embedding, topics, and a number of clusters
def semantic_similarity(word_vecs, topics, k, beta=5):
    # Show PCA plot for debugging purposes
    print("Displaying Data...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(word_vecs)
    t = reduced.transpose()
    plt.scatter(t[0], t[1])
    plt.show()

    # Generate Cluster
    print("Clustering...")
    kmeans = KMeans(n_clusters=k, random_state=0).fit(word_vecs)

    # Compute NMF
    print("Computing NMF...")
    nmf = normalized_mutual_info_score(topics, kmeans.labels_)
    print("NMF: " + str(nmf))

    # Compute F_beta
    print("Computing F_beta...")
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(word_vecs)):
        for j in range(len(word_vecs)):
            if kmeans.labels_[i] == kmeans.labels_[j]:
                if topics[i] == topics[j]:
                    tp += 1
                else:
                    fp += 1
            else:
                if topics[i] == topics[j]:
                    fn += 1
                else:
                    tn += 1
    p = tp/(tp+fp)
    r = (tp/tp+fn)
    fbeta = ((beta**2 + 1)*p*r)/(beta**2 * p + r)
    print("F_beta: " + str(fbeta))
    return nmf, fbeta

# Allow user to choose which embedding to use
print("__BLOG EMBEDDINGS__")
print("0: Static")
print("1: Time")
print("2: Age")
print("3: Topic (Computer Similarity)")
print("4: Topic (Human Similarity)")
choice = input("To select an embedding, enter a number 0 through 4: ")
choice = int(choice)
print()

# Based on which embedding the user chooses, define the necessary settings
if choice == 0:
    file_str = 'embeddings/blog_dataset_sample.h5'
    vocab_file_str = 'embeddings/blog_vocab_age.pkl'
if choice == 1:
    file_str = 'embeddings/embeddings-time-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_age.pkl'
elif choice == 2:
    file_str = 'embeddings/embeddings-age-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_age.pkl'
elif choice == 3:
    file_str = 'embeddings/embeddings-topic-human-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_topic.pkl'
elif choice == 4:
    file_str = 'embeddings/embeddings-topic-ppmi_cosine-5iter.h5'
    vocab_file_str = 'embeddings/blog_vocab_topic.pkl'
else:
    print("The input you gave was invalid")

# Load the dictionaries
f = open('blog_vocab.pkl', 'rb')
label_index_to_word = pickle.load(f)
f = open(vocab_file_str, 'rb')
embedding_index_to_word = pickle.load(f)

embedding_word_to_index = {}
for i in range(len(embedding_index_to_word)):
    embedding_word_to_index[embedding_index_to_word[i]] = i

# Load the dataset
print("Loading dataset...")
if choice == 0:
    file = h5py.File(file_str, 'r')

    for key in file.keys():
        print(key)

    dataset = file['ppmi']
    word_vecs = dataset[:, :, 0]
    topics = file['label'][:, 0]
else:
    file = h5py.File(file_str, 'r')

    for key in file.keys():
        print(key)

    dataset = file['U']
    word_vecs = dataset[0, :, :]

    file = h5py.File('embeddings/blog_dataset_sample.h5', 'r')
    topics = file['label'][:, 0]

    print(dataset.shape)
    print(dataset[:, :, 1])

print("The length of topics is: " + str(len(topics)))
print("The length of the embedding dict is: " + str(len(embedding_word_to_index)))
print("The length of the embedding is: " + str(len(word_vecs)))
input()

# Translate the topic labels to the dictionary of the embedding
if choice == 0:
    new_topics = topics
else:
    new_topics = np.zeros(len(embedding_index_to_word))
    for i in range(len(topics)):
        try:
            new_topics[embedding_word_to_index[label_index_to_word[i]]] = topics[i]
        except Exception as e:
            print(e)

k = 10
semantic_similarity(word_vecs, new_topics, k)