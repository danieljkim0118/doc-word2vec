from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def semantic_similarity(word_vecs, topics, k, beta=5):
    # print("Displaying Data...")
    # pca = PCA(n_components=2)
    # reduced = pca.fit_transform(word_vecs)
    # t = reduced.transpose()
    # plt.scatter(t[0], t[1])
    # plt.show()

    print("Clustering...")
    kmeans = KMeans(n_clusters=k, random_state=0).fit(word_vecs)
    print("Computing NMF...")
    nmf = normalized_mutual_info_score(topics, kmeans.labels_)
    print("NMF: " + str(nmf))

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


print("Loading dataset...")
file = h5py.File('baseline/blog_dataset_sample.h5', 'r')
dataset = file['ppmi']
word_vecs = dataset[:, :, 0]
topics = file['label'][:,0]
k = 10
semantic_similarity(word_vecs, topics, k)
