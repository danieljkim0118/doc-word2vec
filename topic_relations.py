import numpy as np
import pandas as pd
import swifter
from preprocess_blogs import extract_tokens, filter_blogs


# Computes similarity between every pair of topics
# Inputs
#   vt_matrix: the vocab-topic matrix of size V x T, where V is the number of terms and
#              T is the number of topics
#   metric: 'cosine' computes cosine similarity, 'jaccard' computes jaccard similarity
#           and 'dice' computes dice similarity
# Outputs
#   sim_matrix: the cosine-similarity matrix of size T x T
def compute_similarity(vt_matrix, metric='cosine'):
    n_topics = np.size(vt_matrix, axis=-1)
    sim_matrix = np.zeros((n_topics, n_topics))
    for ii in range(n_topics):
        u = vt_matrix[:, ii]
        for jj in range(n_topics):
            v = vt_matrix[:, jj]
            if metric == 'cosine':
                sim_matrix[ii, jj] = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            elif metric in ['jaccard', 'dice']:
                sim_matrix[ii, jj] = np.sum(np.minimum(u, v)) / np.sum(np.maximum(u, v))
                if metric == 'dice':
                    sim_matrix[ii, jj] = 2 * sim_matrix[ii, jj] / (1 + sim_matrix[ii, jj])
    return sim_matrix


# Define hyperparameters used in the pre-processing step for topic-based category
age_interval = 5
threshold = 3000
method = 'dice'

# Load the blog dataset
blog_df = pd.read_csv('data/blogtext.csv', encoding='utf8')

# Filter rows with unknown topics and short sentences
keep = blog_df.swifter.apply(filter_blogs, axis=1)
blog_df = blog_df[keep]

# Extract age bracket information
blog_df['age_bracket'] = blog_df.swifter.apply(lambda x: int(x['age'] / age_interval) * age_interval, axis=1)

# Filter underrepresented blogs
blog_df = blog_df.groupby('topic').filter(lambda x: len(x) >= threshold)
blog_df = blog_df.groupby('age_bracket').filter(lambda x: len(x) >= threshold)

# Obtain topic information
topics = blog_df.groupby('topic').groups.keys()
print(topics)
num_topics = len(topics)
topic_to_id = dict(zip(topics, range(num_topics)))

# Obtain the vocabulary of the blog dataset
blog_df['text'] = blog_df.swifter.apply(extract_tokens, axis=1)
vocab = dict()
for idx, text in enumerate(blog_df['text'].values):
    for token in text:
        if token in vocab:
            vocab[token] += 1
        else:
            vocab[token] = 0
print('size of original vocabulary: ', len(vocab))

# Iterate over the vocabulary list and remove words that appear too frequently or infrequently
word_count = sum(vocab.values())
use_vocab = dict.fromkeys(vocab.keys(), True)
vocab_items = [(k, v) for k, v in vocab.items()]
for key, value in vocab_items:
    freq = value / word_count
    if freq > 1e-2 or freq < 1e-6:
        vocab.pop(key, None)
        use_vocab[key] = False
print('size of new vocabulary: ', len(vocab))

# Initialize a vocab-topic matrix
vocab_topic_matrix = np.zeros((len(vocab), num_topics))
vocab_to_id = dict(zip(vocab, range(len(vocab))))

# Iterate over every row of the blog dataset and update the vocab-topic matrix
for _, row in blog_df.iterrows():
    token_list = row['text']
    topic = row['topic']
    for token in token_list:
        if use_vocab[token]:
            vocab_topic_matrix[vocab_to_id[token], topic_to_id[topic]] += 1


# Compute the tf-idf matrix
tf_matrix = np.log10(vocab_topic_matrix + 1) + 1
idf_matrix = np.log10(num_topics / np.sum(np.heaviside(vocab_topic_matrix, 0), axis=1))
tf_idf_matrix = (tf_matrix.T * idf_matrix).T
similarity_1 = compute_similarity(tf_idf_matrix, metric=method)
np.save('topics_tf-idf_%s.npy' % method, similarity_1)
print(similarity_1)

# Compute the PPMI matrix
full_sum = np.sum(vocab_topic_matrix)
row_sum = np.sum(vocab_topic_matrix, axis=1)
col_sum = np.sum(vocab_topic_matrix, axis=0)
outer_sum = np.outer(row_sum, col_sum) / (full_sum ** 2 + 1e-6)
ppmi_matrix = np.maximum(np.log2(np.multiply(vocab_topic_matrix / (full_sum + 1e-6), 1 / (outer_sum + 1e-6)) + 1),
                         np.zeros(np.shape(vocab_topic_matrix)))
similarity_2 = compute_similarity(ppmi_matrix, metric=method)
np.save('topics_ppmi_%s.npy' % method, similarity_2)
print(similarity_2)
