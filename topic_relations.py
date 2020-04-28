import numpy as np
import pandas as pd
import swifter
from preprocess_blogs import extract_tokens, filter_blogs

# Define hyperparameters used in the pre-processing step for topic-based category
age_interval = 5
threshold = 30

# Load the blog dataset
blog_df = pd.read_csv('data/blogtext_sample_large.csv', encoding='utf8')

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
    if freq > 1e-2 or freq < 1e-4:
        vocab.pop(key, None)
        use_vocab[key] = False
print('size of new vocabulary: ', len(vocab))

# Initialize a vocab-topic matrix
vocab_topic_matrix = np.zeros((len(vocab), num_topics))
vocab_to_id = dict(zip(vocab, range(len(vocab))))

# Iterate over every row of the blog dataset and update the vocab-topic matrix
for _, row in blog_df.iterrows():
    token_list = row['text']
    for token in token_list:
        if use_vocab[token]:
            topic = row['topic']
            vocab_topic_matrix[vocab_to_id[token], topic_to_id[topic]] += 1


# Compute the tf-idf matrix
tf_matrix = np.log10(vocab_topic_matrix + 1) + 1
idf_matrix = np.log10(num_topics / np.sum(np.heaviside(vocab_topic_matrix, 0), axis=1))
tf_idf_matrix = (tf_matrix.T * idf_matrix).T
similarity_1 = np.dot(tf_idf_matrix.T, tf_idf_matrix)
similarity_1 = np.multiply(similarity_1.T, 1 / np.amax(similarity_1, axis=1)).T
print(similarity_1)

# Compute the PPMI matrix
full_sum = np.sum(vocab_topic_matrix)
row_sum = np.sum(vocab_topic_matrix, axis=1)
col_sum = np.sum(vocab_topic_matrix, axis=0)
outer_sum = np.outer(row_sum, col_sum) / (full_sum ** 2 + 1e-6)
ppmi_matrix = np.maximum(np.log2(np.multiply(vocab_topic_matrix / (full_sum + 1e-6), 1 / (outer_sum + 1e-6)) + 1),
                         np.zeros(np.shape(vocab_topic_matrix)))
similarity_2 = np.dot(ppmi_matrix.T, ppmi_matrix)
similarity_2 = np.multiply(similarity_2.T, 1 / np.amax(similarity_2, axis=1)).T
print(similarity_2)


print(vocab_topic_matrix[:10, :10])
print(tf_idf_matrix[:10, :10])
print(ppmi_matrix[:10, :10])
