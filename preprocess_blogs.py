from datetime import datetime
from nltk.corpus import wordnet
from nltk.tag import PerceptronTagger
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import h5py
import nltk
import numpy as np
import os
import pandas as pd
import re
import swifter

# Download wordnet from NLTK
nltk.download('wordnet')

# Unmodifiable variables
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
          'October', 'November', 'December']
nltk_to_wordnet = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}
count = 0

# Modifiable parameters
use_sample = False
category = 'time'  # the category for representing word embeddings (other options are 'age' and 'topic').
threshold = 30 if use_sample else 3000  # minimum number of documents for each group
age_interval = 5  # size of each age bracket
window_size = 5  # size of the context window (one-side length)


# Checks whether string contains only English-related characters
# Inputs
#   string: input string to be checked
# Outputs
#   a boolean indicating whether the given string is in English
def in_english(string):
    try:
        string.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


# Filters blog data on a first-pass round
# Inputs
#   row: a pandas dataframe object that represents each row of the table
# Outputs
#   a boolean indicating whether the row should be kept (True) or not (False)
def filter_blogs(row):
    if row['topic'] == 'indUnk':
        return False
    elif len(row['text'].split()) <= 10:
        return False
    elif not in_english(row['text']):
        return False
    elif row['date'].split(',')[1] not in months:
        return False
    return True


# Counts number of iterations of a function
# Inputs
#   display_freq: the frequency at which to display execution status
#                 relative to pre-defined threshold
# Outputs
#   print statements indicating number of iterative calls to the function
def count_iterations(display_freq):
    global count
    count += 1
    if count % (threshold / display_freq) == 0:
        print('documents processed: ', count)


# Generates a list of tokens for each blog text
# Inputs
#   row: a pandas dataframe object that represents each row of the table
# Outputs
#   token_list: a list of unique tokens present within the text
def extract_tokens(row, lemmatize=True, use_tag=True):
    tokenizer = WhitespaceTokenizer()
    if lemmatize:  # reduce words to lemmas
        pattern = '[().*+,?!\'\";:]*'
        token_list = list()
        if use_tag:  # use POS tags to obtain more accurate lemmas
            pos_tags = PerceptronTagger().tag(tokenizer.tokenize(row['text']))
            lemmatizer_input = map(lambda x: (x[0], nltk_to_wordnet.get(x[1][0])), pos_tags)
            lemmatizer = WordNetLemmatizer()
            for word, tag in lemmatizer_input:
                if word != 'urlLink' and 'http:' not in word:
                    word = word.lower()
                    if tag is None:
                        tok = lemmatizer.lemmatize(word)
                        tok = re.sub(pattern, '', tok)
                        if not tok.isdigit():
                            token_list.append(tok)
                    else:
                        tok = lemmatizer.lemmatize(word, tag)
                        tok = re.sub(pattern, '', tok)
                        if not tok.isdigit():
                            token_list.append(tok)
        else:  # do not use a tagger if not specified and speed up computation
            lemmatizer_input = tokenizer.tokenize(row['text'])
            lemmatizer = WordNetLemmatizer()
            for word in lemmatizer_input:
                if word != 'urlLink' and 'http:' not in word:
                    tok = lemmatizer.lemmatize(word.lower())
                    tok = re.sub(pattern, '', tok)
                    if not tok.isdigit():
                        token_list.append(tok)
    else:  # simply tokenize based on whitespaces
        token_list = tokenizer.tokenize(row['text'])
    return token_list


# Main method for the file
if __name__ == '__main__':

    # Load the blog dataset
    data_file = 'blogtext_sample' if use_sample else 'blogtext'
    blog_df = pd.read_csv('data/%s.csv' % data_file, encoding='utf8')

    # Initially filter all rows that contain unknown topics and short sentences
    keep = blog_df.swifter.apply(filter_blogs, axis=1)
    blog_df = blog_df[keep]

    # Extract publication date information
    blog_df['date'] = blog_df.swifter.apply(lambda x: datetime.strptime(x['date'], '%d,%B,%Y'), axis=1)
    blog_df['year'] = blog_df.swifter.apply(lambda x: x['date'].year, axis=1)
    blog_df['month'] = blog_df.swifter.apply(lambda x: x['date'].month, axis=1)

    # Extract age bracket information
    blog_df['age_bracket'] = blog_df.swifter.apply(lambda x: int(x['age'] / age_interval) * age_interval, axis=1)

    # Group documents by publication year/month and topics and remove documents within
    # categories that have less than a certain number of documents
    blog_df = blog_df.groupby(['year', 'month']).filter(lambda x: len(x) >= threshold)
    blog_df = blog_df.groupby('topic').filter(lambda x: len(x) >= threshold)

    # Group the preprocessed blog dataset by the specified category
    if category == 'age':
        blog_groups = blog_df.groupby('age_bracket')
        test_category = 'topic'
    elif category == 'topic':
        blog_groups = blog_df.groupby('topic')
        test_category = 'age_bracket'
    else:
        blog_groups = blog_df.groupby(['year', 'month'])
        test_category = 'topic'
    category_list = blog_groups
    num_groups = blog_groups.ngroups

    # Create pointers to test indices
    test_category_list = blog_df[test_category].unique()
    test_pointers = dict(zip(test_category_list, range(len(test_category_list))))

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

    # Pre-allocate a h5py group that contains both term-context and ppmi matrix
    path = 'blog_dataset_sample.h5' if use_sample else 'blog_dataset.h5'
    file = h5py.File(path, 'w')
    dataset_context = file.create_dataset('context', (len(vocab), len(vocab), num_groups), dtype=np.float32, chunks=True)
    test_matrix = np.zeros((len(vocab), num_groups), dtype=np.int)

    # Compute the term-context matrix, grouped by categories
    print('======== computing term-context matrix ========')
    vocab_to_id = dict(zip(vocab, range(len(vocab))))
    for label, (_, group_df) in enumerate(blog_groups):
        group_size = len(group_df['text'].values)
        for cnt, txt, info in zip(range(group_size), group_df['text'].values, group_df[test_category].values):
            if cnt > 0 and cnt % threshold == 0:
                print('completed documents: ', cnt)
            for ii, token in enumerate(txt):
                if use_vocab[token]:
                    word_idx = vocab_to_id[token]
                    test_matrix[word_idx][label] = test_pointers.get(info, -1)
                    for jj in range(max(0, ii - window_size), min(ii + window_size + 1, len(txt))):
                        if use_vocab[txt[jj]]:
                            context_idx = vocab_to_id[txt[jj]]
                            dataset_context[word_idx][context_idx][label] += 1
        print('groups processed: %d (%d documents)' % (label + 1, group_size))

    # Allocate data storage for PPMI matrix and test-label matrix
    dataset_ppmi = file.create_dataset('ppmi', (len(vocab), len(vocab), num_groups), dtype=np.float32, chunks=True)
    dataset_label = file.create_dataset('label', (len(vocab), num_groups), dtype=np.int, data=test_matrix)

    # Compute the PPMI matrix for every group and store results
    print('======== computing PPMI matrix ========')
    for idx in range(num_groups):
        tc_matrix = dataset_context[:, :, idx]
        full_sum = np.sum(tc_matrix)
        row_sum = np.sum(tc_matrix, axis=1)
        col_sum = np.sum(tc_matrix, axis=0)
        outer_sum = np.outer(row_sum, col_sum) / (full_sum**2 + 1e-6)
        ppmi_matrix = np.maximum(np.log2(np.multiply(tc_matrix / (full_sum + 1e-6), 1 / (outer_sum + 1e-6)) + 1),
                                 np.zeros(np.shape(tc_matrix)))
        dataset_ppmi[:, :, idx] = ppmi_matrix
        print('groups processed: ', idx + 1)

    # Indicate that the pre-processing step has completed
    print('dataset processing completed.')
