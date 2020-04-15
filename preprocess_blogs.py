from datetime import datetime
from nltk.corpus import wordnet
from nltk.tag import PerceptronTagger
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import pandas as pd
import re

# Unmodifiable variables
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
          'October', 'November', 'December']
nltk_to_wordnet = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}

# Modifiable variables
threshold = 30
age_interval = 5
window_size = 5
category = 'time'  # can also be 'age' or 'topic'


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


# Generates a list of tokens for each blog text
# Inputs
#   row: a pandas dataframe object that represents each row of the table
# Outputs
#   token_list: a list of unique tokens present within the text
def extract_tokens(row, lemmatize=False):
    tokenizer = WhitespaceTokenizer()
    if lemmatize:
        pos_tags = PerceptronTagger().tag(tokenizer.tokenize(row['text']))
        lemmatizer_input = map(lambda x: (x[0], nltk_to_wordnet.get(x[1][0])), pos_tags)
        token_list = list()
        lemmatizer = WordNetLemmatizer()
        for word, tag in lemmatizer_input:
            if filter_token(word):
                word = word.lower()
                if tag is None:
                    token = lemmatizer.lemmatize(word)
                    token = re.sub('[().*+]+|[0-9]+', '', token)
                    token_list.append(token)
                else:
                    token = lemmatizer.lemmatize(word, tag)
                    token = re.sub('[().*+]+|[0-9]+', '', token)
                    token_list.append(token)
    else:
        token_list = tokenizer.tokenize(row['text'])
    return token_list


# Determines whether to incorporate the token into the vocabulary
# Inputs
#   tok: the token of interest
# Outputs
# a boolean indicating whether to include the token (True) or not (False)
def filter_token(tok):
    if tok == 'urlLink':
        return False
    elif 'http:' in tok:
        return False


# Load the blog dataset
blog_df = pd.read_csv('data/blogtext_sample.csv', encoding='utf8')

# Initially filter all rows that contain unknown topics and short sentences
keep = blog_df.apply(filter_blogs, axis=1)
blog_df = blog_df[keep]

# Extract publication date information
blog_df['date'] = blog_df.apply(lambda x: datetime.strptime(x['date'], '%d,%B,%Y'), axis=1)
blog_df['year'] = blog_df.apply(lambda x: x['date'].year, axis=1)
blog_df['month'] = blog_df.apply(lambda x: x['date'].month, axis=1)

# Extract age bracket information
blog_df['age_bracket'] = blog_df.apply(lambda x: int(x['age'] / age_interval) * age_interval, axis=1)

# Group documents by publication year/month and topics and remove documents within
# categories that have less than a certain number of documents
blog_df = blog_df.groupby(['year', 'month']).filter(lambda x: len(x) >= threshold)
blog_df = blog_df.groupby('topic').filter(lambda x: len(x) >= threshold)

# Obtain the vocabulary of the entire blog dataset
blog_df['tokens'] = blog_df.apply(extract_tokens, axis=1)
vocab = {}
for text in blog_df['tokens'].values:
    for token in text:
        if token in vocab:
            vocab[token] += 1
        else:
            vocab[token] = 0
print(len(vocab))

# Compute the term-context matrix
tc_matrix = np.zeros((len(vocab), len(vocab)))
vocab_to_id = dict(zip(vocab, range(len(vocab))))
for text in blog_df['tokens'].values:
    for ii, token in enumerate(text):
        word_idx = vocab_to_id[token]
        for jj in range(max(0, ii - window_size), min(ii + window_size + 1, len(text))):
            context_idx = vocab_to_id[text[jj]]
            tc_matrix[word_idx][context_idx] += 1

# Compute the PPMI matrix
full_sum = np.sum(tc_matrix)
row_sum = np.sum(tc_matrix, axis=1)
col_sum = np.sum(tc_matrix, axis=0)
outer_sum = np.outer(row_sum, col_sum) / full_sum**2
ppmi_matrix = np.maximum(np.log2(np.multiply(tc_matrix / (full_sum + 1e-6), 1 / (outer_sum + 1e-6)) + 1),
                         np.zeros(np.shape(tc_matrix)))
print(np.shape(ppmi_matrix))
