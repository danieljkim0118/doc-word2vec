# Data

*Please refer to the main README in the code folder for the full description of running the project.*

The dataset for this project is taken from the Blog Authorship Corpus, which contains over 680,000 posts from 19,320 different bloggers spanning ages between 13 to 47, topics ranging from banking to technology. The wide variety of features present within the dataset allows for diverse approaches to build customized word embeddings. A sample of the raw dataset is shown below.

|ID|gender|age|topic|sign|date|text|
|---|---|---|---|---|---|---|
|0001|male|24|Food|Aries|14,May,2004|I ate pizza today...|

The full dataset can be downloaded [here](https://www.kaggle.com/rtatman/blog-authorship-corpus).

All pre-processed data (more description below) for the project can be found [here](https://drive.google.com/open?id=1UM289agQDAwjwO30izEh7dTI428suyww).

## Setup
First, clone the following repository and configure a Python interpreter.

On the command line, install the relevant libraries on the Python.

`pip install h5py nltk numpy pandas swifter`

If the single-line command does not work, install the libraries separately.

Next, download preprocessed data from the [link](https://drive.google.com/open?id=1UM289agQDAwjwO30izEh7dTI428suyww) above to quickly explore the project. To unzip the tar gzipped file on the drive, run the following on the command line (for linux systems):

`tar -zxvf embeddings.tar.gz`

Rename the extracted files as indicated by the following table:

|Google Drive|Local Folder|
|---|---|
|data1.tar.gz|blog_dataset_time.h5|
|data2.tar.gz|blog_dataset_age.h5|
|data3.tar.gz|blog_dataset_topic.h5|
|baseline_embeddings.tar.gz|baseline_embeddings.h5|
|embeddings.tar.gz|embeddings-time-5iter.h5, embeddings-age-5iter.h5|
|embeddings.topic.gz|embeddings-topic-human-5iter.h5, embeddings-topic-ppmi_cosine-5iter.h5|
|blog_vocab_date.pkl|blog_vocab_date.pkl|
|blog_vocab_age.pkl|blog_vocab_age.pkl|
|blog_vocab_topic.pkl|blog_vocab_topic.pkl|
|topics_human.npy|topics_human.npy|
|topics_ppmi_cosine.npy|topics_ppmi_cosine.npy|

The raw dataset can be downloaded [here](https://www.kaggle.com/rtatman/blog-authorship-corpus), and must be named as **blogtext.csv** for the preprocessing code to function properly.

Ensure that the directory is set up as the following:
```
-project
    |-baseline
        -baseline_embeddings.h5
        -baseline_vocab.pkl
    |-data
        -blogtext.csv
        -topics_human.npy
        -topics_ppmi_cosine.npy
    |-embeddings
        -embeddings-time-5iter.h5
        -embeddings-age-5iter.h5
        -embeddings-topic-human-5iter.h5
        -embeddings-topic-ppmi_cosine-5iter.h5
    -blog_dataset_age.h5
    -blog_dataset_time.h5
    -blog_dataset_topic.h5
    -blog_vocab_age.pkl
    -blog_vocab_date.pkl
    -blog_vocab_topic.pkl
    -blog_trajectory.py
    -generate_expert_adj.py
    -nyt_trajectory.py
    -preprocess_blogs.py
    -score.py
    -topic_relations.py
    -train_embeddings.py
    -train_embeddings_graph.py
```

Again, check that the project directory structure and file names match the format provided above.
