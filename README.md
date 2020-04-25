# doc-word2vec

## Introduction
This project implements a method to train dynamic word embeddings based on document attributes such as publication date, topic and writer's age, thereby training three-dimensional word embeddings that capture specific features of each word in the context of the input dataset. Previous work can be found [here](https://arxiv.org/abs/1703.00607).

## Dataset
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

Next, download preprocessed data from the [link](https://drive.google.com/open?id=1UM289agQDAwjwO30izEh7dTI428suyww) above to quickly explore the project. Ensure that the directory is set up as the following:
```
-project
    |-baseline
        -baseline_embeddings.h5
    |-data
        -blogtext.csv
    |-embeddings
        -embeddings-5iter.h5
    blog_dataset.h5
    preprocess_blogs.py
    train_embeddings.py

```

The raw dataset can be downloaded [here](https://www.kaggle.com/rtatman/blog-authorship-corpus), and must be named as **blogtext.csv** for the preprocessing code to function properly.

## Related Works
