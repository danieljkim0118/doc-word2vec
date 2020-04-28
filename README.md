# doc-word2vec

## Introduction
This project implements a method to train dynamic word embeddings based on document attributes such as publication date, topic and writer's age, thereby training three-dimensional word embeddings that capture specific features of each word in the context of the input dataset. Previous work can be found [here](https://arxiv.org/abs/1703.00607), which trains year-specific word embeddings in the New York Times dataset crawled between 1990 to 2016. The *baseline* of this project refers to word embeddings generated by the authors of this study. We extend this analysis by training word embeddings based on novel features, including age and document topic, from an independent dataset.

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

Next, download preprocessed data from the [link](https://drive.google.com/open?id=1UM289agQDAwjwO30izEh7dTI428suyww) above to quickly explore the project. To unzip the tar gzipped file on the drive, run the following on the command line (for linux systems):

`tar -zxvf embeddings.tar.gz`

Rename the extracted files as indicated by the following table:

|Google Drive|Local Folder|
|---|---|
|data1.tar.gz|blog_dataset.h5|
|baseline_embeddings.tar.gz|baseline_embeddings.h5|
|embeddings.tar.gz|embeddings-5iter.h5|

The raw dataset can be downloaded [here](https://www.kaggle.com/rtatman/blog-authorship-corpus), and must be named as **blogtext.csv** for the preprocessing code to function properly.

Ensure that the directory is set up as the following:
```
-project
    |-baseline
        -baseline_embeddings.h5
        -baseline_vocab.pkl
    |-data
        -blogtext.csv
    |-embeddings
        -embeddings-5iter.h5
    -blog_dataset.h5
    -blog_vocab.pkl
    -nyt_trajectory.py
    -preprocess_blogs.py
    -train_embeddings.py
```

Again, check that the project directory structure and file names match the format provided above.

## Workflow

The workflow of this project is mainly divided into three steps:

**Preprocessing** => **Training** => **Evaluation**

Data has been provided as input for all steps in the process. Refer to the sections below for more information on each step.

## Preprocessing
Compared to formalized texts such as news reports or scientific articles, blogs inherently lack consistency and structure, thereby requiring a careful preprocessing step to remove noise from the training dataset. The preprocessing methods taken in this project are as follows, in order:

1. Removal of null values, short posts, foreign languages, ill-formatted data
2. Extraction of date/age information
3. Removal of posts within under-represented categories (date, age, topic)
4. Tokenization and Lemmatization using perceptron-based POS taggers and WordNet from NLTK
5. Construction of vocabulary set, filtering outlier words in terms of frequency
6. Generation of term-context matrix using a sliding window
7. Computation of PPMI matrix over different values of user-defined feature

To preprocess the data, simply run the following in the terminal:

`python preprocess_blogs.py`

Hyperparameters including category of interest (including date, age, and topic), window size, threshold for filtering underrepresented categories, etc. can be tuned within the file.

The PPMI matrices generated over different values of user-selected category (default is date) are stored in **blog_dataset.h5**.

## Training
The three-dimensional PPMI matrices generated serve as inputs to a custom loss function to train word embeddings. As mentioned in the previous paper, Block Coordinate Descent (BCD) is used to effectively train word embeddings through improved asymptotic runtime.

To train the word embeddings, run the following in the command line:

`python train_embeddings.py`

Hyperparameters including number of training iterations, coefficients in the loss function, embedding dimension and batch size can be tuned within the file.

The default word embeddings, trained over five iterations, are stored in the HDF file **embeddings-5iter.h5**.

## Evaluation
The baseline evaluation script computes the 2D t-SNE projection of a word and returns the most similar words, along with a plot over various time slices based on the user-determined perplexity value.

To execute the baseline evaluation script, run the following in the terminal:

`python nyt_trajectory.py`

The script will prompt the user with the following question: 

```Which word would you like to visualize the trajectory for?:```

Enter the word to create a time-based trajectory based on the pre-existing word embeddings.

After it finds the most similar words at each time slice, the code will prompt the user to choose the perplexity:

```Enter the perplexity value (5-50) for the tSNE projection or 'n' to quit:```

The perplexity value is used to generate a plot of a 2D tSNE projection. After closing the plot, it will prompt the user with this option again, so the user can generate projections with varying perplexities.

Visualized results for the baseline embeddings can be found in the file **baseline.md**.

Evaluation methods for feature-based word embeddings generated from the blog dataset will be added in the next iteration.

## References
1. Zijun Yao, Yifan Sun, Weicong Ding, Nikhil Rao, and Hui Xiong. 2018. Dynamic word embeddings for evolving semantic discovery. arXiv:1703.00607 (2018).
2. Nikhil Garg, Londa Schiebinger, Dan Jurafsky, and James Zou. 2017. Word embeddings quantify 100 years of gender and ethnic stereotypes. arXiv:1711.08412 (2017).
3. Bin Wang, Angela Wang, Fenxiao Chen, Yuncheng Wang, C.-C. and Jay Kuo. 2019. Evaluating word embedding models: methods and experimental results. arXiv:1901.09785 (2019).
4. Alessio Ferrari, Beatrice Donati, and Stefania Gnesi. 2017. Detecting domain-specific ambiguities: An NLP approach based on wikipedia crawling and word embeddings. In Proceedings of the 2017 IEEE 25th International Requirements Engineering Conference Workshops, pages 393–399.
