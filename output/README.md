# doc-word2vec evaluation
## How To Run Trajectory Visualization
Trajectory visualization for temporal embeddings per section 5.1 in [Yao et al., 2018](https://arxiv.org/pdf/1703.00607.pdf)

The 2D t-SNE projection of a word and its most similar words over various time slices are plotted with a user determined perplexity.

After downloading the repo, download the baseline_embeddings.tar.gz we generated from https://drive.google.com/drive/folders/1UM289agQDAwjwO30izEh7dTI428suyww?usp=sharing.
Unzip and place it in baseline/baseline_embeddings.h5

To execute the baseline evaluation script, run the following in terminal:
```python3 nyt_trajectory.py```

To execute the extension evaluation script, run the following in terminal:
```python3 blog_trajectory.py```

You will be prompted with
```Which word would you like to visualize the trajectory for?:```
Type the word you would like to create a trajectory for

Then, after it finds the most similar words at each time slice, it will prompt you to choose perplexity
```Enter the perplexity value (5-50) for the tSNE projection or 'n' to quit:```
This perplexity is used to generate a plot of a 2D tSNE projection.
After you close the plot, it will prompt you with this again, so you can generate projections with varying perplexities.
Play with it and have fun! : )

## How To Run Quantitative Analysis
We implement the semantic similarity baselines outlined in section 6.1 of [Yao et al., 2018](https://arxiv.org/pdf/1703.00607.pdf). 
In particular, we are clustering words according to their embeddings via spherical k-means. (The evaluation script currently uses k=10). Then, we use Normalized Mutual Information (NML) and F-Score to determine semantic similarity. 

To execute the extension evaluation script, run the following in terminal:
```python3 score.py```

You will be prompted with
```To select an embedding, enter a number 0 through 4:```
Enter the number corresponding with the embedding you want to evaluate.  
0: Static  
*1*: Time  
2: Age  
3: Topic (Computer Similarity)  
4: Topic (Human Similarity)

The script will then output the NMF and F_beta scores, as well as a numpy array assigning a cluster to each word in the embedding.

##Normalized Mutual Information (NML)
NML(L,C) = I(L;C)/(H(L)+H(C))/2
Where L represents the set of labels, C represents the set of clusters, I represents the summed mutual information, and H represents the entropy. This is a statistical measure of how pure the clusters are. A higher score is better.
##F-Score
F_beta = ((beta^2 + 1) precision * recall) / (beta^2 * precision + recall)
Here, the clustering is used as a classifier. If two words have the same label and same cluster, they are classified correctly. Otherwise, they are classified incorrectly. The paper above used beta=5, so we do as well. A higher score is better.
