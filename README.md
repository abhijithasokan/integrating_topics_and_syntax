# Implementation of the paper - "Integrating Topics and Syntax"
Link to the [paper](https://papers.nips.cc/paper_files/paper/2004/hash/ef0917ea498b1665ad6c701057155abe-Abstract.html)

# Installation
Setup an environment in conda or pip and install the below packages
  - python=3.10
  - ipykernel
  - jupyter
  - scikit-learn
  - spacy
  - pandas
  - matplotlib
  - nltk
  - gensim
  - tqdm

# Setting up the dataset
## Preprocessing 
```python
python preprocess_data.py
  --size <num-of-docs>
  --dataset <options - news/nips>
```
The above command would preprocess your datasets, and writes the vocab and the document as a list of token ids, into a folder named `{dataset}_{num-docs}`

# Training the model
```python
python main.py
  --alpha <document specific topic distribution's symmetric Dirichlet parameter> 
  --beta <topic specific word distribution> 
  --delta <document specific topic distribution>
  --gamma <distribution of transition between classes>
  --num_iter <iterations of gibbs sampling>
  --num_topics <T>
  --num_classes <C>
  --dataset <path-to-preprocessed-dataset>
```

The model output files are written to the folder - `out/{alpha}_{beta}_{gamma}_{delta}_{num_topics}_{num_classes}_{num_iterations}_{dataset}`





# Evaluation
## Document classification
To run the document classification on newsgroup dataset.
```python
python doc_classifier.py
  --theta_file <path-to-theta.txt>
  --skip_indices_file <path-to-skipped_indices.txt>
  --train_test_split <split-fraction>
```
Where the `theta.txt` contains the document topic counts. And the `skipped_indices.txt` files contains the list of indices of documents you skipped when training our model. 

## Topic Coherence score
Creates a plot of topic coherence score against iterations calculated using gensim. One must supply correct directory containing phi_z.txt file.
```python
python metrics.py
```
## Pure LDA
Trains a ldamodel in gensim on supplied data. Creates a plot of coherence against number of topics.
```python
python pure_lda.py
```



# References
- [LDA from scratch](https://www.depends-on-the-definition.com/lda-from-scratch/#how-do-we-find-theta-and-varphi-gibbs-sampling)

