import os
from pprint import pprint
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import matplotlib.pyplot as plt

PHI_Z_FILE = "phi_z.txt"
dir_name = "output/0.02_0.02_0.02_0.02_10_8_4000_news_2000"

def plot(x,y,type):
    # Show graph
    plt.figure(figsize=(8,6))
    plt.plot(x, y,label=type)
    plt.title(f"Coherence score per iteration ({type})")
    plt.xlabel("n_iter")
    plt.ylabel("Coherence score")
    plt.legend()
    plt.xticks(rotation=45)

    plt.savefig(f"{type}_n_iter_news.png")

def metrics(type,PHI_Z_FILE=PHI_Z_FILE, dir_name=dir_name):
    def get_data():
        dir_name = os.path.join(".","news_3000_test")
        with open(os.path.join(dir_name, "documents.txt"), "r") as documents_file:
            data = documents_file.read().splitlines()
            documents = [[int(w) for w in d.split(' ')] for d in data if d != '']
        with open(os.path.join(dir_name, "vocab.txt"), "r") as vocab_file:
            vocab = vocab_file.read().split(' ')
        return vocab,documents
    
    vocab,docs = get_data()
    list_of_docs = []
    for doc in docs:
        doc_str_list = []
        for word_id in doc:
            word = vocab[word_id]
            doc_str_list.append(word)
        list_of_docs.append(doc_str_list)

    id2word = corpora.Dictionary(list_of_docs)
    corpus = [id2word.doc2bow(text) for text in list_of_docs]
    def get_top_k_words_from_topic(topic_id: int, k: int):
        top_k_words = np.argsort(topic_word_counts[topic_id])[:-k - 1:-1]
        return [vocab[word_id] for word_id in top_k_words]
    
    iter_list =[]
    coh_list = []
    for subdir, _, files in os.walk(dir_name):
        for file in files:
            if 'iter' in subdir and PHI_Z_FILE==file:
                iter_name = subdir.split('\\')[1]
                topic_word_counts = np.loadtxt(os.path.join(subdir, PHI_Z_FILE))
                topics = [get_top_k_words_from_topic(t, 5) for t in range(10)]
                iter_list.append(iter_name)
                cm = CoherenceModel(topics=topics,texts = list_of_docs,corpus=corpus, dictionary=id2word, coherence=type)
                coh_list.append(cm.get_coherence())
    plot(iter_list,coh_list,type)
    


if __name__ == "__main__":
    metrics(type='c_npmi')