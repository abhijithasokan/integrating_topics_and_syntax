import os
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import matplotlib.pyplot as plt

PHI_Z_FILE = "phi_z.txt"
dir_name = "out_slow/0.02_0.02_0.02_0.02_5_5_10_nips_100"

def metrics(PHI_Z_FILE=PHI_Z_FILE, dir_name=dir_name):
    def get_data():
        dir_name = os.path.join(".","nips_100")
        with open(os.path.join(dir_name, "documents.txt"), "r") as documents_file:
            data = documents_file.read().splitlines()
            documents = [[int(w) for w in d.split(' ')] for d in data if d != '']
        with open(os.path.join(dir_name, "vocab.txt"), "r") as vocab_file:
            vocab = vocab_file.read().split(' ')
        return vocab,documents

    topic_word_counts = np.loadtxt(os.path.join(dir_name, PHI_Z_FILE))
    vocab,docs = get_data()
    def get_top_k_words_from_topic(topic_id: int, k: int):
        top_k_words = np.argsort(topic_word_counts[topic_id])[:-k - 1:-1]
        return [vocab[word_id] for word_id in top_k_words]

    
    list_of_docs = []
    for doc in docs:
        doc_str_list = []
        for word_id in doc:
            word = vocab[word_id]
            doc_str_list.append(word)
        list_of_docs.append(doc_str_list)

    id2word = corpora.Dictionary(list_of_docs)
    corpus = [id2word.doc2bow(text) for text in list_of_docs]
    
    k_list = []
    coh_list = []
    for k in range(1,20,5):
        k_list.append(k)
        topics = [get_top_k_words_from_topic(t, k) for t in range(5)]
        cm = CoherenceModel(topics=topics,texts = list_of_docs,corpus=corpus, dictionary=id2word, coherence='c_v')
        coh_list.append(cm.get_coherence())
    return coh_list, k_list    


if __name__ == "__main__":
    y,x = metrics()
    # Show graph

    plt.figure(figsize=(8,6))
    plt.plot(x, y,label="c_v")
    plt.title("Coherence score comparing (c_v)")
    plt.xlabel("n_topics")
    plt.ylabel("Coherence score")
    plt.legend()
    plt.xticks(rotation=45)

    plt.savefig("coherence_score_n_topics_nips_slow.png")