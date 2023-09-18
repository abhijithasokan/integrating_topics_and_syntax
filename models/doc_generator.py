import os
import uuid
from pathlib import Path

from .hmm_lda_model import THETA_FILE, PHI_C_FILE, PHI_Z_FILE, PI_FILE, SEMANTIC_CLASS
import numpy as np
from numpy.random import Generator, PCG64



def get_top_k_words_from_distrbution(class_id: int, k: int, vocab, distribution):
    top_k_words = np.argsort(distribution[class_id])[:-k - 1:-1]
    return [vocab[word_id] for word_id in top_k_words]




def generate_documents(vocab, alpha, beta, gamma, delta, num_classes, num_topics, dataset, num_iterations, iteration,
                       doc_num):
    dir_name = os.path.join("out",
                            f"{alpha}_{beta}_{gamma}_{delta}_{num_topics}_{num_classes}_{num_iterations}_{dataset}")
    class_word_counts = np.loadtxt(os.path.join(dir_name, f"iter_{iteration}", PHI_C_FILE))
    topic_word_counts = np.loadtxt(os.path.join(dir_name, f"iter_{iteration}", PHI_Z_FILE))
    transition_counts = np.loadtxt(os.path.join(dir_name, f"iter_{iteration}", PI_FILE))
    stop_word_idx = vocab.index('.')
    prev_class = np.argmax(class_word_counts[:, stop_word_idx])
    stop_word_cls = prev_class
    rng = Generator(PCG64())
    theta_d = rng.dirichlet(alpha * np.ones(num_topics))
    dir_output = os.path.join("gen_out", f"{alpha}_{beta}_{gamma}_{delta}_{num_topics}_{num_classes}_{dataset}",
                              f"{iteration}/{num_iterations}")
    Path(dir_output).mkdir(exist_ok=True, parents=True)
    for i in range(doc_num):
        filename = os.path.join(dir_output, str(uuid.uuid4()))
        file = open(filename, "x")
        document = []
        for _ in range(50):
            sentence = []
            while True:
                z_i = int(np.random.multinomial(1, theta_d).argmax())
                pi_c = transition_counts[prev_class] / np.sum(transition_counts[prev_class])
                c_i = int(np.random.multinomial(1, pi_c).argmax())

                if c_i == SEMANTIC_CLASS:
                    words_distr = topic_word_counts[z_i]
                else:
                    words_distr = class_word_counts[c_i]

                if np.sum(words_distr) != 0:
                    words_distr = words_distr / np.sum(words_distr)
                word = int(np.random.multinomial(1, words_distr).argmax())
                prev_class = c_i
                sentence.append(vocab[word])
                if prev_class == stop_word_cls:
                    break
            file.write(' '.join(sentence) + "\n")
        file.close()

    topics, classes = get_topics_classes(vocab, class_word_counts, topic_word_counts, num_topics, num_classes)
    topic_filename = os.path.join(dir_output, "topics.txt")
    with open(topic_filename, "x") as topic_file:
        topic_file.writelines('\n'.join(topics))
    with open(os.path.join(dir_output, "classes.txt"), "x") as classes_file:
        classes_file.writelines('\n'.join(classes))




def get_topics_classes(vocab, class_word_counts, topic_word_counts, num_topics, num_classes):
    topics = [f"Topic {t}: {' '.join(get_top_k_words_from_distrbution(t, 20, vocab, topic_word_counts))}" for t in range(num_topics)]
    classes = [f"Class {c}: {' '.join(get_top_k_words_from_distrbution(c, 20, vocab, class_word_counts))}" for c in range(num_classes)]
    return topics, classes




