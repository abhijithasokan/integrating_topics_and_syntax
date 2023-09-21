import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from numpy.random import Generator, PCG64
import uuid

SEMANTIC_CLASS = 0
THETA_FILE = "theta.txt"
PHI_C_FILE = "phi_c.txt"
PHI_Z_FILE = "phi_z.txt"
PI_FILE = "pi.txt"


def draw_from_empirical(self, counts):
    if np.sum(counts) > 0:
        probabilities = counts / np.sum(counts)
    else:
        probabilities = counts
    draw = int(np.random.multinomial(1, probabilities).argmax())
    return draw


class HMM_LDA_Model:
    def __init__(self, documents, vocabulary: dict, 
                        alpha: float, beta: float, gamma: float, delta: float, 
                        num_topics: int, num_classes: int,
                        dataset: str,
                        num_iterations: int = 10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.num_iterations = num_iterations
        self.docs = documents
        self.num_docs = len(self.docs)
        self.vocab_size = len(vocabulary)
        self.vocab = vocabulary
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.dataset = dataset
        self.init_counts()
        self.init_dir()

    def init_dir(self):
        training_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        dir_name = f"{self.alpha}_{self.beta}_{self.gamma}_{self.delta}_{self.num_topics}_{self.num_classes}_{self.num_iterations}_{self.dataset}"
        self.__output_dir = os.path.join("out", dir_name)
        Path(self.__output_dir).mkdir(exist_ok=True,parents=True)

    def init_counts(self):
        self.doc_topic_counts = np.zeros((self.num_docs, self.num_topics))
        self.topic_word_counts = np.zeros((self.num_topics, self.vocab_size))
        self.class_word_counts = np.zeros((self.num_classes, self.vocab_size))
        self.transitions_counts = np.zeros((self.num_classes, self.num_classes))

        self.docs_topics = [np.random.randint(0, self.num_topics, size=len(doc)) for doc in self.docs]
        self.docs_classes = [np.random.randint(0, self.num_classes, size=len(doc)) for doc in self.docs]

        self.n_z = np.zeros((self.num_topics))
        self.n_c = np.zeros((self.num_classes))

    def run_counts(self):
        for d, doc in enumerate(self.docs):
            prev_class = None
            for i, word in enumerate(doc):
                c = self.docs_classes[d][i]
                if c == SEMANTIC_CLASS:
                    z = self.docs_topics[d][i]
                    self.doc_topic_counts[d][z] += 1
                    self.topic_word_counts[z, word] += 1
                    self.n_z[z] += 1

                self.class_word_counts[c, word] += 1
                self.n_c[c] += 1

                if prev_class is not None:
                    self.transitions_counts[prev_class, c] += 1
                prev_class = c

    def draw_class(self, doc_idx, word_idx, word, doc_size):
        z = self.docs_topics[doc_idx][word_idx]
        c = self.docs_classes[doc_idx][word_idx]

        self.class_word_counts[c, word] -= 1
        self.n_c[c] -= 1

        if c == SEMANTIC_CLASS:
            self.doc_topic_counts[doc_idx][z] -= 1
            self.topic_word_counts[z, word] -= 1
            self.n_z[z] -= 1

        p_w_c = (self.class_word_counts[:, word] + self.delta) / (self.n_c + self.vocab_size * self.delta)
        p_w_c[0] = (self.topic_word_counts[z, word] + self.beta) / (self.n_z[z] + self.vocab_size * self.beta)

        previous = None if word_idx == 0 else self.docs_classes[doc_idx][word_idx - 1]
        future_class = None if word_idx == (doc_size - 1) else self.docs_classes[doc_idx][word_idx + 1]

        if previous is not None:
            self.transitions_counts[previous][c] -= 1

        if future_class is not None:
            self.transitions_counts[c][future_class] -= 1

        n_c_minus = np.zeros(self.num_classes) if previous is None else self.transitions_counts[previous, :]
        n_c_plus = np.zeros(self.num_classes) if future_class is None else self.transitions_counts[:, future_class]

        i_c_minus_plus = np.zeros(self.num_classes)
        i_c_minus = np.zeros(self.num_classes)

        if previous is not None and future_class is not None and previous == future_class:
            i_c_minus_plus[previous] = 1

        if previous is not None:
            i_c_minus[previous] = 1

        p_c_minus = (n_c_minus + self.gamma) * (n_c_plus + i_c_minus_plus + self.gamma) / (self.n_c + i_c_minus + self.num_classes * self.gamma)

        p_c = p_w_c * p_c_minus

        p_c = p_c / np.sum(p_c)
        try:

            new_c = np.random.choice(np.arange(self.num_classes), p=p_c)
        except Exception as e:
            raise e

        self.docs_classes[doc_idx][word_idx] = new_c

        self.class_word_counts[new_c, word] += 1
        self.n_c[new_c] += 1

        if previous is not None:
            self.transitions_counts[previous][new_c] += 1

        if future_class is not None:
            self.transitions_counts[new_c][future_class] += 1

        if new_c == SEMANTIC_CLASS:
            self.doc_topic_counts[doc_idx][z] += 1
            self.topic_word_counts[z, word] += 1
            self.n_z[z] += 1

    def draw_topic(self, doc_idx: int, word_idx: int, word):
        z = self.docs_topics[doc_idx][word_idx]
        c = self.docs_classes[doc_idx][word_idx]

        if c == SEMANTIC_CLASS: 
            self.doc_topic_counts[doc_idx][z] -= 1
            self.topic_word_counts[z, word] -= 1
            self.n_z[z] -= 1

        p_t_minus = self.doc_topic_counts[doc_idx] + self.alpha
        p_w_z = 1.0

        if c == 0:
            p_w_z = (self.topic_word_counts[:, word] + self.beta) / (self.n_z + self.vocab_size * self.beta)

        p_z = p_t_minus * p_w_z

        p_z = p_z / np.sum(p_z)

        new_z = np.random.choice(np.arange(self.num_topics), p=p_z)
        self.docs_topics[doc_idx][word_idx] = new_z

        if c == SEMANTIC_CLASS:
            self.doc_topic_counts[doc_idx][new_z] += 1
            self.topic_word_counts[new_z, word] += 1
            self.n_z[new_z] += 1

    def train(self):
        self.run_counts()
        for i in tqdm(range(1, self.num_iterations + 1)):
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = executor.map(
                    self.fancy_counting,range(len(self.docs)),self.docs)

            if i > 2000 and not i % 200:
                self.save_iteration_model(i)
        self.save()

    def fancy_counting(self, d, doc):
        for w, word in enumerate(doc):
            doc_size = len(doc)
            self.draw_class(d, w, word, doc_size)
            self.draw_topic(d, w, word)

    def save_iteration_model(self, i):
        dir_src = os.path.join(self.__output_dir, f"{i}")
        Path(dir_src).mkdir(exist_ok=True)
        np.savetxt(os.path.join(dir_src, THETA_FILE), self.doc_topic_counts)
        np.savetxt(os.path.join(dir_src, PHI_C_FILE), self.class_word_counts)
        np.savetxt(os.path.join(dir_src, PHI_Z_FILE), self.topic_word_counts)
        np.savetxt(os.path.join(dir_src, PI_FILE), self.transitions_counts)

    def save_model_generation(self, i):
        filename = os.path.join(self.__output_dir, f"{i}_doc")
        rng = Generator(PCG64())
        theta_d = rng.dirichlet(self.alpha * np.ones(self.num_topics))
        prev_class = np.random.randint(self.num_classes)
        document = []
        for i in range(5000):
            z_i = int(np.random.multinomial(1, theta_d).argmax())
            pi_c = self.transitions_counts[prev_class] / np.sum(self.transitions_counts[prev_class])
            c_i = int(np.random.multinomial(1, pi_c).argmax())

            if c_i == SEMANTIC_CLASS:
                words_distr = self.topic_word_counts[z_i]
            else:
                words_distr = self.class_word_counts[c_i]

            if np.sum(words_distr) != 0:
                words_distr = words_distr / np.sum(words_distr)
            word = int(np.random.multinomial(1, words_distr).argmax())
            document.append(self.vocab[word])
            prev_class = c_i

        document_str = ' '.join(document)

        topics = [f"Topic {t}: {' '.join(self.get_top_k_words_from_topic(t, 20))}" for t in range(self.num_topics)]
        classes = [f"Class {c}: {' '.join(self.get_top_k_words_from_class(c, 20))}" for c in range(self.num_classes)]
        with open(filename, 'w+') as doc_file:
            doc_file.write(document_str)
            doc_file.write("\n")
            doc_file.writelines('\n'.join(topics))
            doc_file.write("\n")
            doc_file.writelines('\n'.join(classes))

    def save(self):
        folder_name = f"{self.alpha}_{self.beta}_{self.gamma}_{self.delta}_{self.num_topics}_{self.num_classes}_{self.num_iterations}_{self.dataset}"
        dir_src = os.path.join("out", folder_name)
        Path(dir_src).mkdir(exist_ok=True, parents=True)
        np.savetxt(os.path.join(dir_src, THETA_FILE), self.doc_topic_counts)
        np.savetxt(os.path.join(dir_src, PHI_C_FILE), self.class_word_counts)
        np.savetxt(os.path.join(dir_src, PHI_Z_FILE), self.topic_word_counts)
        np.savetxt(os.path.join(dir_src, PI_FILE), self.transitions_counts)

    def generate_doc(self, length: int):
        dir_name = os.path.join("out", f"{self.alpha}_{self.beta}_{self.gamma}_{self.delta}_{self.num_topics}_{self.num_classes}_{self.num_iterations}_{self.dataset}")
        theta = np.loadtxt(os.path.join(dir_name, THETA_FILE))
        phi_c = np.loadtxt(os.path.join(dir_name, PHI_C_FILE))
        phi_z = np.loadtxt(os.path.join(dir_name, PHI_Z_FILE))
        pi = np.loadtxt(os.path.join(dir_name, PI_FILE))

        doc_idx = np.random.randint(low=0, high=theta.shape[0])
        theta_d = theta[doc_idx] / np.sum(theta[doc_idx])
        phi_c = phi_c / np.sum(phi_c)
        phi_z = phi_z / np.sum(phi_z)
        num_topics = len(theta_d)
        num_classes = len(phi_c)
        prev_class = None
        document = []

        for i in range(length):
            z_i = np.random.choice(np.arange(num_topics), p=theta_d)
            if prev_class is not None:
                pi_c = pi[prev_class] / np.sum(pi[prev_class])
                c_i = np.random.choice(np.arange(num_classes), p=pi_c)
            else:
                c_i = np.random.randint(num_classes)

            if c_i == SEMANTIC_CLASS:
                words_distr = phi_z[z_i]
            else:
                words_distr = phi_c[c_i]

            words_distr = words_distr / np.sum(words_distr)
            
            word_idx = np.random.choice(np.arange(len(words_distr)), p=words_distr)
            word = self.vocab[word_idx]

            document.append(word)

        document_str = ' '.join(document)

        filename = os.path.join(dir_name, f"{str(uuid.uuid4())}_{doc_idx}")

        with open(filename, 'x') as doc_file:
            doc_file.write(document_str)

        return filename, document_str

    def get_top_k_words_from_topic(self, topic_id: int, k: int):

        top_k_words = np.argsort(self.topic_word_counts[topic_id])[:-k - 1:-1]
        return [self.vocab[word_id] for word_id in top_k_words]

    def get_top_k_words_from_class(self, class_id: int, k: int):
        top_k_words = np.argsort(self.class_word_counts[class_id])[:-k - 1:-1]
        return [self.vocab[word_id] for word_id in top_k_words]



            
                









        







