import math
import os
import re

import numpy as np
from numpy.random._generator import Generator
from numpy.random._pcg64 import PCG64

from data_loader import DataLoader

SEMANTIC_CLASS = 0
THETA_FILE = "theta.txt"
PHI_C_FILE = "phi_c.txt"
PHI_Z_FILE = "phi_z.txt"
PI_FILE = "pi.txt"


class Evaluator:
    def __init__(self, alpha, beta, gamma, delta, dataset: str, num_topics: int, num_classes: int, num_iterations: int, iteration: int):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dataset = dataset
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.iteration = iteration
        size = int(re.search(r'\d+', dataset).group())
        data_loader = DataLoader(size)
        self.documents, self.vocab = data_loader.load_data_from_local()
        self.oov_idx = self.vocab.index('<UNK>') if '<UNK>' in self.vocab else None
        self.load()
        self.eps = self.default_zero()

    def load(self):
        dir_name = os.path.join("out",
                                f"{self.alpha}_{self.beta}_{self.gamma}_{self.delta}_{self.num_topics}_{self.num_classes}_{self.num_iterations}_{self.dataset}", str(self.iteration))

        self.class_word_counts = np.loadtxt(os.path.join(dir_name, PHI_C_FILE))
        self.topic_word_counts = np.loadtxt(os.path.join(dir_name, PHI_Z_FILE))
        self.transition_counts = np.loadtxt(os.path.join(dir_name, PI_FILE))

        self.phi_c = self.class_word_counts / np.sum(self.class_word_counts)
        self.phi_z = self.topic_word_counts / np.sum(self.topic_word_counts)
        self.pi = self.transition_counts / np.sum(self.transition_counts, axis=1)[:, None]

    def get_word_idx(self, word: str):
        try:
            word_idx = self.vocab.index(word)
        except ValueError:
            word_idx = self.oov_idx
        return word_idx

    def calculate_word_probability(self, word, prev_class_distribution, topic_distribution):
        class_probabilities = np.dot(prev_class_distribution.T, self.pi)
        word_idx = self.get_word_idx(word)
        if not word_idx:
            return class_probabilities, self.eps
        class_counts = self.class_word_counts[:, word_idx]
        class_word_prob_ties = class_counts / np.sum(class_counts) if np.sum(class_counts) > 0.0 else class_counts
        topic_counts = self.topic_word_counts[:, word_idx]
        topic_probabilities = topic_counts / np.sum(topic_counts) if np.sum(topic_counts) > 0 else topic_counts
        word_probability = np.sum(class_probabilities[1:] * class_word_prob_ties[1:]) + 1.0 \
            * np.sum(topic_probabilities * topic_distribution)
        return class_probabilities, word_probability

    def calculate_document_likelihood(self, document):
        prev_class_counts = self.class_word_counts[:, self.vocab.index('.')]
        prev_class_distribution = prev_class_counts / np.sum(prev_class_counts)
        total_log_likelihood = 0.0
        total_log2_likelihood = 0.0
        rng = Generator(PCG64())
        theta_d = rng.dirichlet(self.alpha * np.ones(self.num_topics))
        for word in document:
            prev_class_distribution, word_probability = self.calculate_word_probability(word, prev_class_distribution,
                                                                                        theta_d)
            log_marginal = math.log(word_probability) if word_probability > 0.0 else math.log(self.eps)
            total_log_likelihood -= log_marginal
            log_2_probability = math.log2(word_probability) if word_probability > 0.0 else math.log2(self.eps)
            total_log2_likelihood -= log_2_probability

        try:
            perplexity = 2 ** (total_log2_likelihood / len(document))
        except Exception as e:
            print(f"Couldn't compute perplexity because: document length: {len(document)}, so an exception: {str(e)} occured")
        return total_log_likelihood, perplexity

    def calculate_corpus_likelihood(self, corpus):
        average_likelihood = 0.0
        average_perplexity = 0.0
        count = 0
        for doc in corpus:
            if len(doc) == 0:
                continue
            likelihood, perplexity = self.calculate_document_likelihood(doc)
            average_likelihood += likelihood
            average_perplexity += perplexity
            count += 1
        average_likelihood /= count
        average_perplexity /= count
        return average_likelihood, average_perplexity

    def default_zero(self):
        amount_of_words = sum(len(d) for d in self.documents)
        eps_1 = (self.gamma * self.gamma) / (amount_of_words + self.num_classes * self.gamma)
        eps = (self.num_classes - 1) * self.delta * eps_1 / (len(self.vocab) * self.delta + amount_of_words) + self.beta * eps_1 / (amount_of_words + self.delta * len(self.vocab))
        return eps


