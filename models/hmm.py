import pickle
import os

import numpy as np
import tqdm


class HMM:
    HMM_CLASS_FOR_START_END_MARKER = 1
    HMM_CLASS_FOR_TOPIC = 0
    '''
    Parameters of symmetric Dirichlet distribution -
        - alpha: for document specific topic distribution
        - beta: for topic specific word distribution
        - delta: for class specific word distribution
        - gamma: for transition probability between classes

    Other args:
        - vocab_maps: dict of word to word_id and word_id to word
    '''
    def __init__(
            self, 
            num_topics: int,
            num_classes: int,
            vocab_map: dict,
            alpha: float = None, 
            beta: float = None, 
            delta: float = None, 
            gamma: float = None
        ):
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.vocab_size = len(vocab_map)

        T, V, C = self.num_topics, self.vocab_size, self.num_classes

        self.alpha = alpha if alpha is not None else 1 / T
        self.beta = beta if beta is not None else 1 / V
        self.delta = delta if delta is not None else 1 / V
        self.gamma = gamma if gamma is not None else 1 / C

        self.vocab_map = vocab_map

        self.wordwise_count_in_topic = np.zeros((T, V)) # n_z_w
        self.wordwise_count_in_class = np.zeros((C, V)) # n_c_w
        
        self.wc_topics = np.zeros((T)) # n_z
        self.wc_classes = np.zeros((C)) # n_c
        self.transision_count = np.zeros((C, C))

        
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        npz_path = os.path.join(path, 'model.npz')
        np.savez(
            npz_path,
            wordwise_count_in_topic=self.wordwise_count_in_topic,
            wordwise_count_in_class=self.wordwise_count_in_class,
            wc_topics=self.wc_topics,
            wc_classes=self.wc_classes,
            transision_count=self.transision_count
        )
        non_np_items = {
            'vocab_map': self.vocab_map,
            'alpha': self.alpha,
            'beta': self.beta,
            'delta': self.delta,
            'gamma': self.gamma,
            'num_topics': self.num_topics,
            'num_classes': self.num_classes,
        }
        with open(os.path.join(path, 'non_np_items.pkl'), 'wb') as fp:
            pickle.dump(non_np_items, fp)


    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, 'non_np_items.pkl'), 'rb') as fp:
            non_np_items = pickle.load(fp)

        hmm = cls(
            num_topics=non_np_items['num_topics'],
            num_classes=non_np_items['num_classes'],
            vocab_map=non_np_items['vocab_map'],
            alpha=non_np_items['alpha'],
            beta=non_np_items['beta'],
            delta=non_np_items['delta'],
            gamma=non_np_items['gamma']
        )

        npz_path = os.path.join(path, 'model.npz')
        npz = np.load(npz_path)
        hmm.wordwise_count_in_topic = npz['wordwise_count_in_topic']
        hmm.wordwise_count_in_class = npz['wordwise_count_in_class']
        hmm.wc_topics = npz['wc_topics']
        hmm.wc_classes = npz['wc_classes']
        hmm.transision_count = npz['transision_count']

        return hmm

        
    def get_probability_dists(self):
        unk_word_id = self.vocab_map['<UNK>']
        oov_word_id = self.vocab_map['<OOV>']
        supressed_word_ids = np.array([unk_word_id, oov_word_id])

        filtered_class_prob = self.wordwise_count_in_class.copy()
        filtered_class_prob[:, supressed_word_ids] = 0
        class_prob = filtered_class_prob / np.sum(filtered_class_prob, axis=1, keepdims=True)                                                   

        filtered_topic_prob = self.wordwise_count_in_topic.copy()
        filtered_topic_prob[:, supressed_word_ids] = 0
        topic_prob = filtered_topic_prob / np.sum(filtered_topic_prob, axis=1, keepdims=True)

        trans_prob = self.transision_count / np.sum(self.transision_count, axis=1, keepdims=True)
        return trans_prob, class_prob, topic_prob


    def get_reverse_vocab_map(self):
        return {v: k for k, v in self.vocab_map.items()}


    def get_document_generator(self):
        trans_prob, class_prob, topic_prob = self.get_probability_dists()
        reverse_vocab_map = self.get_reverse_vocab_map()

        def document_generator(doc_len: int):
            c = HMM.HMM_CLASS_FOR_START_END_MARKER
            theta_d = np.random.dirichlet(self.alpha * np.ones(self.num_topics))
            for _ in range(doc_len):
                if c == 1:
                    topic = int(np.random.multinomial(1, theta_d).argmax())
                    word = int(np.random.multinomial(1, topic_prob[topic]).argmax())
                else:
                    word = int(np.random.multinomial(1, class_prob[c]).argmax())
                
                c = int(np.random.multinomial(1, trans_prob[c]).argmax())
                yield reverse_vocab_map[word]

        return document_generator

    def get_top_k_words_from_topic(self, topic_id: int, k: int):
        reverse_vocab_map = self.get_reverse_vocab_map()
        top_k_words = np.argsort(self.wordwise_count_in_topic[topic_id])[:-k-1:-1]
        return [reverse_vocab_map[word_id] for word_id in top_k_words]


    def get_top_k_words_from_class(self, class_id: int, k: int):
        reverse_vocab_map = self.get_reverse_vocab_map()
        top_k_words = np.argsort(self.wordwise_count_in_class[class_id])[:-k-1:-1]
        return [reverse_vocab_map[word_id] for word_id in top_k_words]

        






class HMMTrainer:
    def __init__(self, hmm: HMM):
        self.hmm = hmm


    def train(self, docs: list, num_iterations: int = 100):
        self._init_params(docs)
        for _ in tqdm.tqdm(range(num_iterations)):
            self._train_loop(docs)

    
    def _init_params(self, docs: list):
        D, T, C = len(docs), self.hmm.num_topics, self.hmm.num_classes

        self.class_assignments = [[0 for _ in range(len(d))] for d in docs] # c_i_j
        self.topic_assignments = [[0 for _ in range(len(d))] for d in docs] # z_i_j
        self.topicwise_word_count_for_doc = np.zeros((D, T)) # n_d_z
        self._impluse_fn_vector = np.zeros((C))

        for doc_id, doc in enumerate(docs):
            last_class = HMM.HMM_CLASS_FOR_START_END_MARKER
            self.hmm.wc_classes[HMM.HMM_CLASS_FOR_START_END_MARKER] += 1
            for w_ind, word_id in enumerate(doc):
                # assign a topic randomly to words
                topic_id = self.topic_assignments[doc_id][w_ind] = int(np.random.randint(T))
                # assign a class randomly to words
                cur_class = self.class_assignments[doc_id][w_ind] = int(np.random.randint(C))
                # keep track of our counts
                self._update_counts(doc_id, topic_id, word_id, last_class, cur_class, count=1)
                last_class = cur_class

            self.hmm.transision_count[last_class, HMM.HMM_CLASS_FOR_START_END_MARKER] += 1
            self.hmm.wc_classes[HMM.HMM_CLASS_FOR_START_END_MARKER] += 1
            del last_class


    def _update_counts(self, doc_id, topic_id, word_id, last_class, cur_class, count):
        self.topicwise_word_count_for_doc[doc_id][topic_id] += count
        self.hmm.transision_count[last_class, cur_class] += count
        self.hmm.wordwise_count_in_topic[topic_id, word_id] += count
        self.hmm.wordwise_count_in_class[cur_class, word_id] += count
        self.hmm.wc_topics[topic_id] += count
        self.hmm.wc_classes[cur_class] += count


    def get_cond_topic_dist(self, topicwise_word_count: np.ndarray, cur_class: int, word_id: int):
        V = self.hmm.vocab_size
        p_z = (topicwise_word_count + self.hmm.alpha)
        p_z_t2 = None 
        if cur_class == HMM.HMM_CLASS_FOR_TOPIC:
            # p_z_t2 is the second term of topic dist eqn, that also appears in class dist eqn
            p_z_t2 = (self.hmm.wordwise_count_in_topic[:, word_id] + self.hmm.beta) / (self.hmm.wc_topics + V * self.hmm.beta)
            p_z *= p_z_t2
        
        p_z /= np.sum(p_z)
        return p_z, p_z_t2


    def get_cond_class_dist(self, last_class: int, cur_class: int, next_class: int, topic_id: int, word_id: int, p_z_t2: np.ndarray = None):
        V, C = self.hmm.vocab_size, self.hmm.num_classes
        p_c = (self.hmm.transision_count[last_class] + self.hmm.gamma) # second term in class dist eqn
        if cur_class == HMM.HMM_CLASS_FOR_TOPIC:
            p_c *= p_z_t2[topic_id] # first term in eqn
        else:
            p_c *= (self.hmm.wordwise_count_in_class[:, word_id] + self.hmm.delta) / (self.hmm.wc_classes + V * self.hmm.delta)
        
        if last_class == next_class:
            self._impluse_fn_vector[next_class] = 1
            p_c *= (self.hmm.transision_count[:, next_class] + self._impluse_fn_vector + self.hmm.gamma)
            self._impluse_fn_vector[next_class] = 0
        else:
            p_c *= (self.hmm.transision_count[:, next_class] + self.hmm.gamma)

        self._impluse_fn_vector[last_class] = 1
        p_c /=  (self.hmm.wc_classes + self._impluse_fn_vector + C * self.hmm.gamma)
        self._impluse_fn_vector[last_class] = 0

        p_c /= np.sum(p_c)
        return p_c


    def get_cond_topic_and_class_dist(self, topicwise_word_count: np.ndarray, last_class: int, cur_class: int, next_class: int, topic_id: int, word_id: int):
        p_z, p_z_t2 = self.get_cond_topic_dist(topicwise_word_count, cur_class, word_id)
        p_c = self.get_cond_class_dist(last_class, cur_class, next_class, topic_id, word_id, p_z_t2)
        return p_z, p_c


    def _train_loop(self, docs: list):
        for doc_id, doc in enumerate(docs):
            # last_class - the class of the previous word in the previous training iteration
            # last_class_new - the class of the previous word in the current training iteration
            last_class, last_class_new = HMM.HMM_CLASS_FOR_START_END_MARKER, HMM.HMM_CLASS_FOR_START_END_MARKER
            for w_ind, word_id in enumerate(doc):
                # get current topic and class assignments
                topic_id = self.topic_assignments[doc_id][w_ind]
                cur_class = self.class_assignments[doc_id][w_ind]

                # decrement counts
                self._update_counts(doc_id, topic_id, word_id, last_class, cur_class, count=-1)

                # New topic and class assignments  
                next_class = self.class_assignments[doc_id][w_ind + 1] if w_ind + 1 < len(doc) else HMM.HMM_CLASS_FOR_START_END_MARKER
                p_z, p_c = self.get_cond_topic_and_class_dist(
                    self.topicwise_word_count_for_doc[doc_id],
                    last_class, cur_class, next_class, 
                    topic_id, word_id
                )
                new_topic_id = int(np.random.multinomial(1, p_z).argmax())
                new_class = int(np.random.multinomial(1, p_c).argmax())
    
                # New assignments
                old_class = int(cur_class)
                cur_class = self.class_assignments[doc_id][w_ind] = new_class
                topic_id = self.topic_assignments[doc_id][w_ind] = new_topic_id

                # increment counts
                self._update_counts(doc_id, topic_id, word_id, last_class_new, cur_class, count=1)
                last_class = old_class
                last_class_new = cur_class
                
            self.hmm.transision_count[old_class, HMM.HMM_CLASS_FOR_START_END_MARKER] -= 1
            self.hmm.transision_count[new_class, HMM.HMM_CLASS_FOR_START_END_MARKER] += 1

            del old_class, new_class # just for safeguarding