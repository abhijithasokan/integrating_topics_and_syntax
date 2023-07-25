from sklearn.datasets import fetch_20newsgroups
import spacy
import tqdm
from collections import Counter
import pandas as pd
import re


class NewsgroupDataset:
    def __init__(self, num_docs: int = None, rare_words_threshold: int = 1):
        data = fetch_20newsgroups(subset="train", remove=('headers', 'footers', 'quotes'))
        if num_docs is not None:
            self.unprocessed_docs = data['data'][:num_docs]
        else:
            self.unprocessed_docs = data['data']

        self.nlp = spacy.load("en_core_web_sm")

        docs = self.pre_process_docs_before_vocab(self.unprocessed_docs)
        vocab = self.build_vocab(docs, rare_words_threshold)
        self.docs, self.vocab = self.remove_out_of_vocab_tokens(docs, vocab)


    def build_vocab_map(self):
        words = list(self.vocab.keys())
        words.sort()
        vocab_map = {token: ind for ind, token in enumerate(words)}
        return vocab_map
    

    def get_tokenized_docs_and_vocab_map(self):
        tokenised_docs = []
        vocab_map = self.build_vocab_map()
        for doc in self.docs:
            tokenised_docs.append([vocab_map[token] for token in doc])
        return tokenised_docs, vocab_map


    def pre_process_docs_before_vocab(self, unprocessed_docs):
        docs = []
        patterns_and_replacements = {
            '<EMAIL>' : re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        }
        
        for udoc in tqdm.tqdm_notebook(self.nlp.pipe(unprocessed_docs, batch_size=64), total=len(unprocessed_docs)):
            doc = []
            for token in udoc:
                if token.is_alpha:
                    doc.append(token.text.lower())
                elif token.is_punct:
                    # since punctuation would be one of the syntactic classes
                    doc.append(token.text[0]) # why just text[0]? to handle cases like '!!!' or '...'
                elif token.is_space:
                    # all space char including '\n' provides no meaning 
                    continue
                elif token.is_digit:
                    doc.append('<NUM>') 
                elif token.is_currency:
                    doc.append('<CUR>')
                else:
                    for replacement, pattern in patterns_and_replacements.items():
                        if pattern.match(token.text):
                            doc.append(replacement)
                            break
                    else:
                        doc.append('<UNK>')
            if doc:
                docs.append(doc)
        return docs
    

    def build_vocab(self, docs, rare_words_threshold): 
        vocab = Counter()
        for doc in docs:
            vocab.update(doc)

        # ignore words that are rare
        vocab = Counter({key: count for key, count in vocab.items() if count > rare_words_threshold})
        return vocab

        
    def remove_out_of_vocab_tokens(self, docs, vocab):
        oov_count = 0
        for doc in docs:
            for ind, token in enumerate(doc):
                if token not in vocab:
                    doc[ind] = '<OOV>'
                    oov_count += 1
        vocab['<OOV>'] = oov_count
        return docs, vocab
                



