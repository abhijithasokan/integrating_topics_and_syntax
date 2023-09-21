
from collections import Counter

import spacy
from sklearn.datasets import fetch_20newsgroups
import tqdm
import re
from models.evaluator import Evaluator
import click

def pre_process_docs_before_vocab(nlp, unprocessed_docs):
    docs = []
    patterns_and_replacements = {
        '<EMAIL>' : re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    }
    stopwords = nlp.Defaults.stop_words

    for udoc in tqdm.tqdm(nlp.pipe(unprocessed_docs, batch_size=64), total=len(unprocessed_docs)):
        doc = []
        for token in udoc:
            if token.text.lower() in stopwords:
                continue
            elif token.is_alpha:
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

        docs.append(doc)
    return docs

def build_vocab(docs, rare_words_threshold):
    vocab = Counter()
    for doc in tqdm.tqdm(docs):
        vocab.update(doc)

    # ignore words that are rare
    vocab = Counter({key: count for key, count in vocab.items() if count > rare_words_threshold})
    return vocab

def remove_out_of_vocab_tokens(docs, vocab):
    oov_count = 0
    for doc in docs:
        for ind, token in enumerate(doc):
            if token not in vocab:
                del doc[ind]
                oov_count += 1
    return docs, vocab


@click.command()

@click.option("--alpha", type=float, default=0.1)
@click.option("--beta", type=float, default=0.01)
@click.option("--gamma", type=float, default=0.1)
@click.option("--delta", type=float, default=0.1)
@click.option("--num_topics", type=int, default=30)
@click.option("--num_classes", type=int, default=10)
@click.option("--num_iter", type=int, default=6000)
@click.option("--iteration", type=int, default=4000)
@click.option("--dataset", type=str, default="data2000")


def evaluate(alpha: float, beta: float, gamma: float, delta: float, num_iter: int, num_topics: int, num_classes: int,
             iteration: int, dataset: str):
    evaluator = Evaluator(alpha, beta, gamma, delta, dataset, num_topics, num_classes, num_iter, iteration)
    data = fetch_20newsgroups(subset="train", remove=('headers', 'footers', 'quotes'))
    nlp = spacy.load("en_core_web_sm")
    unprocessed_docs = data['data'][:200]
    docs = pre_process_docs_before_vocab(nlp, unprocessed_docs)
    vocab = build_vocab(docs, 3)
    docs, vocab = remove_out_of_vocab_tokens(docs, vocab)
    log_probability, perplexity = evaluator.calculate_corpus_likelihood(docs)
    print(f"Log probability: {log_probability}, perplexity: {perplexity}")





if __name__ == '__main__':
    evaluate()



