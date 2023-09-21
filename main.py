import os

from models.hmm_lda_model import HMM_LDA_Model
import click


def get_data(dataset):
    dir_name = os.path.join(".", dataset)
    with open(os.path.join(dir_name, "documents.txt"), "r") as documents_file:
        data = documents_file.read().splitlines()
        documents = [[int(w) for w in d.split(' ')] for d in data if d != '']

    with open(os.path.join(dir_name, "vocab.txt"), "r") as vocab_file:
        vocab = vocab_file.read().split(' ')

    return documents, vocab


@click.command()

@click.option("--alpha", type=float, default=0.02)
@click.option("--beta", type=float, default=0.02)
@click.option("--gamma", type=float, default=0.02)
@click.option("--delta", type=float, default=0.02)
@click.option("--num_iter", type=int, default=10)
@click.option("--num_topics", type=int, default=5)
@click.option("--num_classes", type=int, default=5)
@click.option("--dataset", type=str, default="nips_100")
def main(alpha: float, beta: float, gamma: float, delta: float, num_iter: int, num_topics: int, num_classes: int, dataset: str):
    docs, vocab = get_data(dataset)
    model = HMM_LDA_Model(docs, vocab, alpha, beta, gamma, delta, num_topics, num_classes, dataset, num_iter)
    model.train()


if __name__ == '__main__':
    main()



