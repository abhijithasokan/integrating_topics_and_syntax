import os

from models.hmm_lda_model import HMM_LDA_Model
import click


def get_data():
    with open(os.path.join(".", "data", "vocab.txt"), "r") as vocab_file:
        vocab = vocab_file.read().split(' ')

    return vocab

@click.command()

@click.option("--alpha", type=float, default=0.1)
@click.option("--beta", type=float, default=0.01)
@click.option("--gamma", type=float, default=0.1)
@click.option("--delta", type=float, default=0.1)
@click.option("--num_topics", type=int, default=30)
@click.option("--num_classes", type=int, default=10)
@click.option("--num_iter", type=int, default=100)
@click.option("--n", type=int, default=1000)





def generate(alpha: float, beta: float, gamma: float, delta: float, num_iter: int, num_topics: int, num_classes: int,
             n: int):
    vocab = get_data()
    model = HMM_LDA_Model([], vocab, alpha, beta, gamma, delta, num_topics, num_classes, num_iter)
    file_src, document_str = model.generate_doc(n)
    print(file_src)
    print(document_str)


if __name__ == '__main__':
    generate()



