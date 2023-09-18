import os

from models.hmm_lda_model import HMM_LDA_Model
from models import doc_generator
import click


def get_data(dataset):
    dir_name = os.path.join(".", dataset)

    with open(os.path.join(dir_name, "vocab.txt"), "r") as vocab_file:
        vocab = vocab_file.read().split(' ')

    return vocab

@click.command()

@click.option("--alpha", type=float, default=0.1)
@click.option("--beta", type=float, default=0.01)
@click.option("--gamma", type=float, default=0.1)
@click.option("--delta", type=float, default=0.1)
@click.option("--num_topics", type=int, default=30)
@click.option("--num_classes", type=int, default=10)
@click.option("--num_iter", type=int, default=6000)
@click.option("--iteration", type=int, default=4000)
@click.option("--num_docs", type=int, default=10)
@click.option("--dataset", type=str, default="data2000")





def generate(alpha: float, beta: float, gamma: float, delta: float, num_iter: int, num_topics: int, num_classes: int,
             iteration: int, num_docs: int, dataset: str):
    vocab = get_data(dataset)
    doc_generator.generate_documents(vocab, alpha, beta, gamma, delta, num_classes, num_topics, dataset, num_iter, iteration, num_docs)
    # model = HMM_LDA_Model([], vocab, alpha, beta, gamma, delta, num_topics, num_classes, num_iter)
    # file_src, document_str = model.generate_doc(n)
    # print(file_src)
    # print(document_str)


if __name__ == '__main__':
    generate()



