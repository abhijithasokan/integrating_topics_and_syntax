import os
from preprocess_data import preprocess_data

class DataLoader:
    def __init__(self, size):
        self.size = size
        self.dataset = f"data{size}"

    def load_data(self):
        try:
            return self.load_data_from_local()
        except FileNotFoundError:
            preprocess_data(self.size)
            return self.load_data_from_local()



    def load_data_from_local(self):
        dir_name = os.path.join(".", self.dataset)
        with open(os.path.join(dir_name, "documents.txt"), "r") as documents_file:
            data = documents_file.read().splitlines()
            documents = [[int(w) for w in d.split(' ')] for d in data if d != '']

        with open(os.path.join(dir_name, "vocab.txt"), "r") as vocab_file:
            vocab = vocab_file.read().split(' ')

        return documents, vocab
