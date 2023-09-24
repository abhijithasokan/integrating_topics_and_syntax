import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import click
import json
import logging
logging.basicConfig(level=logging.INFO)

def build_classifier_ds_from_dumped_delta_of_newsgroup_trained_lda(theta_file_path: str, skipped_indices_path: str, train_test_split: float = 0.8):
    doc_to_topics_count = np.loadtxt(theta_file_path)

    with open(skipped_indices_path, "r") as skipped_indices_file:
        skipped_indices = json.load(skipped_indices_file)

    assert all(isinstance(ind, int) for ind in skipped_indices)
    num_docs, num_topics = doc_to_topics_count.shape
    original_num_docs = num_docs + len(skipped_indices)
    logging.info(f'Original number of docs: {original_num_docs}')

    # normalize the topic vector
    topic_vector = doc_to_topics_count / np.sum(doc_to_topics_count, axis=1)[:, np.newaxis]
    # set all nan values to 1/num_topics
    topic_vector[np.isnan(topic_vector)] = 1/num_topics


    data = fetch_20newsgroups(
        subset="train", remove=('headers', 'footers', 'quotes'),
        categories=[
            'alt.atheism','sci.med','sci.space','sci.electronics',
            'talk.politics.guns','comp.graphics','rec.motorcycles',
            'comp.graphics','soc.religion.christian','comp.os.ms-windows.misc'
            ]
    )
    
    
    cat_to_indices = defaultdict(lambda : ([], []))   
    topic_vec_offset = 0
    for ind in range(original_num_docs):
        if ind in skipped_indices:
            topic_vec_offset += 1
            continue
        category = data.target_names[data.target[ind]]
        cat_to_indices[category][0].append(ind)
        cat_to_indices[category][1].append(ind - topic_vec_offset) # to maitain correspondence with the theta file

    logging.info(f'Number of categories: {len(cat_to_indices)}')
    # logging.info(f'Docs per category: {[(cat, len(indices)) for cat, indices in cat_to_doc_indices.items()]}') 
    
    # train test split by category
    train_data_ind, test_data_ind = [], []
    train_label_ind, test_label_ind = [], []
    for cat, (doc_indices, topic_vec_indices) in cat_to_indices.items():
        num_train = int(len(doc_indices) * train_test_split)
        train_data_ind.extend(topic_vec_indices[:num_train])
        test_data_ind.extend(topic_vec_indices[num_train:])
        train_label_ind.extend(doc_indices[:num_train])
        test_label_ind.extend(doc_indices[num_train:])

    train_data = topic_vector[np.array(train_data_ind)] # shape: (num_train, num_topics)
    train_labels = data.target[np.array(train_label_ind)] # shape: (num_train,)
    
    test_data = topic_vector[np.array(test_data_ind)] # shape: (num_test, num_topics)
    test_labels = data.target[np.array(test_label_ind)] # shape: (num_test,)

    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels
    }


def train_naive_bayes_classifier(data: dict):
    # train a naive bayes classifier
    from sklearn.naive_bayes import MultinomialNB #, GaussianNB
    clf = MultinomialNB()

    clf.fit(data["train_data"], data["train_labels"])
    
    predicted_labels = clf.predict(data["test_data"])
    accuracy = np.sum(predicted_labels == data["test_labels"]) / len(predicted_labels)
    return accuracy

    

@click.command()
@click.option("--theta_file", type=str)
@click.option("--skip_indices_file", type=str)
@click.option("--train_test_split", type=float, default=0.8)
def run_classifier(theta_file: str, skip_indices_file: str, train_test_split: float):
    data = build_classifier_ds_from_dumped_delta_of_newsgroup_trained_lda(theta_file, skip_indices_file, train_test_split)
    accuracy = train_naive_bayes_classifier(data)
    logging.info(f'Accuracy: {accuracy}')
    

if __name__ == "__main__":
    run_classifier()