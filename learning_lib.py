import numpy as np

def train_test_split(data, test_size=0.25):
    if test_size <= 0 or test_size >= 1:
        raise Exception(
            f"Test split size must be between 0 and 1 but got {test_size}")
    train_indexes, test_indexes = [], []
    for i in range(data.shape[0]):
        if np.random.uniform(0, 1) > test_size:
            train_indexes.append(i)
        else:
            test_indexes.append(i)
    train_data = data.reindex(train_indexes).reset_index()
    test_data = data.reindex(test_indexes).reset_index()
    return train_data, test_data

def accuracy_score(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(labels.shape[0]):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    accuracy = (true_pos + true_neg) / \
        (true_pos + true_neg + false_pos + false_neg)
    return accuracy