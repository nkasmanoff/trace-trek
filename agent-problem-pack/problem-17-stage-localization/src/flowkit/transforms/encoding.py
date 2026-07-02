"""Categorical encodings (not registered as pipeline stages)."""


def one_hot(labels):
    vocab = sorted(set(labels))
    index = {label: i for i, label in enumerate(vocab)}
    return [[1 if index[label] == i else 0 for i in range(len(vocab))] for label in labels]
