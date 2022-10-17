import torch.nn.functional as F


def make_one_hot_labels(labels, n_classes=2):
    """
    Making a one-hot array from a tensor of labels.
    """
    return F.one_hot(labels.long(), num_classes=n_classes)
