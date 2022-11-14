import torch
import torch.nn.functional as F


def make_noise(batch_size, z_dim, device='cpu'):
    """
    Making a noise vector.

    Args:
        batch_size (int): batch size
        z_dim (int): dimension of the noise vector
    """
    return torch.randn(batch_size, z_dim, device=device)

def make_label(batch_size, n_classes, device='cpu'):
    """
    Making a random lable vector.

    Args:
        batch_size (int): batch size
        n_classes (int):  number of the classes
    """
    return torch.randint(0, n_classes, (batch_size, ), device=device)

def make_one_hot_labels(labels, n_classes, n_repeats=1):
    """
    Making a one-hot array from a tensor of labels.
    """
    return F.one_hot(labels, n_classes).repeat((1, n_repeats))

def combine_vectors(x, y):
    """
    Concatenate two vectors.
    """
    return torch.cat((x.float(),y.float()), 1)
