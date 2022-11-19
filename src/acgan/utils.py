import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid


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

def plot_tensor(image_tensor, num_img=20, dim_img=(1, 32, 32)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_img], nrow=5)
    return image_grid.permute(1, 2, 0).squeeze()

def load_dataset(root):
    transform =  transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(0, 1, inplace=False)
            ]
        )
    return datasets.ImageFolder(root=root, transform=transform)
