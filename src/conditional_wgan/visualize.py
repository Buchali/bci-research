from torchvision.utils import make_grid


def plot_tensor(image_tensor, num_img=20, dim_img=(1, 32, 32)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_img], nrow=5)
    return image_grid.permute(1, 2, 0).squeeze()
