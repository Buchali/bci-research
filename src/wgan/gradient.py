from torch.autograd import grad
import torch

def get_gradient(critic, real, fake, epsilon):
    """
    Getting the gradient of the critic's scores with respect to mixes of fake and real images.

    Args:
        critic: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vectore of uniformaly random proportions of real/fake images.
    """

    mixed_images = epsilon * real + (1 - epsilon) * fake
    mixed_scores = critic(mixed_images)

    gradient = grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    """
    Calculating the gradient penalty.

    Args:
        gradient: the gradient of the critic's scores with respect to mixes of fake and real images.
    """
    gradient = gradient.view(len(gradient), -1)
    return ((gradient.norm(2, dim=1) - 1) ** 2).mean()
