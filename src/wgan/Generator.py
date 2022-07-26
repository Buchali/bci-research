import torch
from torch import nn


class Generator(nn.Module):
    """
    Generator Class

    Args:
        z_dim (int): dimension of the noise vector
        img_chan (int): the number of the output image channels
        hidden_dim (int): the number of the hidden units
    """
    def __init__(self, z_dim=10, img_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        # The Network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, 4*hidden_dim),
            self.make_gen_block(4*hidden_dim, 2*hidden_dim, padding=1),
            self.make_gen_block(2*hidden_dim, hidden_dim, padding=1),
            self.make_gen_block(hidden_dim, img_chan, padding=1, final_layer=True)
        )

    def make_gen_block(self, input_dim, output_dim, kernel_size=4, stride=2, padding=0, final_layer=False):
        """
        Making a generator block

        Args:
            input_dim (int): the number of the input channels
            output_dim (int): the number of the output channels
            kernel_size (int): the size of the kernel
            stride (int): the stride of the t-convolution
            padding (int): the padding of the t-convolution
            final_layer (bool): if the layer is the final layer
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output_dim),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Tanh()
            )

    def unsqueeze_noise(self, z):
        """
        Unsqueezing the noise vector

        Args:
            z (torch.Tensor): the noise vector
        """
        return z.view(len(z), self.z_dim, 1, 1)

    def forward(self, z):
        """
        Forward pass of the network

        Args:
            z (torch.Tensor): the z vector
        """
        z = self.unsqueeze_noise(z)
        return self.gen(z)

def make_noise(batch_size, z_dim, device='cpu'):
    """
    Making a noise vector

    Args:
        batch_size (int): the number of the batch
        z_dim (int): dimension of the noise vector
    """
    return torch.randn(batch_size, z_dim, device=device)


if __name__ == '__main__':
    # Testing the Generator class
    gen = Generator(z_dim=10, img_chan=1, hidden_dim=64)
    z = make_noise(batch_size=2, z_dim=10)
    gen_pred = gen(z)
    print(gen_pred.shape)
