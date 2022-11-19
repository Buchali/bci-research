from torch import nn


class Generator(nn.Module):
    """
    Generator Network.

    Args:
        z_dim (int): dimension of the noise vector
        img_chan (int): the number of the output image channels
        hidden_dim (int): the number of the hidden units
    """
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, 4*hidden_dim),
            self.make_gen_block(4*hidden_dim, 2*hidden_dim, padding=1),
            self.make_gen_block(2*hidden_dim, hidden_dim, padding=1),
            self.make_gen_block(hidden_dim, im_chan, padding=1, final_layer=True)
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, final_layer=False):
        """
        Making a generator block

        Args:
            input_channels (int): the number of the input channels
            output_channels (int): the number of the output channels
            kernel_size (int): the size of the kernel
            stride (int): the stride of the t-convolution
            padding (int): the padding of the t-convolution
            final_layer (bool): if the layer is the final layer
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding),
                    nn.Tanh(),
                )

    def unsqueeze_noise(self, noise):
        """
        Unsqueezing the noise vector

        Args:
            z (torch.Tensor): the noise vector
        """
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        """
        Forward pass of the network

        Args:
            z (torch.Tensor): the z vector
        """
        noise = self.unsqueeze_noise(noise)
        return self.gen(noise)
