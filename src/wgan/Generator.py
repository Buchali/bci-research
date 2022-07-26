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
            self.make_gen_block(z_dim, 4*hidden_dim, padding=0),
            self.make_gen_block(4*hidden_dim, 2*hidden_dim),
            self.make_gen_block(2*hidden_dim, hidden_dim),
            self.make_gen_block(hidden_dim, img_chan, final_layer=True)
        )

    def make_gen_block(self, input_dim, img_chan, kernel_size=4, stride=2, padding=1, final_layer=False):
        """
        Making a generator block

        Args:
            input_dim (int): the number of the input channels
            img_chan (int): the number of the output channels
            kernel_size (int): the size of the kernel
            stride (int): the stride of the t-convolution
            padding (int): the padding of the t-convolution
            final_layer (bool): if the layer is the final layer
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dim, img_chan, kernel_size, stride, padding),
                nn.BatchNorm2d(img_chan),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_dim, img_chan, kernel_size, stride, padding)
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
        return self.gen(z)
