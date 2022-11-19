from torch import nn


class Critic(nn.Module):
    """
    Classifier Network

    Args:
        img_chan (int): the number of the output image channels
        hidden_dim (int): the number of the hidden units
    """
    def __init__(self, im_chan=1, hidden_dim=32):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            self.make_critic_block(4*hidden_dim, 8*hidden_dim),
            self.make_critic_block(8*hidden_dim, 1, kernel_size=2, padding=0, final_layer=True)
        )

    def make_critic_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        """
        Making a critic block

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
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2),
                )
        else:
                return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding),
                )

    def forward(self, feature_map):
        """
        Forward pass of the network

        Args:
            feature_map (torch.Tensor): the input features.
        """
        pred_source = self.critic(feature_map)
        return pred_source.view(-1)
