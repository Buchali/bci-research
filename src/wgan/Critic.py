import torch
from torch import nn



class Critic(nn.Module):
    """
    Critic Class

    Args:
        img_chan (int): the number of the input image channels
        hidden_dim (int): the number of the hidden units
    """
    def __init__(self, img_chan=1, hidden_dim=32):
        super(Critic, self).__init__()

        # The Network
        self.critic = nn.Sequential(
            self.make_critic_block(img_chan, hidden_dim),
            self.make_critic_block(hidden_dim, 2*hidden_dim),
            self.make_critic_block(2*hidden_dim, 4*hidden_dim),
            self.make_critic_block(4*hidden_dim, 8*hidden_dim),
            self.make_critic_block(8*hidden_dim, img_chan, kernel_size=2, padding=0, final_layer=True)
        )

    def make_critic_block(self, input_dim, output_dim, kernel_size=4, stride=2, padding=1, final_layer=False):
        """
        Making a critic block

        Args:
            input_dim (int): the number of the input channels
            output_dim (int): the number of the output channels
            kernel_size (int): the size of the kernel
            stride (int): the stride of the convolution
            padding (int): the padding of the convolution
            final_layer (bool): if the layer is the final layer
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
            )

    def forward(self, img):
        """
        Forward pass of the network

        Args:
            img (torch.Tensor): the input image
        """
        critic_pred = self.critic(img)
        return critic_pred.view(len(critic_pred), -1)


if __name__ == '__main__':
    # Testing the Critic model
    img = torch.randn(1, 1, 32, 32)
    critic = Critic()
    critic_pred = critic(img)
    print(critic_pred.shape)
