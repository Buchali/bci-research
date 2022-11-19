from torch import nn


class Classifier(nn.Module):
    """
    Classifier Network

    Args:
        img_chan (int): the number of the output image channels
        hidden_dim (int): the number of the hidden units
    """
    def __init__(self, im_chan=1, hidden_dim=32):
        super(Classifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            self.make_classifier_block(im_chan, hidden_dim),
            self.make_classifier_block(hidden_dim, 2*hidden_dim),
            self.make_classifier_block(2*hidden_dim, 4*hidden_dim),

        )

        self.classifier = nn.Sequential(
            self.make_classifier_block(4*hidden_dim, 8*hidden_dim),
            self.make_classifier_block(8*hidden_dim, 1, kernel_size=2, padding=0, final_layer=True)
        )

    def make_classifier_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        """
        Making a classifier block

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
                nn.Sigmoid(),
                )
    def forward(self, img):
        """
        Forward pass of the network

        Args:
            img (torch.Tensor): the input image
        """
        feature_map = self.feature_extractor(img)
        pred_class = self.classifier(feature_map)
        return pred_class.view(-1), feature_map
