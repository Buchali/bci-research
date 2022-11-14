from torch import nn

def weights_init(m):
    """
    Initializing the weights of the network

    Args:
        m (nn.Module): the module
    """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.2)

    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.2)
        nn.init.constant_(m.bias, 0)