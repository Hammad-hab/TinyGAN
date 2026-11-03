from torch import nn
import torch.nn.functional as F

class ResNetEncoderBlock(nn.Module):
    """
        For texture generation we use ResNet diffusion. Each block inside the
        Diffusion model is either a resnet encoder block or a decoder block.
        The benefit of using resnet is that gradients flow freely and don't
        disappear gradually over training. This is because we include a skip feature
        that adds the original image to the output of the block. 

        Here the general flow is
        x0 (Image) (*Skip)
        Conv2D (Decreases the size by 2)
        BatchNorm
        [SiLU]
        Linear (Time Injector, includes the time parameter into the net)
        Conv2D (Size stays the same)
        BatchNorm
        Add(x2, Conv2D(*Skip))

        We included a Conv2D for the skip connections so that the dimensions 
        of the input image and x2 (output of the second layer) match.
    """
    def __init__(self, in_channels, out_channels, time_emb_d):
        self.c1 = nn.Conv2d(in_channels, out_channels, 3, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels,)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.time_injector = nn.Linear(time_emb_d, out_channels)
        self.skip_feature = nn.Conv2d(in_channels, out_channels, 3, 2, padding=0)

    def forward(self, x0, time):
        x1 = F.silu(self.bn1(self.c1(x0)))
        x1 = x1 + self.time_injector(time)[:, :, None, None];
        x2 = (self.bn2(self.c2(x1)))
        return F.silu(x2 + self.skip_feature(x0))
