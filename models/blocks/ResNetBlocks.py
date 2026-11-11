from functools import partial

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
    def __init__(self, in_channels, out_channels, time_emb_d, stride, padding=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels,)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.time_injector = nn.Linear(time_emb_d, out_channels)
        self.skip_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x0, time):
        x1 = F.silu(self.bn1(self.c1(x0)))
        print(x1.shape, x0.shape)
        x1 = x1 + self.time_injector(time)[:, :, None, None];
        x2 = (self.bn2(self.c2(x1)))
        print(x2.shape, self.skip_feature(x0).shape)
        return F.silu(x2 + self.skip_feature(x0))


class ResNetDecoderBlock(nn.Module):
    """
            For texture generation we use ResNet diffusion. Each block inside the
            Diffusion model is either a resnet encoder block or a decoder block.
            The decoder block does the opposite job of the encoder: instead of
            shrinking the image down to find features, it grows it back up
            so we can rebuild the texture. We use ResNet here too, because we
            want gradients to flow freely through the decoder as well, and the skip
            feature adds the original input to the output of the block.
    
            Here the general flow is
            x0 (Feature map) (*Skip)
            ConvTranspose2D (Increases the size by 2)
            BatchNorm
            [SiLU]
            Linear (Time Injector, includes the time parameter into the net)
            ConvTranspose2D (Size stays the same)
            BatchNorm
            Add(x2, ConvTranspose2D(*Skip))
    
            We included a ConvTranspose2D for the skip connections so that the dimensions
            of the input and x2 (output of the second layer) match. It uses ksize=1 and the
            same stride as the main path, so it upsamples the input the same way.
        """
    def __init__(self, in_channels, out_channels, time_emb_d, stride=2, padding=0, output_padding=1) -> None:
        super().__init__()
        self.cd1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.cd2 =  nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.time_injector = nn.Linear(time_emb_d, out_channels)

        self.skip_feature = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x0, time):
        x1 = F.silu(self.bn1(self.cd1(x0)))
        x1 = x1 + self.time_injector(time,)[:, :, None, None]
        x2 = self.bn2(self.cd2(x1))
        return F.silu(x2 + self.skip_feature(x0))

ResNetBottleneck = partial(ResNetEncoderBlock, stride=1)