"""
The work in this code is from https://github.com/snakers4/spacenet-three/
"""

from functools import partial

import torch.nn as nn
import torch.nn.functional as F

# Define a partial function for ReLU with in-place operation
nonlinearity = partial(F.relu, inplace=True)


# ----------------------------------------------------------
# Dblock: Dilated Convolution Block for Multi-scale Context
# ----------------------------------------------------------
class Dblock(nn.Module):
    """
    Dilated Convolution Block to capture multi-scale context.
    This block applies 4 convolutions with increasing dilation rates (1, 2, 4, 8) 
    to expand the receptive field without reducing feature map resolution. 
    The outputs are summed with the input (residual connection).
    """
    def __init__(self, channel):
        super(Dblock, self).__init__()

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass through the Dblock.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor after summing the input and dilated convolutions.
        """
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out

# ----------------------------------------------------------
# DecoderBlock: Upsampling and Feature Refinement
# ----------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    Decoder Block for upsampling feature maps and refining them.
    This block reduces channels using a 1x1 convolution, upsamples using a 
    transposed convolution, and refines features with additional convolutions.
    Commonly used in segmentation models like U-Net.
    """
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )  # stride was 2
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        """
        Forward pass through the DecoderBlock.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Upsampled and refined feature map.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
