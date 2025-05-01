"""
Code adapted from https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, sigmoid_bool = False):
        super(UNet, self).__init__()

        self.sigmoid_bool = sigmoid_bool

        # Encoder Path
        self.enc1 = self.double_conv(num_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Decoder Path
        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.double_conv(1024 + 512, 512)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.double_conv(512 + 256, 256)

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.double_conv(256 + 128, 128)

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.double_conv(128 + 64, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """
        Defines a double convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Final Output
        out = self.final_conv(d1)
        if self.sigmoid_bool: 
            return torch.sigmoid(out)
        else: 
            return out

