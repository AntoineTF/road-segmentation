"""
Code adapted from https://github.com/tkuanlun350/Tensorflow-SegNet/tree/master
Adapted from TensorFlow based implementation
The Local Response Normalization was replace by Batch Normalization which is much more common
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, sigmoid_bool=False):
        super(SegNet, self).__init__()
        self.sigmoid_bool = sigmoid_bool

        # Encoder
        self.enc1 = self._conv_block(num_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder
        self.upool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256)

        self.upool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)

        self.upool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)

        self.upool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # Final output layer for binary classification
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output channel set to 1
        )
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x, indices1 = self.pool1(self.enc1(x))
        x, indices2 = self.pool2(self.enc2(x))
        x, indices3 = self.pool3(self.enc3(x))
        x, indices4 = self.pool4(self.enc4(x))

        # Decoder
        x = self.upool4(x, indices4)
        x = self.dec4(x)

        x = self.upool3(x, indices3)
        x = self.dec3(x)

        x = self.upool2(x, indices2)
        x = self.dec2(x)

        x = self.upool1(x, indices1)
        x = self.dec1(x)

        if self.sigmoid_bool:
            return torch.sigmoid(x)
        else:
            return x
