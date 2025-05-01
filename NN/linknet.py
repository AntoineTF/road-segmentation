"""
Code based on the following work: https://github.com/snakers4/spacenet-three/tree/master
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from NN.linknet_tools.common_module import Dblock, DecoderBlock, nonlinearity
from NN.linknet_tools.non_local_operations.embedded_gaussian import NONLocalBlock2D_EGaussian


class NL_LinkNet_EGaussian(nn.Module): 
    def __init__(self, num_classes=1, num_channels=3,sigmoid_bool = True):
        super(NL_LinkNet_EGaussian, self).__init__()

        filters = (64, 128, 256, 512)
        resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.firstconv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=True)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.nonlocal3 = NONLocalBlock2D_EGaussian(128)
        self.encoder3 = resnet.layer3
        self.nonlocal4 = NONLocalBlock2D_EGaussian(256)
        self.encoder4 = resnet.layer4

        # SEB Modules
        self.seb_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.seb_conv3 = nn.Conv2d(256, 128, 3, padding=1)
        self.seb_conv4 = nn.Conv2d(512, 256, 3, padding=1)
        self.seb_us = nn.UpsamplingBilinear2d(scale_factor=2)

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(
            filters[0], 32, kernel_size=3, stride=1, padding=0
        )  
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=0)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.nonlocal3(e2)
        e3 = self.encoder3(e3)
        e4 = self.nonlocal4(e3)
        e4 = self.encoder4(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, sigmoid_bool = True):
        super(LinkNet34, self).__init__()

        filters = (64, 128, 256, 512)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1  
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  
        self.encoder2 = resnet.layer2  
        self.encoder3 = resnet.layer3 
        self.encoder4 = resnet.layer4  

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(
            filters[0], 32, kernel_size=3, stride=1, padding=0
        )  
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=0)  
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)  

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1,num_channels=3, sigmoid_bool = True):
        super(DLinkNet34, self).__init__()

        filters = (64, 128, 256, 512)
        resnet = models.resnet18(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)