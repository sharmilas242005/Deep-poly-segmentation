"""
LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation (Chaurasia et al., 2017).
Authentic PyTorch implementation with residual encoder blocks and additive feature fusion decoders.
"""

import torch
import torch.nn as nn
from typing import List


class BasicBlock(nn.Module):
    """Residual block for LinkNet encoder"""
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class LinkNetEncoderBlock(nn.Module):
    """Two residual blocks composing one LinkNet encoder stage"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block1 = BasicBlock(in_channels, out_channels, stride=2)
        self.block2 = BasicBlock(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class LinkNetDecoderBlock(nn.Module):
    """
    LinkNet decoder block:
    1x1 Conv (channel reduction) -> 3x3 ConvTranspose2d (upsample) -> 1x1 Conv (restore)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = in_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


class LinkNet(nn.Module):
    """
    LinkNet Architecture for Fast Semantic Segmentation.
    Employs additive skip connections instead of memory-intensive concatenation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, filters: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial stem
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoders 1 to 4
        self.encoder1 = nn.Sequential(
            BasicBlock(filters[0], filters[0]),
            BasicBlock(filters[0], filters[0])
        )
        self.encoder2 = LinkNetEncoderBlock(filters[0], filters[1])
        self.encoder3 = LinkNetEncoderBlock(filters[1], filters[2])
        self.encoder4 = LinkNetEncoderBlock(filters[2], filters[3])

        # Decoders 4 down to 1 (with additive skip fusion)
        self.decoder4 = LinkNetDecoderBlock(filters[3], filters[2])
        self.decoder3 = LinkNetDecoderBlock(filters[2], filters[1])
        self.decoder2 = LinkNetDecoderBlock(filters[1], filters[0])
        self.decoder1 = LinkNetDecoderBlock(filters[0], filters[0])

        # Final reconstruction head (H/2 -> H)
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stem: H -> H/2 -> H/4
        x = self.init_conv(x)
        x = self.maxpool(x)

        # Encoders
        e1 = self.encoder1(x)   # H/4, C=64
        e2 = self.encoder2(e1)  # H/8, C=128
        e3 = self.encoder3(e2)  # H/16, C=256
        e4 = self.encoder4(e3)  # H/32, C=512

        # Decoders with ADDITIVE skip connections
        d4 = self.decoder4(e4) + e3  # H/16
        d3 = self.decoder3(d4) + e2  # H/8
        d2 = self.decoder2(d3) + e1  # H/4
        d1 = self.decoder1(d2)       # H/2

        # Final upsampling to original H, W
        out = self.final_deconv(d1)
        out = self.final_conv(out)
        logits = self.final_classifier(out)

        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()
