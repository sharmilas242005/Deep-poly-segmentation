"""
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (Xie et al., NeurIPS 2021).
PyTorch implementation featuring:
- Hierarchical Mix Transformer (MiT) Encoder with Overlapped Patch Merging & Mix-FFN
- Lightweight All-MLP Decoder fusing multi-scale feature stages
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class OverlapPatchEmbed(nn.Module):
    """Image / Feature to Patch Embedding with Overlap"""
    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        # Flatten spatial dimensions for Transformer blocks: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Efficient Multi-Head Self-Attention with Spatial Reduction"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, sr_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    """Mix-FeedForward Network with 3x3 Depthwise Convolution for Positional Encoding"""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        # Spatial convolution for positional bias
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Mix Transformer Encoder Block"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class MiTEncoder(nn.Module):
    """Mix Transformer (MiT-B0 configuration)"""
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: List[int] = [32, 64, 160, 256],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[float] = [4, 4, 4, 4],
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1]
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # Patch Embeddings
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stage Blocks
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0]) for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1]) for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2]) for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3]) for _ in range(depths[3])
        ])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []

        # Stage 1: H/4, W/4
        x, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H1, W1)
        x1 = self.norm1(x).transpose(1, 2).view(B, self.embed_dims[0], H1, W1)
        outs.append(x1)

        # Stage 2: H/8, W/8
        x, H2, W2 = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H2, W2)
        x2 = self.norm2(x).transpose(1, 2).view(B, self.embed_dims[1], H2, W2)
        outs.append(x2)

        # Stage 3: H/16, W/16
        x, H3, W3 = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H3, W3)
        x3 = self.norm3(x).transpose(1, 2).view(B, self.embed_dims[2], H3, W3)
        outs.append(x3)

        # Stage 4: H/32, W/32
        x, H4, W4 = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H4, W4)
        x4 = self.norm4(x).transpose(1, 2).view(B, self.embed_dims[3], H4, W4)
        outs.append(x4)

        return outs


class SegFormerHead(nn.Module):
    """All-MLP Decoder Head for SegFormer"""
    def __init__(self, in_channels: List[int], embedding_dim: int = 128, num_classes: int = 1):
        super().__init__()
        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        h, w = c1.shape[2], c1.shape[3]

        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=(h, w), mode="bilinear", align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=(h, w), mode="bilinear", align_corners=False)
        _c4 = F.interpolate(self.linear_c4(c4), size=(h, w), mode="bilinear", align_corners=False)

        fused = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        fused = self.linear_fuse(fused)
        logits = self.classifier(fused)

        # Upsample directly to target image size (H, W)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


class SegFormer(nn.Module):
    """
    SegFormer: Transformer-Based Semantic Segmentation for Medical Imaging.
    Comprises Hierarchical MiT Encoder and All-MLP Decoder.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dims: List[int] = [32, 64, 160, 256],
        decoder_dim: int = 128
    ):
        super().__init__()
        self.encoder = MiTEncoder(in_chans=in_channels, embed_dims=embed_dims)
        self.decoder = SegFormerHead(in_channels=embed_dims, embedding_dim=decoder_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        features = self.encoder(x)
        logits = self.decoder(features, target_size=(orig_h, orig_w))
        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()
