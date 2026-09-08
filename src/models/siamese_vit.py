"""
siamese_vit.py - Vision Transformer (ViT) Backbone using timm with L2-Normalized Embeddings
Implements a twin Siamese Vision Transformer for offline signature verification.
Uses pretrained ViT-Tiny or ViT-Small from timm, projecting to an L2-normalized hypersphere embedding.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = object
    TORCH_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    timm = None
    TIMM_AVAILABLE = False


# ==============================================================================
# Built-in Lightweight ViT-Tiny / ViT-Small Fallback Architecture
# ==============================================================================

if TORCH_AVAILABLE:
    class PatchEmbedding(nn.Module):
        def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192):
            super().__init__()
            self.n_patches = (img_size // patch_size) ** 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x).flatten(2).transpose(1, 2)

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int = 3, dropout: float = 0.0):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (self.drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
            return self.drop(self.proj(x))

    class ViTBlock(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)
            self.norm2 = nn.LayerNorm(embed_dim)
            hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class FallbackViT(nn.Module):
        """Pure-PyTorch ViT-Tiny (192-dim, 12 layers) or ViT-Small (384-dim, 12 layers)."""
        def __init__(self, img_size: int = 224, in_chans: int = 3, embed_dim: int = 192, depth: int = 12, num_heads: int = 3):
            super().__init__()
            self.patch_embed = PatchEmbedding(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
            self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads=num_heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)[:, 0]


# ==============================================================================
# Siamese Vision Transformer with L2-Normalized Metric Head
# ==============================================================================

ModuleBase = nn.Module if TORCH_AVAILABLE else object

class SiameseViT(ModuleBase):
    """
    Twin Siamese Vision Transformer using timm ViT-Tiny or ViT-Small.
    Maps signature images to strictly L2-normalized embedding vectors on a unit hypersphere.
    
    Supported Models:
      - 'vit_tiny_patch16_224' (embed_dim = 192)
      - 'vit_small_patch16_224' (embed_dim = 384)
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        img_size: int = 224,
        freeze_prefix: bool = True
    ):
        if not TORCH_AVAILABLE:
            super().__init__()
            return

        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        # 1. Initialize ViT Backbone (via timm if available, otherwise native fallback)
        if TIMM_AVAILABLE:
            try:
                # timm pretrained ViT-Tiny or ViT-Small
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    in_chans=in_channels,
                    num_classes=0 # removes classification head, outputs pooled token
                )
                feat_dim = self.backbone.num_features
            except Exception as e:
                print(f"[Warning] timm model creation failed ({e}), falling back to internal ViT...")
                feat_dim = 192 if "tiny" in model_name else 384
                heads = 3 if "tiny" in model_name else 6
                self.backbone = FallbackViT(img_size=img_size, in_chans=in_channels, embed_dim=feat_dim, num_heads=heads)
        else:
            feat_dim = 192 if "tiny" in model_name else 384
            heads = 3 if "tiny" in model_name else 6
            self.backbone = FallbackViT(img_size=img_size, in_chans=in_channels, embed_dim=feat_dim, num_heads=heads)

        # 2. Metric Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )

        # 3. Freeze patch embedding layer and first 4 transformer blocks
        if freeze_prefix:
            self.freeze_prefix_layers(freeze_blocks=4)

    def freeze_prefix_layers(self, freeze_blocks: int = 4):
        """
        Freezes the patch embedding layer and the first `freeze_blocks` transformer blocks of vit_tiny_patch16_224.
        By default (freeze_blocks=4), freezes blocks 0 to 3, leaving the upper 8 transformer blocks
        (blocks 4 to 11), final norm, and projection head fully trainable.
        """
        if not TORCH_AVAILABLE:
            return

        # Freeze patch embedding & position parameters
        if hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self.backbone, "cls_token") and isinstance(self.backbone.cls_token, (torch.Tensor, nn.Parameter)):
            self.backbone.cls_token.requires_grad = False
        if hasattr(self.backbone, "pos_embed") and isinstance(self.backbone.pos_embed, (torch.Tensor, nn.Parameter)):
            self.backbone.pos_embed.requires_grad = False

        # Freeze first `freeze_blocks` transformer blocks (blocks 0 to freeze_blocks - 1)
        if hasattr(self.backbone, "blocks"):
            for i, blk in enumerate(self.backbone.blocks):
                if i < freeze_blocks:
                    for p in blk.parameters():
                        p.requires_grad = False
                else:
                    for p in blk.parameters():
                        p.requires_grad = True

        # Unfreeze final LayerNorm if present
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        # Projection head parameters always train
        for p in self.projection_head.parameters():
            p.requires_grad = True

    def get_trainable_parameter_summary(self) -> dict:
        """Returns parameter count statistics for frozen and trainable layers."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts raw backbone representation."""
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            if hasattr(self.backbone, "forward_head"):
                return self.backbone.forward_head(feats, pre_logits=True)
            elif isinstance(feats, torch.Tensor) and feats.ndim == 3:
                return feats[:, 0]
            return feats
        return self.backbone(x)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes a single signature through the ViT backbone and projection head.
        Returns L2-normalized embedding z where ||z||_2 = 1.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")

        # Ensure correct channel dimension (e.g. 1 channel grayscale)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1] != self.in_channels:
            if self.in_channels == 1 and x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True)
            elif self.in_channels == 3 and x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

        feats = self.extract_features(x)
        embeddings = self.projection_head(feats)
        
        # Strict L2 normalization onto unit hypersphere: ||z||_2 = 1
        l2_embeddings = F.normalize(embeddings, p=2, dim=-1)
        return l2_embeddings

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Siamese forward pass for a pair of signatures.
        
        Returns:
            dist: Euclidean distance ||z1 - z2||_2 in [0, 2]
            z1: L2-normalized embedding for img1
            z2: L2-normalized embedding for img2
        """
        z1 = self.forward_one(img1)
        z2 = self.forward_one(img2)
        
        # Enforce explicit L2 normalization on projection embeddings before computing pairwise distances
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        # Ensure Euclidean distance is bounded on the unit hypersphere: ||z1 - z2||_2 in [0, 2]
        dist = torch.norm(z1 - z2, p=2, dim=-1)
        return dist, z1, z2

    def compute_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Computes Cosine Similarity between signature pair:
        Since embeddings are L2-normalized: CosSim(z1, z2) = 1 - 0.5 * ||z1 - z2||^2.
        """
        z1 = self.forward_one(img1)
        z2 = self.forward_one(img2)
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        return F.cosine_similarity(z1, z2, dim=-1)
