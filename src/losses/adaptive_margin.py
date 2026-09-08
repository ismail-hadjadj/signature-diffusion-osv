"""
adaptive_margin.py - Sigmoidal Adaptive Margin Contrastive Loss for Signature Verification
Implements a dynamic contrastive loss where the separation margin m(t) expands sigmoidally
over training epochs and scales with negative hardness (random, skilled, or synthetic S_diff).
"""

from __future__ import annotations
import math
from typing import Optional, List, Union, Dict
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

ModuleBase = nn.Module if TORCH_AVAILABLE else object


class SigmoidalAdaptiveMarginLoss(ModuleBase):
    """
    Adaptive Margin Contrastive Loss with Sigmoidal Temporal Expansion.
    
    Mathematical Formulation:
      Base margin m(t) expands over training epoch t:
        m(t) = m_min + (m_max - m_min) / (1 + exp(-lambda * (t - t_mid)))
        
      Per-sample adaptive margin:
        m_i(t) = m(t) * gamma(pair_type, S_diff)
        
      Contrastive Loss:
        L_i = (1/2) * y_i * D_i^2 + (1/2) * (1 - y_i) * max(0, m_i(t) - D_i)^2
    """

    def __init__(
        self,
        m_min: float = 0.60,
        m_max: float = 1.10,
        total_epochs: int = 15,
        steepness: float = 0.35,
        midpoint_ratio: float = 0.50,
        skilled_factor: float = 0.95,
        synthetic_factor: float = 1.00,
        random_factor: float = 1.05,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.total_epochs = total_epochs
        self.steepness = steepness
        self.midpoint_ratio = midpoint_ratio
        self.t_mid = total_epochs * midpoint_ratio

        self.skilled_factor = skilled_factor
        self.synthetic_factor = synthetic_factor
        self.random_factor = random_factor
        self.label_smoothing = label_smoothing

        self.current_epoch = 0
        self.current_base_margin = self._compute_sigmoidal_margin(1)

    def _compute_sigmoidal_margin(self, epoch: float) -> float:
        """Computes base margin m(t) via logistic sigmoid function expanding from m_min to m_max."""
        if self.total_epochs <= 1:
            return float(self.m_min)
        progress = (epoch - 1.0) / max(1.0, float(self.total_epochs - 1))
        x = (progress - self.midpoint_ratio) * 6.0
        x = max(-20.0, min(20.0, x))
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        margin = self.m_min + (self.m_max - self.m_min) * sigmoid
        return float(margin)

    def update_epoch(self, epoch: int, total_epochs: Optional[int] = None):
        """Updates the training epoch and refreshes the current sigmoidal margin m(t)."""
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
            self.t_mid = total_epochs * self.midpoint_ratio
        self.current_base_margin = self._compute_sigmoidal_margin(epoch)

    def get_current_margin(self) -> float:
        """Returns the current epoch's base margin."""
        return self.current_base_margin

    def forward(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
        pair_types: Optional[List[str]] = None,
        s_diff_scores: Optional[Union[torch.Tensor, List[float]]] = None
    ) -> torch.Tensor:
        """
        Contrastive Loss Distance Formulation with Label Smoothing:
          L_pos = (1 - eps) * ||z1 - z2||^2 + eps * max(0, margin - ||z1 - z2||)^2
          L_neg = (1 - eps) * max(0, margin - ||z1 - z2||)^2 + eps * ||z1 - z2||^2
        Guarantees strict 1:1 balance between positive and negative pair terms.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")

        device = distances.device
        b = distances.shape[0]

        # Base sigmoidal margin for current epoch
        base_m = self.current_base_margin
        margins = torch.full((b,), base_m, device=device, dtype=torch.float32)

        # Apply sample-level hardness scaling
        if pair_types is not None:
            multipliers = []
            for i, p_type in enumerate(pair_types):
                if p_type in ('skilled_forgery', 'skilled'):
                    mult = self.skilled_factor
                elif p_type in ('synthetic_hard_negative', 'synthetic', 'diffusion'):
                    mult = self.synthetic_factor
                    if s_diff_scores is not None and i < len(s_diff_scores):
                        s_val = float(s_diff_scores[i])
                        boost = max(0.0, 0.15 * (1.0 - min(1.0, s_val * 5.0)))
                        mult += boost
                elif p_type in ('random_negative', 'random'):
                    mult = self.random_factor
                else:
                    mult = 1.0
                multipliers.append(mult)

            mult_tensor = torch.tensor(multipliers, device=device, dtype=torch.float32)
            margins = margins * mult_tensor

        margin_delta = F.relu(margins - distances)
        eps = float(self.label_smoothing)

        # 1. Genuine Loss with Label Smoothing
        loss_pos = (1.0 - eps) * torch.square(distances) + eps * torch.square(margin_delta)

        # 2. Forgery / Non-Match Loss with Label Smoothing
        loss_neg = (1.0 - eps) * torch.square(margin_delta) + eps * torch.square(distances)

        pos_mask = (labels == 1.0)
        neg_mask = (labels == 0.0)

        n_pos = torch.sum(pos_mask)
        n_neg = torch.sum(neg_mask)

        # Compute separate means for positive and negative pairs to guarantee 1:1 gradient weighting
        pos_term = torch.sum(loss_pos * pos_mask.float()) / torch.clamp(n_pos, min=1.0) if n_pos > 0 else torch.tensor(0.0, device=device)
        neg_term = torch.sum(loss_neg * neg_mask.float()) / torch.clamp(n_neg, min=1.0) if n_neg > 0 else torch.tensor(0.0, device=device)

        total_loss = 0.5 * (pos_term + neg_term)
        return total_loss
