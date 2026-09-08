# Model Checkpoints Guide

This directory stores trained model weights (`.pth`) for the Siamese Vision Transformer verifier and ControlNet Latent Diffusion stroke generator.

---

## 1. Directory Structure

```
checkpoints/
├── best_verifier_cedar.pth           # Best Siamese ViT verifier on CEDAR 5-fold CV
├── best_verifier_icdar2011_dutch.pth # Best verifier on ICDAR 2011 Dutch
├── best_verifier_icdar2011_chinese.pth # Best verifier on ICDAR 2011 Chinese
└── README.md
```

---

## 2. Checkpoint Format

Checkpoints save standard PyTorch state dictionaries (`state_dict`):

```python
import torch
from src.models.siamese_vit import SiameseViT

# Initialize architecture
model = SiameseViT(
    model_name="vit_tiny_patch16_224",
    in_channels=3,
    embedding_dim=128,
    img_size=224
)

# Load weights
state_dict = torch.load("checkpoints/best_verifier_cedar.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

---

## 3. Training Checkpoints from Scratch

To train and automatically save best checkpoints:

```bash
# Train CEDAR verifier
python train.py --config configs/cedar.yaml --save_checkpoint checkpoints/best_verifier_cedar.pth

# Train ICDAR 2011 Dutch verifier
python train.py --config configs/icdar2011_dutch.yaml --save_checkpoint checkpoints/best_verifier_icdar2011_dutch.pth
```
