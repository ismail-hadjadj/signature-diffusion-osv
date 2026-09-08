"""
train_diffusion.py - Phase 1: Train Latent Diffusion Model via ControlNet using Hugging Face diffusers
Conditions stroke diffusion on morphological skeleton maps and curvature fields.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import yaml
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from diffusers.optimization import get_cosine_schedule_with_warmup
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.dataset.cedar_loader import CEDARPairDataset
from src.dataset.icdar_loader import ICDAR2011PairDataset
from src.models.controlnet_ldm import SignatureControlNetLDM, DIFFUSERS_AVAILABLE

def train_diffusion(config_path: str, fold: int = 0):
    if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
        print("[Notice] PyTorch and Hugging Face 'diffusers' are required to train the diffusion model.")
        print("To install dependencies: pip install torch torchvision diffusers transformers accelerate")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1] Training Hugging Face diffusers ControlNet-LDM on: {device}")

    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"]["data_dir"]
    target_size = (cfg["image"]["height"], cfg["image"]["width"])
    batch_size = cfg["diffusion"]["batch_size"]
    lr = float(cfg["diffusion"]["learning_rate"])
    epochs = cfg["diffusion"]["epochs"]
    save_dir = cfg["diffusion"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1. Dataset Loader
    if dataset_name == "CEDAR":
        train_ds = CEDARPairDataset(
            data_dir=data_dir,
            fold=fold,
            is_train=True,
            target_size=target_size,
            extract_conditions=True
        )
    else:
        train_ds = ICDAR2011PairDataset(
            data_dir=data_dir,
            is_train=True,
            target_size=target_size,
            extract_conditions=True
        )

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {dataset_name} | Training pairs: {len(train_ds)} | Batches/epoch: {len(loader)}")

    # 2. ControlNet LDM Model Setup
    ldm = SignatureControlNetLDM(
        latent_channels=4,
        condition_channels=3,
        block_channels=tuple(cfg["diffusion"]["controlnet_channels"]),
        timesteps=cfg["diffusion"]["timesteps"],
        device=str(device)
    )

    # Train ControlNet and UNet adapters
    trainable_params = list(ldm.controlnet.parameters()) + list(ldm.unet.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(loader) * epochs
    )

    print("\nStarting Diffusion Training Loop...")
    for epoch in range(1, epochs + 1):
        ldm.controlnet.train()
        ldm.unet.train()
        total_loss = 0.0

        for step, batch in enumerate(loader):
            # Signature image x_0 and skeleton condition c
            images = batch['img1'].to(device) # (B, 1, H, W)
            conditions = batch['cond1'].to(device) # (B, 3, H, W)
            b = images.shape[0]

            # 1. Encode image to latent space
            latents = ldm.encode_to_latents(images)

            # 2. Sample noise and random timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, ldm.timesteps, (b,), device=device).long()

            # 3. Add noise via diffusers scheduler
            noisy_latents = ldm.noise_scheduler.add_noise(latents, noise, timesteps)

            # 4. Predict noise with ControlNet-LDM
            model_pred = ldm.forward(
                latents=noisy_latents,
                timesteps=timesteps,
                skeleton_condition=conditions
            )

            # 5. MSE Diffusion Loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Diffusion MSE Loss: {avg_loss:.6f}")
            # Save diffusers-compatible weights
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(ckpt_path, exist_ok=True)
            ldm.controlnet.save_pretrained(os.path.join(ckpt_path, "controlnet"))
            ldm.unet.save_pretrained(os.path.join(ckpt_path, "unet"))

    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    ldm.controlnet.save_pretrained(os.path.join(final_dir, "controlnet"))
    ldm.unet.save_pretrained(os.path.join(final_dir, "unet"))
    print(f"[Phase 1 Complete] ControlNet-LDM saved to: {final_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ControlNet LDM via Hugging Face diffusers")
    parser.add_argument("--config", type=str, default="configs/cedar.yaml")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    train_diffusion(args.config, args.fold)
