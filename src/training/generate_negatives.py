"""
generate_negatives.py - Generate & Rank Synthetic Hard Negatives with Structural Discrepancy (S_diff)
Applies curvature-dependent hesitation noise to skeletonized strokes, synthesizes realistic stroke textures,
and evaluates each candidate using the Structural Discrepancy Score (S_diff) based on SSIM and Sobel gradient difference.
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import glob
import json
import yaml
import cv2
import numpy as np
from typing import Optional, Dict, List

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.preprocessing.skeletonize import extract_condition_tensor, extract_morphological_skeleton
from src.preprocessing.kinematic_tremor import inject_curvature_hesitation_noise
from src.models.controlnet_ldm import SignatureControlNetLDM, DIFFUSERS_AVAILABLE

def compute_structural_discrepancy(
    genuine_image: np.ndarray,
    synthetic_image: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5
) -> dict:
    """
    Computes the Structural Discrepancy Score:
        S_diff = alpha * (1.0 - SSIM(I_gen, I_synth)) + beta * NormalizedSobelGradientDiff(I_gen, I_synth)

    Args:
        genuine_image: (H, W) or (H, W, 1) float32 in [0, 1]
        synthetic_image: (H, W) or (H, W, 1) float32 in [0, 1]
        alpha: Weight for SSIM discrepancy
        beta: Weight for Sobel gradient discrepancy

    Returns:
        dict with 'S_diff', 'ssim', 'ssim_discrepancy', 'sobel_discrepancy'
    """
    gen = genuine_image.squeeze().astype(np.float32)
    syn = synthetic_image.squeeze().astype(np.float32)

    # 1. SSIM Discrepancy
    if SKIMAGE_AVAILABLE:
        # data_range=1.0 for normalized images
        val_ssim = float(ssim_fn(gen, syn, data_range=1.0))
    else:
        # Fallback SSIM calculation
        mu_x = cv2.GaussianBlur(gen, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(syn, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(gen * gen, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(syn * syn, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(gen * syn, (11, 11), 1.5) - mu_x * mu_y
        c1, c2 = (0.01) ** 2, (0.03) ** 2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        val_ssim = float(np.mean(ssim_map))

    ssim_discrepancy = float(np.clip(1.0 - val_ssim, 0.0, 2.0))

    # 2. Sobel Gradient Difference
    # Extract spatial gradient magnitudes
    sobel_x_gen = cv2.Sobel(gen, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y_gen = cv2.Sobel(gen, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_gen = np.sqrt(sobel_x_gen**2 + sobel_y_gen**2)

    sobel_x_syn = cv2.Sobel(syn, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y_syn = cv2.Sobel(syn, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_syn = np.sqrt(sobel_x_syn**2 + sobel_y_syn**2)

    # L1 norm of gradient difference normalized by max gradient
    max_g = max(float(np.max(grad_mag_gen)), float(np.max(grad_mag_syn)), 1e-5)
    abs_grad_diff = np.abs(grad_mag_gen - grad_mag_syn) / max_g
    sobel_discrepancy = float(np.mean(abs_grad_diff))

    # 3. Combined S_diff score
    s_diff = float(alpha * ssim_discrepancy + beta * sobel_discrepancy)

    return {
        'S_diff': s_diff,
        'ssim': float(val_ssim),
        'ssim_discrepancy': ssim_discrepancy,
        'sobel_discrepancy': sobel_discrepancy
    }

def generate_and_rank_hard_negatives(
    config_path: str,
    checkpoint_dir: Optional[str] = None,
    num_samples: int = 10,
    output_dir: str = "data/synthetic_hard_negatives"
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    target_size = (cfg["image"]["height"], cfg["image"]["width"])
    tremor_cfg = cfg["preprocessing"]["kinematic_tremor"]["sigma_lognormal"]

    print(f"[Phase 1 Inference] Generating synthetic hard negatives with S_diff evaluation...")

    # Load ControlNet-LDM if checkpoint and dependencies exist
    model = None
    if TORCH_AVAILABLE and DIFFUSERS_AVAILABLE and checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Loading trained ControlNet-LDM checkpoint from: {checkpoint_dir}")
        model = SignatureControlNetLDM(
            latent_channels=4,
            condition_channels=3,
            block_channels=tuple(cfg["diffusion"]["controlnet_channels"]),
            timesteps=cfg["diffusion"]["timesteps"]
        )
        controlnet_path = os.path.join(checkpoint_dir, "controlnet")
        unet_path = os.path.join(checkpoint_dir, "unet")
        if os.path.exists(controlnet_path):
            from diffusers import ControlNetModel, UNet2DConditionModel
            model.controlnet = ControlNetModel.from_pretrained(controlnet_path).to(model.device)
            model.unet = UNet2DConditionModel.from_pretrained(unet_path).to(model.device)
            model.controlnet.eval()
            model.unet.eval()

    # Find genuine training images
    data_dir = cfg["dataset"]["data_dir"]
    gen_candidates = glob.glob(os.path.join(data_dir, "**", "*original*.png"), recursive=True)
    if not gen_candidates:
        gen_candidates = glob.glob(os.path.join(data_dir, "**", "*Genuine*.png"), recursive=True)
    if not gen_candidates:
        gen_candidates = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)[:num_samples]

    records = []

    for idx, path in enumerate(gen_candidates[:num_samples]):
        raw_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            continue

        raw_resized = cv2.resize(raw_gray, (target_size[1], target_size[0]))
        _, binary = cv2.threshold(raw_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        gen_norm = binary.astype(np.float32) / 255.0

        # 1. Morphological thinning to 1-pixel skeleton
        skel = extract_morphological_skeleton(raw_resized)

        # 2. Inject curvature-dependent Gaussian hesitation noise
        pert_skel, pert_render = inject_curvature_hesitation_noise(
            skeleton=skel,
            base_noise_std=float(tremor_cfg.get("tremor_amplitude", 0.08) * 10),
            curvature_gain=2.5,
            seed=42 + idx
        )

        # 3. Stroke Texture Synthesis (ControlNet-LDM or Kinematic Synthesis)
        if model is not None and TORCH_AVAILABLE and DIFFUSERS_AVAILABLE:
            pert_cond = extract_condition_tensor((pert_skel * 255).astype(np.uint8), target_size=target_size)
            cond_t = torch.from_numpy(
                np.transpose(pert_cond['condition_composite'], (2, 0, 1))
            ).unsqueeze(0).float().to(model.device)
            synth_tensor = model.sample(cond_t, num_inference_steps=25)
            synth_norm = synth_tensor.squeeze().cpu().numpy()
        else:
            # High-fidelity kinematic stroke synthesis fallback
            stroke_blur = cv2.GaussianBlur(pert_render.astype(np.float32) / 255.0, (3, 3), 0.8)
            synth_norm = np.clip(stroke_blur * 1.3, 0.0, 1.0)

        # 4. Compute Structural Discrepancy Score S_diff (SSIM + Sobel)
        metrics = compute_structural_discrepancy(gen_norm, synth_norm, alpha=0.5, beta=0.5)
        s_diff = metrics['S_diff']

        # 5. Save synthetic hard negative image
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_filename = f"hard_neg_{base_name}_Sdiff_{s_diff:.4f}.png"
        out_path = os.path.join(output_dir, out_filename)

        # Invert for standard white-paper black-ink viewing
        save_img = (255.0 * (1.0 - synth_norm)).astype(np.uint8)
        cv2.imwrite(out_path, save_img)

        record = {
            'synthetic_file': out_path,
            'source_genuine': path,
            'S_diff': round(s_diff, 5),
            'ssim': round(metrics['ssim'], 5),
            'ssim_discrepancy': round(metrics['ssim_discrepancy'], 5),
            'sobel_discrepancy': round(metrics['sobel_discrepancy'], 5)
        }
        records.append(record)

    # 6. Rank candidates by S_diff (Hardest negatives have lower S_diff = closest structural fidelity)
    records.sort(key=lambda x: x['S_diff'])
    for rank, r in enumerate(records, start=1):
        r['hardness_rank'] = rank

    manifest_path = os.path.join(output_dir, "synthetic_negatives_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSuccessfully generated {len(records)} synthetic hard negatives with S_diff scores.")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 70)
    print(f"{'Rank':<6} {'File Name':<35} {'S_diff':<10} {'SSIM':<10} {'Sobel Diff':<10}")
    print("=" * 70)
    for r in records[:5]:
        fname = os.path.basename(r['synthetic_file'])
        print(f"{r['hardness_rank']:<6} {fname:<35} {r['S_diff']:<10.4f} {r['ssim']:<10.4f} {r['sobel_discrepancy']:<10.4f}")
    print("=" * 70)

    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and rank hard negative signatures via S_diff")
    parser.add_argument("--config", type=str, default="configs/cedar.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out", type=str, default="data/synthetic_hard_negatives")
    args = parser.parse_args()
    generate_and_rank_hard_negatives(args.config, args.checkpoint, args.num_samples, args.out)
