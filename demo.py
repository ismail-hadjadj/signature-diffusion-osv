"""
demo.py - Pairwise Signature Verification Inference Demo
Compares two signature images and predicts whether they belong to the same writer.
Outputs normalized Euclidean distance d_E, Cosine dissimilarity d_C, and verification verdict.
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.models.siamese_vit import SiameseViT
from src.dataset.cedar_loader import load_and_preprocess_signature

def verify_pair(
    img_path1: str,
    img_path2: str,
    checkpoint_path: str = None,
    threshold: float = 0.85,
    device: str = "cpu",
    save_vis: str = None
):
    """
    Performs pairwise biometric verification on two signature image files.
    """
    if not os.path.exists(img_path1):
        raise FileNotFoundError(f"Signature 1 not found: {img_path1}")
    if not os.path.exists(img_path2):
        raise FileNotFoundError(f"Signature 2 not found: {img_path2}")

    print(f"Loading Signature 1: {img_path1}")
    print(f"Loading Signature 2: {img_path2}")

    # 1. Preprocess images
    t1 = load_and_preprocess_signature(img_path1, target_size=(224, 224))
    t2 = load_and_preprocess_signature(img_path2, target_size=(224, 224))

    tensor1 = torch.from_numpy(t1).unsqueeze(0).to(device)
    tensor2 = torch.from_numpy(t2).unsqueeze(0).to(device)

    # 2. Load model
    model = SiameseViT(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        in_channels=3,
        embedding_dim=128,
        img_size=224
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Note: No checkpoint provided or not found. Running with initialized weights for demonstration.")

    model.to(device)
    model.eval()

    # 3. Forward pass & distance computation
    with torch.no_grad():
        dist_euc, z1, z2 = model(tensor1, tensor2)
        d_E = float(dist_euc.item())
        # Cosine distance: d_C = 0.5 * d_E^2 (hyperspherical duality)
        d_C = 0.5 * (d_E ** 2)
        similarity = float(torch.sum(z1 * z2, dim=-1).item())

    # 4. Decision rule
    verdict = "GENUINE MATCH" if d_E <= threshold else "FORGERY / NON-MATCH"
    confidence = max(0.0, min(100.0, (1.0 - (d_E / (2.0 * threshold))) * 100.0))

    print("\n" + "=" * 55)
    print(" VERIFICATION RESULTS")
    print("=" * 55)
    print(f"  Decision Verdict:     {verdict}")
    print(f"  Euclidean Distance:   d_E = {d_E:.4f} (Threshold tau = {threshold:.4f})")
    print(f"  Cosine Dissimilarity: d_C = {d_C:.4f}")
    print(f"  Cosine Similarity:    cos(theta) = {similarity:.4f}")
    print(f"  Match Confidence:     {confidence:.1f}%")
    print("=" * 55)

    # 5. Optional visualization
    if save_vis:
        im1_raw = cv2.imread(img_path1)
        im2_raw = cv2.imread(img_path2)
        h_target = 180
        w1 = int(im1_raw.shape[1] * (h_target / im1_raw.shape[0]))
        w2 = int(im2_raw.shape[1] * (h_target / im2_raw.shape[0]))
        im1_res = cv2.resize(im1_raw, (w1, h_target))
        im2_res = cv2.resize(im2_raw, (w2, h_target))

        vis_canvas = np.ones((h_target + 80, w1 + w2 + 60, 3), dtype=np.uint8) * 255
        vis_canvas[20:20+h_target, 20:20+w1] = im1_res
        vis_canvas[20:20+h_target, 40+w1:40+w1+w2] = im2_res

        color = (34, 139, 34) if verdict == "GENUINE MATCH" else (0, 0, 205)
        text = f"{verdict} | d_E = {d_E:.4f} (tau={threshold:.2f})"
        cv2.putText(vis_canvas, text, (20, h_target + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(save_vis, vis_canvas)
        print(f"Visualization saved to: {save_vis}")

    return {
        "verdict": verdict,
        "euclidean_distance": d_E,
        "cosine_dissimilarity": d_C,
        "similarity": similarity,
        "threshold": threshold
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise Signature Verification Demo")
    parser.add_argument("--img1", type=str, required=True, help="Path to first signature image")
    parser.add_argument("--img2", type=str, required=True, help="Path to second signature image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Decision threshold tau (default: 0.85)")
    parser.add_argument("--save_vis", type=str, default=None, help="Path to save comparison visualization image")
    parser.add_argument("--device", type=str, default="cuda" if torch and torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    verify_pair(
        img_path1=args.img1,
        img_path2=args.img2,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
        save_vis=args.save_vis
    )
