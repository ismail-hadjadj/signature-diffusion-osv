"""
train.py - Unified Training CLI for Offline Signature Verification
Supports:
  - verifier: Stage 3 curriculum metric learning for Siamese ViT
  - diffusion: Stage 1 stroke-conditioned latent diffusion training
  - negatives: Stage 2 synthetic hard negative generation & discrepancy filtering
  - full: Complete end-to-end 3-stage training pipeline
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Pipeline for Neuromotor Kinematic Offline Signature Verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="verifier",
        choices=["verifier", "diffusion", "negatives", "full"],
        help="Training stage to execute: 'verifier' (Siamese ViT), 'diffusion' (ControlNet LDM), 'negatives' (Hard negative mining), or 'full' (all 3 stages)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cedar.yaml",
        help="Path to YAML configuration file (e.g., configs/cedar.yaml, configs/icdar2011.yaml)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold index (0-4 for 5-fold CV)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Custom path to save the best model checkpoint (.pth)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" NEUROMOTOR KINEMATIC OFFLINE SIGNATURE VERIFICATION")
    print(f" Stage: {args.stage.upper()} | Config: {args.config} | Fold: {args.fold}")
    print("=" * 70)

    if args.stage in ["diffusion", "full"]:
        print("\n>>> Executing Stage 1: ControlNet Latent Diffusion Stroke Synthesizer Training...")
        from src.training.train_diffusion import train_diffusion_generator
        train_diffusion_generator(config_path=args.config)

    if args.stage in ["negatives", "full"]:
        print("\n>>> Executing Stage 2: Hard Negative Generation & Discrepancy Filtering (S_diff)...")
        from src.training.generate_negatives import generate_and_rank_hard_negatives
        generate_and_rank_hard_negatives(config_path=args.config)

    if args.stage in ["verifier", "full"]:
        print("\n>>> Executing Stage 3: Siamese ViT Metric Learning with Curriculum Contrastive Loss...")
        from src.training.train_verifier import train_verifier
        best_ckpt = train_verifier(
            config_path=args.config,
            fold=args.fold,
            epochs_override=args.epochs,
            batch_size_override=args.batch_size,
            save_checkpoint_override=args.save_checkpoint
        )
        print(f"\nTraining successfully finished! Best model saved to: {best_ckpt}")

if __name__ == "__main__":
    main()
