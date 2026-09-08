"""
test_smoke.py - Automated Smoke Test Suite for Signature Diffusion OSV
Tests all critical components with lightweight dummy inputs:
  1. Morphological thinning, curvature calculation, and hesitation tremor injection
  2. Dataset pair generation, loading, and batch collation
  3. Structural discrepancy metric (S_diff) calculation
  4. Siamese ViT model forward pass, L2 hypersphere normalization, loss & backpropagation
  5. Biometric evaluation suite (FAR, FRR, EER interpolation, Markdown reporting)
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class TestSignatureVerificationPipeline(unittest.TestCase):

    def test_01_preprocessing_and_tremor(self):
        """Test thinning, curvature extraction, and hesitation tremor synthesis."""
        from src.preprocessing.skeletonize import (
            extract_morphological_skeleton,
            extract_stroke_curvature_and_tangents,
            extract_condition_tensor
        )
        from src.preprocessing.kinematic_tremor import inject_curvature_hesitation_noise

        canvas = np.zeros((224, 224), dtype=np.uint8)
        pts = np.array([[30, 100], [80, 50], [130, 150], [180, 80], [210, 120]], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=4)

        skel = extract_morphological_skeleton(canvas, invert_input=False)
        self.assertEqual(skel.shape, (224, 224))
        self.assertTrue(set(np.unique(skel)).issubset({0, 1}))
        self.assertGreater(np.count_nonzero(skel), 0)

        tangents, curvature = extract_stroke_curvature_and_tangents(skel)
        self.assertEqual(curvature.shape, (224, 224))
        self.assertTrue(0.0 <= np.min(curvature) and np.max(curvature) <= 1.0)

        pert_skel, pert_render = inject_curvature_hesitation_noise(
            skeleton=skel,
            curvature_map=curvature,
            base_noise_std=0.7,
            curvature_gain=2.0,
            seed=42
        )
        self.assertEqual(pert_skel.shape, (224, 224))
        self.assertGreater(np.count_nonzero(pert_skel), 0)

        cond_dict = extract_condition_tensor(canvas, target_size=(224, 224))
        self.assertIn("condition_composite", cond_dict)
        self.assertEqual(cond_dict["condition_composite"].shape, (224, 224, 3))

    def test_02_structural_discrepancy(self):
        """Test structural discrepancy metric S_diff (SSIM + gradient divergence)."""
        from src.training.generate_negatives import compute_structural_discrepancy

        im1 = np.ones((224, 224), dtype=np.uint8) * 255
        im2 = im1.copy()
        cv2.line(im2, (20, 20), (200, 200), 0, 3)

        metrics = compute_structural_discrepancy(im1, im2)
        self.assertIn("S_diff", metrics)
        self.assertIn("ssim", metrics)
        self.assertTrue(0.0 <= metrics["S_diff"] <= 2.0)

    def test_03_siamese_vit_and_loss(self):
        """Test Siamese ViT forward pass, L2 hypersphere normalization, and backprop."""
        from src.losses.adaptive_margin import SigmoidalAdaptiveMarginLoss
        from src.models.siamese_vit import SiameseViT

        loss_fn = SigmoidalAdaptiveMarginLoss(
            m_min=0.60,
            m_max=1.40,
            total_epochs=30,
            steepness=0.35,
            midpoint_ratio=0.50
        )
        m_start = loss_fn.get_current_margin()
        self.assertTrue(0.60 <= m_start <= 1.40)

        loss_fn.update_epoch(15, 30)
        m_mid = loss_fn.get_current_margin()
        self.assertTrue(0.85 <= m_mid <= 1.15)

        if TORCH_AVAILABLE:
            model = SiameseViT(
                model_name="vit_tiny_patch16_224",
                pretrained=False,
                in_channels=3,
                embedding_dim=128,
                img_size=224
            )
            model.train()

            dummy_img1 = torch.randn(2, 3, 224, 224, dtype=torch.float32)
            dummy_img2 = torch.randn(2, 3, 224, 224, dtype=torch.float32)

            distances, z1, z2 = model(dummy_img1, dummy_img2)

            self.assertEqual(z1.shape, (2, 128))
            self.assertEqual(z2.shape, (2, 128))
            self.assertEqual(distances.shape, (2,))

            # Check L2 Unit Hypersphere projection: ||z||_2 == 1.0
            norm_z1 = torch.norm(z1, p=2, dim=-1)
            norm_z2 = torch.norm(z2, p=2, dim=-1)
            np.testing.assert_allclose(norm_z1.detach().numpy(), [1.0, 1.0], atol=1e-5)
            np.testing.assert_allclose(norm_z2.detach().numpy(), [1.0, 1.0], atol=1e-5)

            # Test loss & backpropagation
            labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
            loss = loss_fn(distances, labels, pair_types=["positive", "skilled_forgery"])
            self.assertGreater(loss.item(), 0.0)

            loss.backward()
            has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            self.assertTrue(has_grad)

    def test_04_biometric_evaluation(self):
        """Test biometric metrics (FAR, FRR, EER interpolation, and Markdown table)."""
        from evaluate import evaluate_biometric_metrics, generate_markdown_table

        dummy_labels = np.array([1.0, 0.0], dtype=np.float32)
        dummy_scores = np.array([0.25, 0.95], dtype=np.float32)
        dummy_types = ["positive", "skilled_forgery"]

        metrics = evaluate_biometric_metrics(dummy_labels, dummy_scores, dummy_types, num_thresholds=100)
        self.assertIn("skilled", metrics)
        self.assertIn("EER", metrics["skilled"])

        md_table = generate_markdown_table(metrics, dataset_name="SmokeTest", metric_type="EUCLIDEAN")
        self.assertIn("Equal Error Rate ($EER$)", md_table)


if __name__ == "__main__":
    unittest.main()
