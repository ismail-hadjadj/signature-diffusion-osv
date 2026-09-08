"""
kinematic_tremor.py - Curvature-Dependent Gaussian Hesitation Noise Injection
Injects synthetic neuromuscular perturbation into stroke coordinate vectors,
where hesitation and motor tremor scale proportionally to local stroke curvature.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
from src.preprocessing.skeletonize import extract_stroke_curvature_and_tangents

def extract_stroke_paths(skeleton: np.ndarray) -> List[np.ndarray]:
    """
    Extracts ordered 2D stroke coordinate vectors (paths) from a 1-pixel skeleton.
    Returns list of paths of shape (N_i, 2) with (x, y) coordinates.
    """
    skel_bin = (skeleton > 0).astype(np.uint8)
    contours, _ = cv2.findContours(skel_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    paths = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 4:
            paths.append(pts)
    return paths

def inject_curvature_hesitation_noise(
    skeleton: np.ndarray,
    curvature_map: Optional[np.ndarray] = None,
    base_noise_std: float = 0.6,
    curvature_gain: float = 2.5,
    curvature_power: float = 1.5,
    hesitation_dwell_prob: float = 0.20,
    dwell_radius: int = 3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic perturbation function that injects curvature-dependent Gaussian
    hesitation noise into the stroke vectors.

    Neuromotor justification:
    When an imitator executes a skilled forgery, handwriting speed is significantly
    reduced at high-curvature regions (loops, corners, directional changes). This
    loss of ballistic momentum causes involuntary neuromuscular motor tremor (8-14 Hz)
    and ink hesitation pauses that scale with the curvature kappa(s).

    Args:
        skeleton: (H, W) binary array containing 1-pixel signature skeleton (1=stroke, 0=bg).
        curvature_map: Optional (H, W) normalized curvature field in [0, 1]. If None, computed.
        base_noise_std: Baseline standard deviation of Gaussian noise in low-curvature straightaways.
        curvature_gain: Multiplier for hesitation noise at high curvature.
        curvature_power: Non-linear exponent alpha for curvature sensitivity: (1 + gain * kappa^alpha).
        hesitation_dwell_prob: Probability of ink dwell / pooling at high-curvature hesitation stops.
        dwell_radius: Max radius of ink pooling artifact at hesitation points.
        seed: Optional random seed for reproducible perturbations.

    Returns:
        perturbed_skeleton: (H, W) binary skeleton with curvature-dependent stroke vector noise.
        perturbed_ink_render: (H, W) grayscale rendering simulating ink line width and hesitation pauses.
    """
    h, w = skeleton.shape[:2]
    rng = np.random.RandomState(seed)

    if curvature_map is None:
        _, curvature_map = extract_stroke_curvature_and_tangents(skeleton)

    # Extract continuous stroke paths
    paths = extract_stroke_paths(skeleton)
    if not paths:
        return skeleton.copy(), (skeleton.copy() * 255).astype(np.uint8)

    perturbed_skel = np.zeros((h, w), dtype=np.uint8)
    perturbed_render = np.zeros((h, w), dtype=np.uint8)

    for path in paths:
        n_pts = len(path)
        # Sample curvature along path coordinates
        x_indices = np.clip(np.round(path[:, 0]), 0, w - 1).astype(int)
        y_indices = np.clip(np.round(path[:, 1]), 0, h - 1).astype(int)
        k_vals = curvature_map[y_indices, x_indices] # Curvature values in [0, 1]

        # Tangent vectors: t = dr/ds
        tangents = np.gradient(path, axis=0)
        norm_t = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-6
        unit_tangents = tangents / norm_t
        # Normal vectors orthogonal to tangent: (-t_y, t_x)
        unit_normals = np.stack([-unit_tangents[:, 1], unit_tangents[:, 0]], axis=1)

        # Curvature-dependent hesitation standard deviation:
        # sigma(s) = base_std * (1.0 + curvature_gain * kappa(s)^curvature_power)
        sigma_hesitation = base_noise_std * (1.0 + curvature_gain * np.power(k_vals, curvature_power))

        # 2D Gaussian hesitation noise components:
        # Tangential noise simulates velocity stuttering / hesitation
        # Normal noise simulates high-frequency neuromotor hand tremor
        eta_normal = rng.normal(loc=0.0, scale=1.0, size=n_pts)
        eta_tangent = rng.normal(loc=0.0, scale=0.5, size=n_pts)

        delta_normal = unit_normals * (eta_normal * sigma_hesitation)[:, np.newaxis]
        delta_tangent = unit_tangents * (eta_tangent * sigma_hesitation)[:, np.newaxis]
        delta_stroke = delta_normal + delta_tangent

        # Perturbed stroke coordinates
        perturbed_path = path + delta_stroke
        perturbed_path[:, 0] = np.clip(perturbed_path[:, 0], 0, w - 1)
        perturbed_path[:, 1] = np.clip(perturbed_path[:, 1], 0, h - 1)
        int_pts = np.round(perturbed_path).astype(np.int32)

        # Draw perturbed skeleton line segments
        for i in range(len(int_pts) - 1):
            pt1 = tuple(int_pts[i])
            pt2 = tuple(int_pts[i + 1])
            cv2.line(perturbed_skel, pt1, pt2, 1, thickness=1)

            # Render ink stroke with variable thickness simulating speed variations:
            # High curvature / high hesitation -> thicker stroke (slower speed)
            thickness = 2 if k_vals[i] > 0.4 else 1
            cv2.line(perturbed_render, pt1, pt2, 255, thickness=thickness)

            # Inject hesitation dwell blots at extreme curvature turns
            if k_vals[i] > 0.6 and rng.uniform() < hesitation_dwell_prob:
                r = rng.randint(2, dwell_radius + 1)
                cv2.circle(perturbed_render, pt1, r, 255, thickness=-1)

    return perturbed_skel, perturbed_render


class KinematicHesitationPerturber:
    """
    Object-oriented wrapper for injecting curvature-dependent hesitation noise into signatures.
    """
    def __init__(
        self,
        base_noise_std: float = 0.6,
        curvature_gain: float = 2.5,
        curvature_power: float = 1.5,
        hesitation_dwell_prob: float = 0.20,
        dwell_radius: int = 3,
        seed: Optional[int] = None
    ):
        self.base_noise_std = base_noise_std
        self.curvature_gain = curvature_gain
        self.curvature_power = curvature_power
        self.hesitation_dwell_prob = hesitation_dwell_prob
        self.dwell_radius = dwell_radius
        self.seed = seed

    def __call__(
        self,
        skeleton: np.ndarray,
        curvature_map: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return inject_curvature_hesitation_noise(
            skeleton=skeleton,
            curvature_map=curvature_map,
            base_noise_std=self.base_noise_std,
            curvature_gain=self.curvature_gain,
            curvature_power=self.curvature_power,
            hesitation_dwell_prob=self.hesitation_dwell_prob,
            dwell_radius=self.dwell_radius,
            seed=self.seed
        )

if __name__ == "__main__":
    import os
    sample_path = r"F:\artDL\signature_diffusion_osv\data\CEDAR\full_org\original_1_1.png"
    if os.path.exists(sample_path):
        from src.preprocessing.skeletonize import extract_morphological_skeleton
        raw_gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        skel = extract_morphological_skeleton(raw_gray)
        pert_skel, pert_render = inject_curvature_hesitation_noise(skel, base_noise_std=0.8, curvature_gain=2.5)
        print(f"[kinematic_tremor.py] Original skeleton pixels: {np.count_nonzero(skel)}")
        print(f"[kinematic_tremor.py] Perturbed stroke pixels: {np.count_nonzero(pert_skel)}")
        print(f"[kinematic_tremor.py] Rendered hesitation pixels: {np.count_nonzero(pert_render)}")
    else:
        print(f"[kinematic_tremor.py] Sample image not found at {sample_path}")
