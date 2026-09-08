"""
skeletonize.py - OpenCV Morphological Thinning & Curvature Extraction for Signatures
Extracts 1-pixel skeletons from raw grayscale signatures using OpenCV morphological operations
and computes local spatial tangents and curvature fields.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, Union

def extract_morphological_skeleton(
    grayscale_image: np.ndarray,
    binarization_method: str = "otsu",
    invert_input: bool = True
) -> np.ndarray:
    """
    Extracts a 1-pixel wide skeleton from a raw grayscale signature using
    OpenCV morphological Hit-or-Miss thinning and morphological erosion.

    Args:
        grayscale_image: Raw 2D grayscale signature image (H, W), dtype uint8.
        binarization_method: 'otsu' for Otsu thresholding, 'adaptive' for adaptive Gaussian thresholding.
        invert_input: If True, assumes dark ink on light paper and inverts so foreground=255, background=0.

    Returns:
        1-pixel wide binary skeleton (H, W) where stroke pixels = 1 and background = 0.
    """
    if len(grayscale_image.shape) == 3:
        gray = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = grayscale_image.copy()

    # 1. Binarize to isolate ink strokes
    if binarization_method == "otsu":
        flag = cv2.THRESH_BINARY_INV if invert_input else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    elif binarization_method == "adaptive":
        flag = cv2.THRESH_BINARY_INV if invert_input else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 15, 5)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV if invert_input else cv2.THRESH_BINARY)

    # 2. Check if cv2.ximgproc has thinning available
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        thinned = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return (thinned > 0).astype(np.uint8)

    # 3. OpenCV Hit-or-Miss Morphological Thinning kernels (8-connected structuring elements)
    # k1: horizontal/vertical directions, k2: diagonal directions
    k1 = np.array([[-1, -1, -1],
                   [ 0,  1,  0],
                   [ 1,  1,  1]], dtype=np.int32)
    k2 = np.array([[ 0, -1, -1],
                   [ 1,  1, -1],
                   [ 0,  1,  0]], dtype=np.int32)
    
    # Generate 8 rotated structuring elements (45 degree increments)
    kernels = [k1, k2]
    for _ in range(3):
        kernels.append(np.rot90(kernels[-2]))
        kernels.append(np.rot90(kernels[-2]))

    # Iterate morphological thinning until convergence
    thinned = binary.copy()
    prev = np.zeros_like(thinned)
    max_iters = 100
    iteration = 0

    while iteration < max_iters:
        for k in kernels:
            # OpenCV Hit-or-Miss morphological transform detects boundary pixels
            hitmiss = cv2.morphologyEx(thinned, cv2.MORPH_HITMISS, k)
            thinned = cv2.subtract(thinned, hitmiss)
        
        # Check convergence
        diff = cv2.countNonZero(cv2.absdiff(thinned, prev))
        if diff == 0:
            break
        prev = thinned.copy()
        iteration += 1

    return (thinned > 0).astype(np.uint8)

def extract_stroke_curvature_and_tangents(
    skeleton: np.ndarray,
    smoothing_ksize: int = 5,
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes local tangent orientation angles and curvature magnitude along the 1-pixel skeleton.

    Args:
        skeleton: (H, W) binary array of stroke centerlines (1=stroke, 0=background).
        smoothing_ksize: Kernel size for spatial derivative smoothing.
        eps: Small constant to avoid zero division.

    Returns:
        tangent_map: (H, W) array of tangent orientation angles in [-pi, pi], masked to skeleton pixels.
        curvature_map: (H, W) array of normalized curvature magnitudes in [0, 1].
    """
    skel_float = skeleton.astype(np.float32)
    # Slight Gaussian blur to compute continuous 1st and 2nd derivatives
    skel_blurred = cv2.GaussianBlur(skel_float, (smoothing_ksize, smoothing_ksize), sigmaX=1.0)

    # 1st order partial spatial derivatives
    dx = cv2.Sobel(skel_blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(skel_blurred, cv2.CV_32F, 0, 1, ksize=3)

    # 2nd order partial spatial derivatives
    dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
    dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)

    # Tangent angle: theta(s) = arctan2(dy, dx)
    tangent = np.arctan2(dy, dx)

    # Curvature formula: kappa = |dx * dyy - dy * dxx| / (dx^2 + dy^2 + eps)^(3/2)
    numerator = np.abs(dx * dyy - dy * dxx)
    denominator = np.power(dx**2 + dy**2 + eps, 1.5)
    curvature = numerator / denominator

    # Mask to only skeleton pixels
    tangent_map = np.where(skeleton > 0, tangent, 0.0)
    curvature_map = np.where(skeleton > 0, curvature, 0.0)

    # Min-Max normalize curvature across the signature
    skel_mask = (skeleton > 0)
    if np.any(skel_mask):
        c_vals = curvature_map[skel_mask]
        c_min, c_max = c_vals.min(), c_vals.max()
        if c_max > c_min:
            curvature_map[skel_mask] = (curvature_map[skel_mask] - c_min) / (c_max - c_min)

    return tangent_map, curvature_map

def extract_condition_tensor(
    image: Union[str, np.ndarray],
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, np.ndarray]:
    """
    End-to-end preprocessing: accepts raw grayscale signature (path or array),
    performs morphological thinning, curvature extraction, and packages condition tensors.

    Returns:
        dict with:
            'image': (H, W, 1) normalized inverted signature image in [0, 1]
            'skeleton': (H, W, 1) 1-pixel skeleton mask in {0, 1}
            'curvature': (H, W, 1) normalized curvature field in [0, 1]
            'condition_composite': (H, W, 3) composite [skeleton, curvature, normalized_tangent]
    """
    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Signature image not found at: {image}")
    else:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

    # Resize raw grayscale signature
    h, w = target_size
    gray_resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    # Morphological thinning
    skeleton = extract_morphological_skeleton(gray_resized, binarization_method="otsu", invert_input=True)

    # Curvature & Tangent extraction
    tangents, curvature = extract_stroke_curvature_and_tangents(skeleton)

    # Normalized inverted image (ink=1, paper=0)
    _, binary = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    norm_img = binary.astype(np.float32) / 255.0

    # Composite condition for ControlNet:
    # Ch 0: Skeleton (binary mask)
    # Ch 1: Curvature intensity [0, 1]
    # Ch 2: Tangent angle normalized to [0, 1]
    norm_tangent = np.where(skeleton > 0, (tangents + np.pi) / (2.0 * np.pi), 0.0)
    composite = np.stack([
        skeleton.astype(np.float32),
        curvature.astype(np.float32),
        norm_tangent.astype(np.float32)
    ], axis=-1)

    return {
        'image': np.expand_dims(norm_img, axis=-1),
        'skeleton': np.expand_dims(skeleton.astype(np.float32), axis=-1),
        'curvature': np.expand_dims(curvature.astype(np.float32), axis=-1),
        'condition_composite': composite
    }

# Alias for standard function call
skeletonize = extract_morphological_skeleton

if __name__ == "__main__":
    import os
    sample_path = r"F:\artDL\signature_diffusion_osv\data\CEDAR\full_org\original_1_1.png"
    if os.path.exists(sample_path):
        raw_gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        skel = extract_morphological_skeleton(raw_gray)
        print(f"[skeletonize.py] Input: {raw_gray.shape} -> 1-pixel skeleton: {np.count_nonzero(skel)} pixels")
        tangents, curvature = extract_stroke_curvature_and_tangents(skel)
        print(f"[skeletonize.py] Curvature range: [{curvature.min():.4f}, {curvature.max():.4f}]")
    else:
        print(f"[skeletonize.py] Sample image not found at {sample_path}")
