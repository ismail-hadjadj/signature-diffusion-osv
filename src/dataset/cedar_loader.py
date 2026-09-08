"""
cedar_loader.py - Writer-Independent (WI) Pair Generator & Dataset for CEDAR
Supports standard Writer-Independent 5-fold cross-validation and generates
positive (genuine-genuine), random negative (genuine across writers), and skilled forgery pairs.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import glob
import re
import random
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import cv2

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    TorchDataset = object
    torch = None

from src.preprocessing.skeletonize import extract_condition_tensor

_SIGNATURE_CACHE: Dict[Tuple[str, Tuple[int, int]], np.ndarray] = {}

def load_and_preprocess_signature(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Loads raw signature image, resizes to target_size, inverts background
    (ink = 1.0, background = 0.0), expands to 3 channels (3, H, W),
    and applies standard ImageNet normalization:
      mean = [0.485, 0.456, 0.406]
      std  = [0.229, 0.224, 0.225]
    
    Returns:
        np.ndarray of shape (3, H, W) and dtype np.float32.
    """
    cache_key = (os.path.normpath(image_path), target_size)
    if cache_key in _SIGNATURE_CACHE:
        return _SIGNATURE_CACHE[cache_key].copy()

    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        im = np.zeros(target_size, dtype=np.uint8)
    im = cv2.resize(im, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    # Invert grayscale: black background (0.0) and white signature ink (1.0)
    im_norm = 1.0 - (im.astype(np.float32) / 255.0)

    # Expand to 3 channels: (3, H, W)
    t = np.stack([im_norm, im_norm, im_norm], axis=0)

    # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    t = (t - mean) / std

    res = t.astype(np.float32)
    _SIGNATURE_CACHE[cache_key] = res
    return res.copy()

class CEDARPairDataset(TorchDataset):
    """
    CEDAR Offline Signature Dataset supporting Writer-Independent (WI) Pair Generation.
    
    Standard Protocol:
      - 55 distinct writers (Writers 1 to 55).
      - 5-Fold Cross Validation: 44 Train writers, 11 Test writers per fold.
      - Disjoint writer sets: No writer in Test is seen during Train (strict WI).
      
    Pair Generation Types:
      - Positive: Genuine signature (Writer W) + Genuine signature (Writer W) [Label = 1]
      - Skilled Negative: Genuine signature (Writer W) + Skilled Forgery (Writer W) [Label = 0]
      - Random Negative: Genuine signature (Writer A) + Genuine signature (Writer B) [Label = 0]
    """

    def __init__(
        self,
        data_dir: str,
        fold: int = 0,
        is_train: bool = True,
        num_folds: int = 5,
        target_size: Tuple[int, int] = (224, 224),
        positive_pairs_per_writer: int = 100, # Combinations out of 276
        num_negatives_per_positive: int = 1.0,
        skilled_neg_ratio: float = 0.5, # 50% skilled, 50% random negatives
        seed: int = 42,
        extract_conditions: bool = True,
        transform=None
    ):
        self.data_dir = data_dir
        self.fold = fold
        self.is_train = is_train
        self.num_folds = num_folds
        self.target_size = target_size
        self.pos_per_writer = positive_pairs_per_writer
        self.num_negatives_per_positive = num_negatives_per_positive
        self.skilled_neg_ratio = skilled_neg_ratio
        self.seed = seed
        self.extract_conditions = extract_conditions
        self.transform = transform

        # 1. Parse directory structure
        self.writers_data = self._parse_cedar_directory()
        self.all_writers = sorted(list(self.writers_data.keys()), key=lambda x: int(x))

        # 2. Strict Writer-Independent (WI) Split
        self.train_writers, self.test_writers = self._get_wi_split()
        self.active_writers = self.train_writers if self.is_train else self.test_writers

        # 3. Generate balanced pair tuples: (img1_path, img2_path, label, pair_type)
        self.pairs = self._generate_wi_pairs()

    def _parse_cedar_directory(self) -> Dict[str, Dict[str, List[str]]]:
        """Scans CEDAR data directory for genuine and forged signatures."""
        writers: Dict[str, Dict[str, List[str]]] = {}
        
        # Check standard folder variants
        org_dir = os.path.join(self.data_dir, "full_org")
        forg_dir = os.path.join(self.data_dir, "full_forg")

        if not os.path.exists(org_dir):
            org_dir = os.path.join(self.data_dir, "signatures", "full_org")
            forg_dir = os.path.join(self.data_dir, "signatures", "full_forg")

        org_files = glob.glob(os.path.join(org_dir, "*.png"))
        for p in org_files:
            fname = os.path.basename(p)
            m = re.search(r"original_(\d+)_", fname)
            if m:
                w = m.group(1)
                if w not in writers:
                    writers[w] = {'genuine': [], 'forgery': []}
                writers[w]['genuine'].append(p)

        forg_files = glob.glob(os.path.join(forg_dir, "*.png"))
        for p in forg_files:
            fname = os.path.basename(p)
            m = re.search(r"forgeries_(\d+)_", fname)
            if m:
                w = m.group(1)
                if w not in writers:
                    writers[w] = {'genuine': [], 'forgery': []}
                writers[w]['forgery'].append(p)

        return writers

    def _get_wi_split(self) -> Tuple[List[str], List[str]]:
        """Splits writers strictly disjointly into 5 folds."""
        rng = random.Random(self.seed)
        shuffled = self.all_writers.copy()
        rng.shuffle(shuffled)

        fold_size = len(shuffled) // self.num_folds
        test_start = self.fold * fold_size
        test_end = test_start + fold_size if self.fold < self.num_folds - 1 else len(shuffled)

        test_writers = shuffled[test_start:test_end]
        train_writers = [w for w in shuffled if w not in test_writers]
        return train_writers, test_writers

    def _generate_wi_pairs(self) -> List[Tuple[str, str, int, str]]:
        """
        Constructs Writer-Independent pairs for the active writers.
        Ensures disjointness and balanced representation.
        """
        rng = random.Random(self.seed + self.fold + (0 if self.is_train else 100))
        positive_pairs = []
        skilled_negatives = []
        random_negatives = []

        # 1. Positive Pairs & Skilled Forgeries
        for w in self.active_writers:
            genuines = sorted(self.writers_data[w]['genuine'])
            forgeries = sorted(self.writers_data[w]['forgery'])

            # Genuine combinations: n*(n-1)/2 = 276
            all_pos = []
            for i in range(len(genuines)):
                for j in range(i + 1, len(genuines)):
                    all_pos.append((genuines[i], genuines[j], 1, 'positive'))
            
            # Subsample positives per writer if specified
            if self.pos_per_writer and len(all_pos) > self.pos_per_writer:
                rng.shuffle(all_pos)
                positive_pairs.extend(all_pos[:self.pos_per_writer])
            else:
                positive_pairs.extend(all_pos)

            # Skilled forgeries: genuine vs forgery of the same writer
            for g in genuines:
                for f in forgeries:
                    skilled_negatives.append((g, f, 0, 'skilled_forgery'))

        # 3. Balance Negatives against Positives
        total_positives = len(positive_pairs)
        target_total_negatives = int(total_positives * self.num_negatives_per_positive)
        target_skilled = int(target_total_negatives * self.skilled_neg_ratio)
        target_random = target_total_negatives - target_skilled

        # 2. Random Negatives: uniform selection across all distinct other active writers
        if len(self.active_writers) > 1:
            num_per_writer = max(1, target_random // len(self.active_writers))
            for w in self.active_writers:
                genuines_w = self.writers_data[w]['genuine']
                other_writers = [ow for ow in self.active_writers if ow != w]
                rng.shuffle(other_writers)
                for idx in range(num_per_writer):
                    other_w = other_writers[idx % len(other_writers)]
                    g1 = rng.choice(genuines_w)
                    g2 = rng.choice(self.writers_data[other_w]['genuine'])
                    random_negatives.append((g1, g2, 0, 'random_negative'))

        rng.shuffle(skilled_negatives)
        rng.shuffle(random_negatives)

        selected_skilled = skilled_negatives[:target_skilled]
        selected_random = random_negatives[:target_random]

        all_pairs = positive_pairs + selected_skilled + selected_random
        rng.shuffle(all_pairs)
        return all_pairs

    def get_stats(self) -> Dict[str, int]:
        """Returns distribution statistics of the active dataset split."""
        pos_cnt = sum(1 for p in self.pairs if p[2] == 1)
        skilled_cnt = sum(1 for p in self.pairs if p[3] == 'skilled_forgery')
        random_cnt = sum(1 for p in self.pairs if p[3] == 'random_negative')
        return {
            'writers_count': len(self.active_writers),
            'total_pairs': len(self.pairs),
            'positive_pairs': pos_cnt,
            'skilled_forgery_pairs': skilled_cnt,
            'random_negative_pairs': random_cnt
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p1, p2, label, pair_type = self.pairs[idx]

        # Load and preprocess using morphological thinning
        if self.extract_conditions:
            cond1 = extract_condition_tensor(p1, target_size=self.target_size)
            cond2 = extract_condition_tensor(p2, target_size=self.target_size)

            t1 = np.transpose(cond1['image'], (2, 0, 1)) # (1, H, W)
            t2 = np.transpose(cond2['image'], (2, 0, 1))
            # Expand to 3 channels if needed
            if t1.shape[0] == 1:
                t1 = np.repeat(t1, 3, axis=0)
                t2 = np.repeat(t2, 3, axis=0)
            c1 = np.transpose(cond1['condition_composite'], (2, 0, 1)) # (3, H, W)
            c2 = np.transpose(cond2['condition_composite'], (2, 0, 1))
        else:
            t1 = load_and_preprocess_signature(p1, target_size=self.target_size)
            t2 = load_and_preprocess_signature(p2, target_size=self.target_size)
            c1, c2 = np.zeros_like(t1), np.zeros_like(t2)

        if torch is not None:
            return {
                'img1': torch.from_numpy(t1).float(),
                'img2': torch.from_numpy(t2).float(),
                'cond1': torch.from_numpy(c1).float(),
                'cond2': torch.from_numpy(c2).float(),
                'label': torch.tensor(label, dtype=torch.float32),
                'pair_type': pair_type
            }

        return {
            'img1': t1,
            'img2': t2,
            'cond1': c1,
            'cond2': c2,
            'label': float(label),
            'pair_type': pair_type
        }

def get_cedar_wi_loaders(
    data_dir: str,
    fold: int = 0,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    extract_conditions: bool = False
):
    """Factory helper to obtain train and test PyTorch DataLoaders for CEDAR WI split."""
    train_ds = CEDARPairDataset(data_dir=data_dir, fold=fold, is_train=True, target_size=target_size, extract_conditions=extract_conditions)
    test_ds = CEDARPairDataset(data_dir=data_dir, fold=fold, is_train=False, target_size=target_size, extract_conditions=extract_conditions)
    
    if torch is not None:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    return train_ds, test_ds

if __name__ == "__main__":
    import os
    data_dir = r"F:\artDL\signature_diffusion_osv\data\CEDAR"
    if os.path.exists(data_dir):
        ds_train = CEDARPairDataset(data_dir, fold=0, is_train=True, positive_pairs_per_writer=20, extract_conditions=False)
        ds_test = CEDARPairDataset(data_dir, fold=0, is_train=False, positive_pairs_per_writer=20, extract_conditions=False)
        print(f"[cedar_loader.py] Train stats: {ds_train.get_stats()}")
        print(f"[cedar_loader.py] Test stats:  {ds_test.get_stats()}")
    else:
        print(f"[cedar_loader.py] Directory not found: {data_dir}")
