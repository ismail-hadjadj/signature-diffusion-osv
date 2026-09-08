"""
icdar_loader.py - Writer-Independent (WI) Pair Generator & Dataset for ICDAR 2011
Supports standard Writer-Independent SigComp protocols across Dutch and Chinese scripts.
Generates positive (genuine-genuine), random negative (genuine across writers),
and skilled forgery pairs.
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
from src.dataset.cedar_loader import load_and_preprocess_signature

class ICDAR2011PairDataset(TorchDataset):
    """
    ICDAR 2011 (SigComp2011) Offline Signature Dataset supporting Writer-Independent (WI) evaluation.

    Standard SigComp Protocol:
      - Strictly disjoint writer partitions between Training and Testing.
      - Dutch Script & Chinese Script partitions.
      - Default: 10 training writers, 54 evaluation test writers (or script-specific partitions).

    Pair Generation:
      - Positive: Genuine (Writer W) + Genuine (Writer W) [Label = 1]
      - Skilled Negative: Genuine (Writer W) + Skilled Forgery (Writer W) [Label = 0]
      - Random Negative: Genuine (Writer A) + Genuine (Writer B) [Label = 0]
    """

    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        subset: str = "Dutch", # 'Dutch' or 'Chinese'
        script: Optional[str] = None, # backward compatibility
        target_size: Tuple[int, int] = (224, 224),
        num_train_writers: int = 10,
        positive_pairs_per_writer: int = 50,
        num_negatives_per_positive: float = 1.0,
        skilled_neg_ratio: float = 0.5,
        seed: int = 42,
        extract_conditions: bool = True,
        transform=None
    ):
        self.data_dir = data_dir
        self.is_train = is_train
        self.subset = script if script is not None else subset
        self.script = self.subset
        self.target_size = target_size
        self.num_train_writers = num_train_writers
        self.pos_per_writer = positive_pairs_per_writer
        self.num_negatives_per_positive = num_negatives_per_positive
        self.skilled_neg_ratio = skilled_neg_ratio
        self.seed = seed
        self.extract_conditions = extract_conditions
        self.transform = transform

        # 1. Parse offline signature images for the official partitions
        self.writers_data, self.train_writers, self.test_writers = self._parse_icdar_directory()
        self.all_writers = self.train_writers if self.is_train else self.test_writers
        self.active_writers = self.all_writers

        # 2. Generate balanced pairs
        self.pairs = self._generate_wi_pairs()

    def _parse_icdar_directory(self) -> Tuple[Dict[str, Dict[str, List[str]]], List[str], List[str]]:
        """
        Parses official ICDAR 2011 SigComp directory structure dynamically for Dutch or Chinese:
        - Train partition (10 reference writers):
            data_dir/train/OfflineSignatures/<subset>/TrainingSet/Offline Genuine
            data_dir/train/OfflineSignatures/<subset>/TrainingSet/Offline Forgeries
        - Test partition (held-out competition writers: 54 for Dutch, 10 for Chinese):
            data_dir/test/OfflineSignatures/<subset>/Reference*/<writer>/
            data_dir/test/OfflineSignatures/<subset>/Questioned*/<writer>/
        """
        train_writers: Dict[str, Dict[str, List[str]]] = {}
        test_writers: Dict[str, Dict[str, List[str]]] = {}

        sub = "Chinese" if self.subset.lower() == "chinese" else "Dutch"

        # 1. Parse TrainingSet (<subset>)
        train_base = os.path.join(self.data_dir, "train", "OfflineSignatures", sub, "TrainingSet")
        if not os.path.exists(train_base):
            train_base = os.path.join(self.data_dir, sub, "TrainingSet")
        if not os.path.exists(train_base):
            train_base = os.path.join(self.data_dir, "TrainingSet")

        if os.path.exists(train_base):
            gen_dir = os.path.join(train_base, "Offline Genuine")
            forg_dir = os.path.join(train_base, "Offline Forgeries")

            if os.path.exists(gen_dir):
                for p in glob.glob(os.path.join(gen_dir, "*.png")) + glob.glob(os.path.join(gen_dir, "*.PNG")):
                    fname = os.path.basename(p)
                    w = fname.split('_')[0]
                    if w not in train_writers:
                        train_writers[w] = {'genuine': [], 'forgery': []}
                    train_writers[w]['genuine'].append(p)

            if os.path.exists(forg_dir):
                tw_list = sorted(list(train_writers.keys()), key=len, reverse=True)
                for p in glob.glob(os.path.join(forg_dir, "*.png")) + glob.glob(os.path.join(forg_dir, "*.PNG")):
                    fname = os.path.basename(p)
                    prefix = fname.split('_')[0]
                    target_w = None
                    for w in tw_list:
                        if prefix.endswith(w) and len(prefix) > len(w):
                            target_w = w
                            break
                    if target_w and target_w in train_writers:
                        train_writers[target_w]['forgery'].append(p)

        # 2. Parse Test Set (<subset> Reference + Questioned)
        test_base = os.path.join(self.data_dir, "test", "OfflineSignatures", sub)
        if not os.path.exists(test_base):
            test_base = os.path.join(self.data_dir, sub)
        if not os.path.exists(test_base):
            test_base = self.data_dir

        ref_dir = None
        quest_dir = None
        if os.path.exists(test_base):
            for entry in os.listdir(test_base):
                entry_path = os.path.join(test_base, entry)
                if os.path.isdir(entry_path):
                    if entry.lower().startswith("ref"):
                        ref_dir = entry_path
                    elif entry.lower().startswith("quest"):
                        quest_dir = entry_path

        if ref_dir and os.path.exists(ref_dir):
            for w in os.listdir(ref_dir):
                w_dir = os.path.join(ref_dir, w)
                if os.path.isdir(w_dir):
                    if w not in test_writers:
                        test_writers[w] = {'genuine': [], 'forgery': []}
                    for p in glob.glob(os.path.join(w_dir, "*.png")) + glob.glob(os.path.join(w_dir, "*.PNG")):
                        test_writers[w]['genuine'].append(p)

        if quest_dir and os.path.exists(quest_dir):
            for w in os.listdir(quest_dir):
                w_dir = os.path.join(quest_dir, w)
                if os.path.isdir(w_dir):
                    if w not in test_writers:
                        test_writers[w] = {'genuine': [], 'forgery': []}
                    for p in glob.glob(os.path.join(w_dir, "*.png")) + glob.glob(os.path.join(w_dir, "*.PNG")):
                        fname = os.path.basename(p)
                        base = os.path.splitext(fname)[0]
                        parts = base.split('_')
                        if len(parts) == 2:
                            _, id_part = parts
                            if id_part == w:
                                test_writers[w]['genuine'].append(p)
                            else:
                                test_writers[w]['forgery'].append(p)
                        else:
                            test_writers[w]['forgery'].append(p)

        # If both parsed successfully, use strict SigComp train/test partitions
        if len(train_writers) > 0 and len(test_writers) > 0:
            active_dict = train_writers if self.is_train else test_writers
            return active_dict, sorted(list(train_writers.keys())), sorted(list(test_writers.keys()))

        # Fallback recursive parser
        fallback_writers: Dict[str, Dict[str, List[str]]] = {}
        files = glob.glob(os.path.join(self.data_dir, "**", "*.png"), recursive=True)
        files += glob.glob(os.path.join(self.data_dir, "**", "*.PNG"), recursive=True)

        for p in files:
            norm_p = p.replace('\\', '/')
            if self.script and self.script.lower() not in norm_p.lower():
                continue
            fname = os.path.basename(p)
            is_genuine = ('genuine' in norm_p.lower()) or ('reference' in norm_p.lower())
            is_forgery = ('forg' in norm_p.lower()) or ('questioned' in norm_p.lower())
            parent = os.path.basename(os.path.dirname(p))
            writer = None
            if parent.isdigit():
                writer = parent
            else:
                m = re.search(r"(\d{3})_\d+", fname)
                if m:
                    writer = m.group(1)
                else:
                    m2 = re.search(r"\d{4}(\d{3})_\d+", fname)
                    if m2:
                        writer = m2.group(1)

            if writer:
                if writer not in fallback_writers:
                    fallback_writers[writer] = {'genuine': [], 'forgery': []}
                if is_genuine:
                    fallback_writers[writer]['genuine'].append(p)
                elif is_forgery:
                    fallback_writers[writer]['forgery'].append(p)
                else:
                    if 'org' in fname.lower():
                        fallback_writers[writer]['genuine'].append(p)
                    else:
                        fallback_writers[writer]['forgery'].append(p)

        valid_writers = {w: data for w, data in fallback_writers.items() if len(data['genuine']) > 0}
        all_w = sorted(list(valid_writers.keys()))
        rng = random.Random(self.seed)
        shuffled = all_w.copy()
        rng.shuffle(shuffled)
        n_train = min(self.num_train_writers, len(shuffled) // 2)
        tr = shuffled[:n_train]
        te = shuffled[n_train:]
        return valid_writers, tr, te

    def _get_wi_split(self) -> Tuple[List[str], List[str]]:
        """Returns disjoint train and test writers."""
        return self.train_writers, self.test_writers

    def _generate_wi_pairs(self) -> List[Tuple[str, str, int, str]]:
        """Generates balanced Writer-Independent pairs for active writers with uniform random negative sampling."""
        rng = random.Random(self.seed + (0 if self.is_train else 200))
        positive_pairs = []
        skilled_negatives = []
        random_negatives = []

        for w in self.active_writers:
            genuines = sorted(self.writers_data[w]['genuine'])
            forgeries = sorted(self.writers_data[w]['forgery'])

            # Positives: Genuine combinations
            all_pos = []
            for i in range(len(genuines)):
                for j in range(i + 1, len(genuines)):
                    all_pos.append((genuines[i], genuines[j], 1, 'positive'))

            if self.pos_per_writer and len(all_pos) > self.pos_per_writer:
                rng.shuffle(all_pos)
                positive_pairs.extend(all_pos[:self.pos_per_writer])
            else:
                positive_pairs.extend(all_pos)

            # Skilled forgeries
            for g in genuines:
                for f in forgeries:
                    skilled_negatives.append((g, f, 0, 'skilled_forgery'))

        # Balance negatives against positives
        total_pos = len(positive_pairs)
        target_total_negs = int(total_pos * self.num_negatives_per_positive)
        target_skilled = int(target_total_negs * self.skilled_neg_ratio)
        target_random = target_total_negs - target_skilled

        # Uniform Random Negatives across all distinct active writers
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
        """Returns pair distribution statistics."""
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

        if self.extract_conditions:
            cond1 = extract_condition_tensor(p1, target_size=self.target_size)
            cond2 = extract_condition_tensor(p2, target_size=self.target_size)

            t1 = np.transpose(cond1['image'], (2, 0, 1))
            t2 = np.transpose(cond2['image'], (2, 0, 1))
            c1 = np.transpose(cond1['condition_composite'], (2, 0, 1))
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

def get_icdar_wi_loaders(
    data_dir: str,
    script: Optional[str] = None,
    subset: Optional[str] = None,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    extract_conditions: bool = False
):
    """Factory helper to obtain train and test PyTorch DataLoaders for ICDAR 2011 SigComp WI split."""
    selected_subset = subset if subset is not None else (script if script is not None else "Dutch")
    train_ds = ICDAR2011PairDataset(data_dir=data_dir, is_train=True, subset=selected_subset, target_size=target_size, extract_conditions=extract_conditions)
    test_ds = ICDAR2011PairDataset(data_dir=data_dir, is_train=False, subset=selected_subset, target_size=target_size, extract_conditions=extract_conditions)
    
    if torch is not None:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    return train_ds, test_ds

if __name__ == "__main__":
    import os
    data_dir = r"F:\artDL\signature_diffusion_osv\data\ICDAR2011"
    if os.path.exists(data_dir):
        ds_train = ICDAR2011PairDataset(data_dir, is_train=True, positive_pairs_per_writer=20, extract_conditions=False)
        ds_test = ICDAR2011PairDataset(data_dir, is_train=False, positive_pairs_per_writer=20, extract_conditions=False)
        print(f"[icdar_loader.py] Train stats: {ds_train.get_stats()}")
        print(f"[icdar_loader.py] Test stats:  {ds_test.get_stats()}")
    else:
        print(f"[icdar_loader.py] Directory not found: {data_dir}")
