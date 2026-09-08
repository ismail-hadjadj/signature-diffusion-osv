"""
train_verifier.py - Phase 2: Curriculum Metric Learning for Siamese ViT
Ingests random negatives first, introduces skilled forgeries, and injects synthetic hard negatives
ranked by structural discrepancy (S_diff) with a sigmoidal expanding margin loss.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import json
import glob
import yaml
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False

from src.dataset.cedar_loader import CEDARPairDataset, load_and_preprocess_signature
from src.dataset.icdar_loader import ICDAR2011PairDataset
from src.models.siamese_vit import SiameseViT
from src.losses.adaptive_margin import SigmoidalAdaptiveMarginLoss

class CurriculumPairDataset(Dataset):
    """
    Curriculum Dataset wrapper that progressively introduces:
      1. Genuine-Genuine & Random Negatives (Stage 1: early epochs)
      2. Skilled Human Forgeries (Stage 2: intermediate epochs)
      3. Synthetic Hard Negatives ranked by S_diff (Stage 3: advanced epochs)
    """

    def __init__(
        self,
        base_dataset,
        synthetic_manifest_path: Optional[str] = "data/synthetic_hard_negatives/synthetic_negatives_manifest.json",
        target_size: tuple = (224, 224)
    ):
        self.base_dataset = base_dataset
        self.target_size = target_size
        self.stage = 1
        self.synthetic_pairs = []

        # Load synthetic hard negatives ranked by S_diff
        if synthetic_manifest_path and os.path.exists(synthetic_manifest_path):
            try:
                with open(synthetic_manifest_path, "r") as f:
                    manifest = json.load(f)
                # Sort by S_diff ascending (lowest S_diff = hardest negative)
                manifest.sort(key=lambda x: x.get("S_diff", 1.0))
                for item in manifest:
                    synth_path = item["synthetic_file"]
                    gen_path = item["source_genuine"]
                    s_diff = item.get("S_diff", 0.1)
                    if os.path.exists(synth_path) and os.path.exists(gen_path):
                        self.synthetic_pairs.append((gen_path, synth_path, 0, "synthetic_hard_negative", s_diff))
                print(f"[Curriculum] Loaded {len(self.synthetic_pairs)} synthetic hard negatives ranked by S_diff.")
            except Exception as e:
                print(f"[Warning] Could not load synthetic manifest: {e}")

        self.current_pairs = []
        self._cache = {}
        self.set_curriculum_stage(1)

    def set_curriculum_stage(self, stage: int):
        """
        Stage 1: Positives + Random Negatives only
        Stage 2: Positives + Random Negatives + Skilled Forgeries
        Stage 3: Positives + Random Negatives + Skilled Forgeries + Synthetic Hard Negatives (ranked by S_diff)
        Maintains a strict 1:1 balance between positive and negative pairs.
        """
        self.stage = stage
        raw_pairs = self.base_dataset.pairs # (p1, p2, label, pair_type)
        pos_pairs = [p for p in raw_pairs if p[3] == 'positive']
        filtered = []

        if stage == 1:
            rnd_pairs = [p for p in raw_pairs if p[3] == 'random_negative']
            for p in pos_pairs:
                filtered.append((p[0], p[1], p[2], p[3], 0.0))
            if len(rnd_pairs) > 0:
                for i in range(len(pos_pairs)):
                    p = rnd_pairs[i % len(rnd_pairs)]
                    filtered.append((p[0], p[1], p[2], p[3], 0.0))
        elif stage == 2:
            neg_pairs = [p for p in raw_pairs if p[3] in ('skilled_forgery', 'random_negative')]
            for p in pos_pairs:
                filtered.append((p[0], p[1], p[2], p[3], 0.0))
            if len(neg_pairs) > 0:
                for i in range(len(pos_pairs)):
                    p = neg_pairs[i % len(neg_pairs)]
                    filtered.append((p[0], p[1], p[2], p[3], 0.0))
        else:
            neg_pairs = [(p[0], p[1], p[2], p[3], 0.0) for p in raw_pairs if p[3] != 'positive']
            neg_pairs.extend(self.synthetic_pairs)
            for p in pos_pairs:
                filtered.append((p[0], p[1], p[2], p[3], 0.0))
            if len(neg_pairs) > 0:
                for i in range(len(pos_pairs)):
                    p = neg_pairs[i % len(neg_pairs)]
                    filtered.append((p[0], p[1], p[2], p[3], p[4] if len(p) > 4 else 0.0))

        self.current_pairs = filtered
        np.random.shuffle(self.current_pairs)

    def __len__(self) -> int:
        return len(self.current_pairs)

    def __getitem__(self, idx: int):
        p1, p2, label, p_type, s_diff = self.current_pairs[idx]

        if p1 not in self._cache:
            self._cache[p1] = load_and_preprocess_signature(p1, target_size=self.target_size)
        t1 = self._cache[p1]

        if p2 not in self._cache:
            self._cache[p2] = load_and_preprocess_signature(p2, target_size=self.target_size)
        t2 = self._cache[p2]

        if torch is not None:
            return {
                'img1': torch.from_numpy(t1).float(),
                'img2': torch.from_numpy(t2).float(),
                'label': torch.tensor(label, dtype=torch.float32),
                'pair_type': p_type,
                's_diff': float(s_diff)
            }

        return {
            'img1': t1,
            'img2': t2,
            'label': float(label),
            'pair_type': p_type,
            's_diff': float(s_diff)
        }


class BalancedBatchSampler(torch.utils.data.Sampler if TORCH_AVAILABLE else object):
    """
    Yields batches containing an exact 1:1 ratio of positive pairs to negative pairs
    per training step so gradients do not get biased toward negative push.
    """
    def __init__(self, dataset, batch_size: int = 32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = max(1, batch_size // 2)

    def __iter__(self):
        pos_indices = [i for i, p in enumerate(self.dataset.current_pairs) if p[2] == 1]
        neg_indices = [i for i, p in enumerate(self.dataset.current_pairs) if p[2] == 0]

        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return

        effective_half = min(self.half_batch, len(pos_indices), len(neg_indices))
        n_batches = min(len(pos_indices), len(neg_indices)) // effective_half
        for b in range(n_batches):
            batch = (
                pos_indices[b * effective_half : (b + 1) * effective_half] +
                neg_indices[b * effective_half : (b + 1) * effective_half]
            )
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        pos_indices = [i for i, p in enumerate(self.dataset.current_pairs) if p[2] == 1]
        neg_indices = [i for i, p in enumerate(self.dataset.current_pairs) if p[2] == 0]
        effective_half = max(1, min(self.half_batch, len(pos_indices), len(neg_indices)))
        return min(len(pos_indices), len(neg_indices)) // effective_half

def train_verifier(
    config_path: str,
    fold: int = 0,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    pairs_per_writer_override: Optional[int] = None,
    save_checkpoint_override: Optional[str] = None
):
    if not TORCH_AVAILABLE:
        print("[Notice] PyTorch is required to execute the Siamese ViT training loop.")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 2] Training Siamese ViT Verifier on: {device}")

    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"]["data_dir"]
    target_size = (cfg["image"]["height"], cfg["image"]["width"])
    batch_size = batch_size_override if batch_size_override is not None else cfg["verifier"]["batch_size"]
    lr = float(cfg["verifier"]["learning_rate"])
    epochs = epochs_override if epochs_override is not None else cfg["verifier"]["epochs"]
    pos_pairs = pairs_per_writer_override if pairs_per_writer_override is not None else cfg["dataset"].get("positive_pairs_per_writer", 20)
    save_dir = cfg["verifier"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1. Base Dataset & Curriculum Wrapper
    if dataset_name == "CEDAR":
        base_train_ds = CEDARPairDataset(data_dir=data_dir, fold=fold, is_train=True, target_size=target_size, positive_pairs_per_writer=pos_pairs, extract_conditions=False)
        val_ds = CEDARPairDataset(data_dir=data_dir, fold=fold, is_train=False, target_size=target_size, positive_pairs_per_writer=pos_pairs, extract_conditions=False)
    else:
        script_name = cfg["dataset"].get("subset", cfg["dataset"].get("script", "Dutch"))
        base_train_ds = ICDAR2011PairDataset(data_dir=data_dir, is_train=True, subset=script_name, target_size=target_size, positive_pairs_per_writer=pos_pairs, extract_conditions=False)
        val_ds = ICDAR2011PairDataset(data_dir=data_dir, is_train=False, subset=script_name, target_size=target_size, positive_pairs_per_writer=min(pos_pairs, 10), extract_conditions=False)

    train_ds = CurriculumPairDataset(base_train_ds, target_size=target_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model: Siamese ViT with L2-normalized embeddings
    backbone_name = cfg["verifier"].get("backbone", "vit_tiny_patch16_224")
    channels = cfg.get("image", {}).get("channels", 3)
    model = SiameseViT(
        model_name=backbone_name,
        pretrained=True,
        in_channels=channels,
        embedding_dim=cfg["verifier"]["embedding_dim"],
        img_size=target_size[0]
    ).to(device)

    # Freeze patch embedding and first 4 transformer blocks; train final 8 blocks and projection head
    model.freeze_prefix_layers(freeze_blocks=4)
    param_summary = model.get_trainable_parameter_summary()
    print(f"[SiameseViT] Trainable: {param_summary['trainable']:,} / {param_summary['total']:,} parameters (Frozen prefix: PatchEmbed + Blocks 0-3)")

    # 3. Loss: Sigmoidal Adaptive Margin Contrastive Loss with Biometric Multipliers & Label Smoothing
    loss_fn = SigmoidalAdaptiveMarginLoss(
        m_min=float(cfg["loss"].get("m_min", 0.60)),
        m_max=float(cfg["loss"].get("m_max", 1.10)),
        total_epochs=epochs,
        steepness=0.35,
        midpoint_ratio=0.50,
        skilled_factor=float(cfg["loss"].get("skilled_margin_factor", 0.95)),
        synthetic_factor=float(cfg["loss"].get("diffusion_margin_factor", 1.00)),
        random_factor=float(cfg["loss"].get("random_margin_factor", 1.05)),
        label_smoothing=float(cfg["loss"].get("label_smoothing", 0.05))
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    # Linear warmup for 2 epochs up to lr, followed by cosine decay over remaining epochs down to 1e-6
    warmup_epochs = min(2, max(1, epochs // 5))
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    # Curriculum stage milestones
    stage1_cutoff = max(2, int(epochs * 0.30)) # Random negatives first
    stage2_cutoff = max(4, int(epochs * 0.65)) # Skilled forgeries next
    # Stage 3: Inject synthetic hard negatives ranked by S_diff

    best_val_loss = float("inf")
    best_val_eer = float("inf")
    best_val_skilled_eer = float("inf")
    if save_checkpoint_override:
        best_model_path = save_checkpoint_override
        os.makedirs(os.path.dirname(os.path.abspath(best_model_path)), exist_ok=True)
        if os.path.exists(best_model_path):
            try:
                os.remove(best_model_path)
            except OSError:
                pass
    else:
        best_model_path = os.path.join(save_dir, "best_verifier.pth")

    print(f"Curriculum milestones: Stage 1 (Epochs 1-{stage1_cutoff}) -> Stage 2 (Epochs {stage1_cutoff+1}-{stage2_cutoff}) -> Stage 3 (Epochs {stage2_cutoff+1}-{epochs})")

    for epoch in range(1, epochs + 1):
        # Update curriculum stage
        if epoch <= stage1_cutoff:
            current_stage = 1
            stage_desc = "Stage 1: Random Negatives First (Coarse Manifold)"
        elif epoch <= stage2_cutoff:
            current_stage = 2
            stage_desc = "Stage 2: Injecting Skilled Human Forgeries"
        else:
            current_stage = 3
            stage_desc = "Stage 3: Injecting Synthetic Hard Negatives (Ranked by S_diff)"

        train_ds.set_curriculum_stage(current_stage)
        # Strict 1:1 ratio of positive pairs to negative pairs per training step
        train_sampler = BalancedBatchSampler(train_ds, batch_size=batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)

        # Update sigmoidal margin m(t)
        loss_fn.update_epoch(epoch=epoch, total_epochs=epochs)
        m_t = loss_fn.get_current_margin()

        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            labels = batch['label'].to(device)
            p_types = batch['pair_type']
            s_diffs = batch['s_diff']

            optimizer.zero_grad()
            distances, _, _ = model(img1, img2)
            loss = loss_fn(distances, labels, pair_types=p_types, s_diff_scores=s_diffs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation on unseen writers
        model.eval()
        total_val_loss = 0.0
        val_dists, val_labels, val_types = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                img1 = batch['img1'].to(device)
                img2 = batch['img2'].to(device)
                labels = batch['label'].to(device)
                p_types = batch['pair_type']
                distances, _, _ = model(img1, img2)
                loss = loss_fn(distances, labels, pair_types=p_types)
                total_val_loss += loss.item()
                val_dists.extend(distances.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_types.extend(p_types)

        avg_val_loss = total_val_loss / max(1, len(val_loader))

        # Dynamic validation biometric metrics (no arbitrary static threshold)
        from evaluate import evaluate_biometric_metrics
        val_metrics = evaluate_biometric_metrics(
            labels=np.array(val_labels),
            scores=np.array(val_dists),
            pair_types=val_types,
            num_thresholds=1000
        )
        val_eer_global = val_metrics["global"]["EER"]
        val_eer_skilled = val_metrics["skilled"]["EER"]
        val_eer_random = val_metrics["random"]["EER"]
        val_opt_tau = val_metrics["global"]["threshold"]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Margin m(t): {m_t:.3f} | {stage_desc}", flush=True)
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EER: {val_eer_global*100:.2f}% (Skilled: {val_eer_skilled*100:.2f}%, Random: {val_eer_random*100:.2f}%) | Opt Tau: {val_opt_tau:.4f}", flush=True)

        # Determine checkpoint saving using dynamic validation EER (prioritizing skilled forgery discrimination and global EER)
        is_improved = False
        if epoch == 1:
            is_improved = True
        elif val_eer_skilled < best_val_skilled_eer and val_eer_random <= 0.05:
            is_improved = True
        elif val_eer_global < best_val_eer:
            is_improved = True
        elif abs(val_eer_global - best_val_eer) < 0.005 and avg_val_loss < best_val_loss:
            is_improved = True

        if is_improved:
            best_val_eer = val_eer_global
            best_val_skilled_eer = min(best_val_skilled_eer, val_eer_skilled)
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best verifier checkpoint (Val EER: {val_eer_global*100:.2f}%, Skilled: {val_eer_skilled*100:.2f}%, Loss: {avg_val_loss:.4f}) to: {best_model_path}", flush=True)

    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    final_path = best_model_path.replace(".pth", "_final.pth")
    torch.save(model.state_dict(), final_path)

    print(f"\n[Phase 2 Complete] Best Siamese ViT verifier saved to: {best_model_path}")
    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese ViT Verifier with Curriculum Learning")
    parser.add_argument("--config", type=str, default="configs/cedar.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--pairs_per_writer", type=int, default=None, help="Override pairs per writer")
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save best checkpoint")
    args = parser.parse_args()
    train_verifier(
        config_path=args.config,
        fold=args.fold,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        pairs_per_writer_override=args.pairs_per_writer,
        save_checkpoint_override=args.save_checkpoint
    )
