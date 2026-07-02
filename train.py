#!/usr/bin/env python3
"""
Train.py

Defines the VIO network (ResNet18 visual encoder + GRU IMU encoder + MLP pose
head) and trains it on the dataset built by DataLoader.py.

The network and CONFIG are defined at module level so Test.py can import them:
    from Train import VIONet

Run:
    python3 Train.py
    python3 Train.py --dataset-root vio_dataset --epochs 80 --batch-size 32
"""

import os
import copy
import time
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from data_loader import (
    CONFIG,
    VIODataset,
    discover_sequences,
    compute_imu_stats,
)


# ===========================================================================
# Model:  ResNet18 + GRU + MLP
# ===========================================================================

def _build_resnet18(pretrained):
    """Load a resnet18 backbone across old and new torchvision APIs."""
    import torchvision.models as models
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(pretrained=pretrained)
    net.fc = nn.Identity()   # output becomes a 512 dim feature
    return net


class VIONet(nn.Module):
    """
    images : (B, N, 3, H, W)  -> shared ResNet18 -> per frame 512 features
                              -> linear projection to visual_embed_dim
                              -> concatenate the N frame embeddings
    imu    : (B, L, 6)        -> GRU or LSTM -> last hidden state
    fusion : concat(visual, imu) -> MLP -> 6 relative pose values
    """

    def __init__(self, config):
        super().__init__()
        self.N = config["stack_size"]
        embed = config["visual_embed_dim"]

        self.backbone = _build_resnet18(config["pretrained_resnet"])
        self.visual_proj = nn.Sequential(
            nn.Linear(512, embed),
            nn.ReLU(inplace=True),
        )

        rnn_cls = nn.LSTM if config["rnn_type"].lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=config["imu_channels"],
            hidden_size=config["imu_hidden"],
            num_layers=config["imu_layers"],
            batch_first=True,
        )
        self.is_lstm = config["rnn_type"].lower() == "lstm"

        fusion_in = self.N * embed + config["imu_hidden"]
        self.head = nn.Sequential(
            nn.Linear(fusion_in, config["fusion_hidden"]),
            nn.ReLU(inplace=True),
            nn.Linear(config["fusion_hidden"], config["fusion_hidden"] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config["fusion_hidden"] // 2, config["output_dim"]),
        )

    def forward(self, images, imu):
        b, n, c, h, w = images.shape
        feats = self.backbone(images.view(b * n, c, h, w))   # (b*n, 512)
        feats = self.visual_proj(feats)                      # (b*n, embed)
        visual = feats.view(b, n * feats.shape[-1])          # (b, N*embed)

        out, hidden = self.rnn(imu)
        if self.is_lstm:
            hidden = hidden[0]
        imu_feat = hidden[-1]                                # (b, imu_hidden)

        fused = torch.cat([visual, imu_feat], dim=1)
        return self.head(fused)                              # (b, 6)


# ===========================================================================
# Loss
# ===========================================================================

class PoseLoss(nn.Module):
    """Weighted sum of translation MSE and rotation MSE."""

    def __init__(self, rot_weight):
        super().__init__()
        self.rot_weight = rot_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        trans = self.mse(pred[:, :3], target[:, :3])
        rot = self.mse(pred[:, 3:], target[:, 3:])
        return trans + self.rot_weight * rot, trans.item(), rot.item()


# ===========================================================================
# Training
# ===========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_sequences(sequences, val_fraction, seed):
    rng = random.Random(seed)
    seqs = list(sequences)
    rng.shuffle(seqs)
    if len(seqs) < 2:
        # not enough trajectories to hold one out, reuse the same set
        return seqs, seqs
    n_val = max(1, int(round(len(seqs) * val_fraction)))
    return seqs[n_val:], seqs[:n_val]


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    tot, tot_t, tot_r, count = 0.0, 0.0, 0.0, 0
    # for images, imu, target in loader:
    for images, imu, target in tqdm(loader, desc="Batch", unit="batch", leave=False):
        images = images.to(device)
        imu = imu.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(training):
            pred = model(images, imu)
            loss, t_item, r_item = loss_fn(pred, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = images.shape[0]
        tot += loss.item() * bs
        tot_t += t_item * bs
        tot_r += r_item * bs
        count += bs

    count = max(count, 1)
    return tot / count, tot_t / count, tot_r / count


def main():
    parser = argparse.ArgumentParser(description="Train the drone VIO model.")
    parser.add_argument("--dataset-root", default=CONFIG["dataset_root"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--checkpoint", default=CONFIG["checkpoint"])
    parser.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    args = parser.parse_args()

    cfg = copy.deepcopy(CONFIG)
    cfg["dataset_root"] = args.dataset_root
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["lr"] = args.lr
    cfg["checkpoint"] = args.checkpoint
    cfg["num_workers"] = args.num_workers

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    sequences = discover_sequences(cfg["dataset_root"])
    if not sequences:
        raise SystemExit(f"No samples.csv found under {cfg['dataset_root']}")
    train_seqs, val_seqs = split_sequences(sequences, cfg["val_fraction"], cfg["seed"])
    print(f"Sequences: {len(sequences)} total, {len(train_seqs)} train, {len(val_seqs)} val")

    # IMU normalisation from the training sequences only
    imu_mean, imu_std = compute_imu_stats(train_seqs)

    train_ds = VIODataset(train_seqs, cfg, imu_mean, imu_std)
    val_ds = VIODataset(val_seqs, cfg, imu_mean, imu_std)
    print(f"Windows: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = TorchDataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = TorchDataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )

    model = VIONet(cfg).to(device)
    loss_fn = PoseLoss(cfg["rot_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_val = float("inf")
    # for epoch in range(1, cfg["epochs"] + 1):
    for epoch in tqdm(range(1, cfg["epochs"] + 1), desc="Training", unit="epoch"):
        t0 = time.time()
        tr_loss, tr_t, tr_r = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_t, va_r = run_epoch(model, val_loader, loss_fn, device, None)
        scheduler.step(va_loss)

        print(f"epoch {epoch:03d}/{cfg['epochs']} "
              f"train {tr_loss:.4f} (t {tr_t:.4f} r {tr_r:.4f})  "
              f"val {va_loss:.4f} (t {va_t:.4f} r {va_r:.4f})  "
              f"{time.time() - t0:.1f}s")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "imu_mean": imu_mean,
                "imu_std": imu_std,
                "epoch": epoch,
                "val_loss": va_loss,
            }, cfg["checkpoint"])
            print(f"  saved best checkpoint to {cfg['checkpoint']} (val {va_loss:.4f})")

    print(f"Done. Best val loss {best_val:.4f}. Checkpoint: {cfg['checkpoint']}")


if __name__ == "__main__":
    main()