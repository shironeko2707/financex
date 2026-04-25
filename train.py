"""
Vietnam stock prediction training script. Single-GPU, single-file.
The agent modifies this file to experiment with architectures.
Usage: uv run train.py [TICKER]
  e.g. uv run train.py VNINDEX
       uv run train.py VNM
"""

import os
import gc
import sys
import time
import copy
import math
import mlflow
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, SEQ_LEN, TICKER, CACHE_DIR,
    download_and_cache, normalize_features, make_dataloader, evaluate,
)

# ---------------------------------------------------------------------------
# Ticker override from CLI
# ---------------------------------------------------------------------------

TARGET_TICKER = sys.argv[1] if len(sys.argv) > 1 else TICKER

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

MLFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("vn-stock-prediction")
mlflow.pytorch.autolog()

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device_type = "cuda"
elif torch.backends.mps.is_available():
    device_type = "mps"
else:
    device_type = "cpu"
device = torch.device(device_type)
print(f"Device: {device_type}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(self, num_features, hidden_dim=128, n_layers=4, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0,
                          bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1]
        return self.head(out)

class GRUWithTCN(nn.Module):
    def __init__(self, num_features, hidden_dim=128, n_layers=4, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0,
                          bidirectional=True)
        self.tcn = TCNModel(num_features=num_features, channels=hidden_dim//2, n_layers=4, dropout=dropout, seq_len=seq_len)
        combined = hidden_dim * 2 + hidden_dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(combined),
            nn.Dropout(dropout),
            nn.Linear(combined, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)  # (B, T, 2H)
        gru_out = gru_out[:, -1]   # (B, 2H)
        tcn_out = self.tcn(x)     # (B, 1)
        tcn_out = tcn_out.squeeze(-1)  # (B,)
        combined = torch.cat([gru_out, tcn_out], dim=-1)
        return self.head(combined)

class LSTMGRUEnsemble(nn.Module):
    def __init__(self, num_features, hidden_dim=128, n_layers=4, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0,
                            bidirectional=True)
        self.gru = nn.GRU(num_features, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0,
                          bidirectional=True)
        combined = hidden_dim * 4
        self.head = nn.Sequential(
            nn.LayerNorm(combined),
            nn.Dropout(dropout),
            nn.Linear(combined, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        combined = torch.cat([lstm_out[:, -1], gru_out[:, -1]], dim=-1)
        return self.head(combined)

class LSTMWithAttention(nn.Module):
    def __init__(self, num_features, hidden_dim=128, n_layers=4, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0,
                            bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1]
        return self.head(out)

class TransformerPool(nn.Module):
    def __init__(self, num_features, model_dim=128, n_heads=4, n_layers=3, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.input_proj = nn.Linear(num_features, model_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.dropout(torch.relu(self.bn1(self.conv1(x))))
        h = self.dropout(torch.relu(self.bn2(self.conv2(h))))
        return h + self.residual(x)


class TCNModel(nn.Module):
    def __init__(self, num_features, channels=64, n_layers=8, dropout=0.2, seq_len=SEQ_LEN):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = num_features
        for i in range(n_layers):
            dilation = 2 ** (i % 6)
            self.blocks.append(TCNBlock(in_ch, channels, kernel_size=7, dilation=dilation, dropout=dropout))
            in_ch = channels
        self.head = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# ---------------------------------------------------------------------------
# Hyperparameters (best config from BTC experiments)
# ---------------------------------------------------------------------------

MODEL_DIM = 192
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.185

LEARNING_RATE = 5e-4
INPUT_NOISE = 0.04
WEIGHT_DECAY = 4.5e-2
BATCH_SIZE = 64

WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.05

VAL_INTERVAL = 10
RDROP_ALPHA = 1.5
LABEL_SMOOTH = 0.05
CONF_PENALTY = 0.15

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
np.random.seed(42)

# Load data
print(f"Target ticker: {TARGET_TICKER}")
features, targets, meta = download_and_cache(TARGET_TICKER)
features = normalize_features(features, meta["train_size"])
num_features = meta["num_features"]
train_size = meta["train_size"]

print(f"Num features: {num_features}")
print(f"Baseline accuracy: {max(targets[:train_size].mean(), 1-targets[:train_size].mean()):.4f}")

# Build model
model = GRUModel(
    num_features=num_features,
    hidden_dim=MODEL_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    seq_len=SEQ_LEN,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,} ({num_params/1000:.1f}K)")

# Start MLflow run
run_name = f"{TARGET_TICKER}_d={MODEL_DIM}_h={N_HEADS}_l={N_LAYERS}_wd={WEIGHT_DECAY}_conf={CONF_PENALTY}"
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "ticker": TARGET_TICKER,
        "model_dim": MODEL_DIM,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "warmup_ratio": WARMUP_RATIO,
        "final_lr_frac": FINAL_LR_FRAC,
        "rdrop_alpha": RDROP_ALPHA,
        "label_smooth": LABEL_SMOOTH,
        "conf_penalty": CONF_PENALTY,
        "seq_len": SEQ_LEN,
        "num_features": num_features,
        "num_params": num_params,
    })

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Dataloader
    train_loader = make_dataloader(
        features, targets,
        start_idx=0, end_idx=train_size,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=True,
    )

    # LR schedule with mid-training restart
    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        # Two cosine cycles: 0-50% and 50-100%
        if progress < 0.5:
            t = (progress - WARMUP_RATIO) / (0.5 - WARMUP_RATIO)
        else:
            t = (progress - 0.5) / 0.5
        return max(FINAL_LR_FRAC, 0.5 * (1 + math.cos(math.pi * t)))

    print(f"Time budget: {TIME_BUDGET}s")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    def sync_device():
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

    t_start_training = time.time()
    total_training_time = 0
    step = 0
    smooth_loss = 0
    smooth_acc = 0

    best_val_acc = 0.0
    best_state = None
    last_val_time = 0.0

    X_batch, y_batch, epoch = next(train_loader)

    while True:
        sync_device()
        t0 = time.time()

        model.train()
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Add Gaussian noise to input
        if INPUT_NOISE > 0:
            X_noisy = X_batch + torch.randn_like(X_batch) * INPUT_NOISE
        else:
            X_noisy = X_batch

        logits1 = model(X_noisy).squeeze(-1)
        logits2 = model(X_noisy).squeeze(-1)  # second forward with different dropout
        y_smooth = y_batch * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
        loss_ce = 0.5 * (F.binary_cross_entropy_with_logits(logits1, y_smooth)
                       + F.binary_cross_entropy_with_logits(logits2, y_smooth))
        # KL divergence between two predictions (symmetric)
        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)
        kl = 0.5 * (p1 * (p1 / (p2 + 1e-8)).log() + (1-p1) * ((1-p1) / (1-p2 + 1e-8)).log()
                  + p2 * (p2 / (p1 + 1e-8)).log() + (1-p2) * ((1-p2) / (1-p1 + 1e-8)).log())
        # Confidence penalty: negative entropy of predictions
        avg_p = 0.5 * (p1 + p2)
        entropy = -(avg_p * (avg_p + 1e-8).log() + (1 - avg_p) * (1 - avg_p + 1e-8).log())
        rdrop_w = RDROP_ALPHA
        conf_w = CONF_PENALTY * min(1.0, total_training_time / (TIME_BUDGET * 0.3))
        loss = loss_ce + rdrop_w * kl.mean() - conf_w * entropy.mean()
        logits = logits1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # LR schedule
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * lrm

        optimizer.step()

        # Prefetch next batch
        X_batch, y_batch, epoch = next(train_loader)

        # Metrics
        train_loss = loss.item()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_batch.to(device)).float().mean().item()

        sync_device()
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        # Periodic validation
        if total_training_time - last_val_time >= VAL_INTERVAL and step > 10:
            val_metrics = evaluate(model, features, targets, train_size, device, SEQ_LEN)
            val_acc = val_metrics["accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                print(f"\n  [VAL] step={step} t={total_training_time:.0f}s acc={val_acc:.4f} sharpe={val_metrics['sharpe']:.4f} ** NEW BEST **")
            else:
                print(f"\n  [VAL] step={step} t={total_training_time:.0f}s acc={val_acc:.4f} sharpe={val_metrics['sharpe']:.4f}")
            last_val_time = total_training_time
            mlflow.log_metrics({
                "val_accuracy": val_acc,
                "val_sharpe": val_metrics["sharpe"],
                "val_loss": val_metrics["avg_loss"],
                "train_accuracy": debiased_acc,
                "train_loss": debiased_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=step)

        # Logging
        ema_c = 0.95
        smooth_loss = ema_c * smooth_loss + (1 - ema_c) * train_loss
        smooth_acc = ema_c * smooth_acc + (1 - ema_c) * acc
        debiased_loss = smooth_loss / (1 - ema_c ** (step + 1))
        debiased_acc = smooth_acc / (1 - ema_c ** (step + 1))
        pct_done = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | acc: {debiased_acc:.3f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

        step += 1

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------

    if best_state is not None:
        print(f"Restoring best checkpoint (val_acc={best_val_acc:.4f})")
        model.load_state_dict(best_state)

    metrics = evaluate(model, features, targets, train_size, device, SEQ_LEN)

    t_end = time.time()

    print("---")
    print(f"ticker:           {TARGET_TICKER}")
    print(f"val_accuracy:     {metrics['accuracy']:.6f}")
    print(f"val_sharpe:       {metrics['sharpe']:.4f}")
    print(f"val_loss:         {metrics['avg_loss']:.6f}")
    print(f"val_samples:      {metrics['n_val']}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_K:     {num_params / 1000:.1f}")
    print(f"model_dim:        {MODEL_DIM}")
    print(f"n_layers:         {N_LAYERS}")

    mlflow.log_metrics({
        "final_val_accuracy": metrics['accuracy'],
        "final_val_sharpe": metrics['sharpe'],
        "final_val_loss": metrics['avg_loss'],
        "best_val_accuracy": best_val_acc,
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "num_steps": step,
    })
    mlflow.log_param("status", "completed")

    print("MLflow run:", mlflow.active_run().info.run_id)
