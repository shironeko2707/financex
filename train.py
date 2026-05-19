"""
Vietnam stock prediction training script. Single-GPU, single-file.
Ensemble pipeline with confidence-based selective trading and cost-aware evaluation.
Usage: uv run train.py [TICKER]
       WALK_FORWARD=1 uv run train.py [TICKER]  # walk-forward validation
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
    make_sequences,
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
mlflow.pytorch.autolog(disable=True)

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
DROPOUT = 0.19

LEARNING_RATE = 5e-4
INPUT_NOISE = 0.04
WEIGHT_DECAY = 5e-2
BATCH_SIZE = 64

WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.05

VAL_INTERVAL = 10
RDROP_ALPHA = 1.5
LABEL_SMOOTH = 0.01
CONF_PENALTY = 0.13

# ---------------------------------------------------------------------------
# Ensemble & Evaluation Configuration
# ---------------------------------------------------------------------------

N_SEEDS = 10
SEEDS = [42, 137, 256, 512, 1024, 2049, 9999, 12345, 7777, 4321]
CONFIDENCE_THRESHOLD = 0.587
COST_PER_TRADE = 0.003
MAX_DRAWDOWN_LIMIT = 0.15
KELLY_FRACTION = 0.05
WALK_FORWARD = os.environ.get("WALK_FORWARD", "0") == "1"

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def find_year_boundaries(dates):
    """Given list of 'YYYY-MM-DD' strings, return {year: exclusive_end_index}."""
    boundaries = {}
    for i, d in enumerate(dates):
        year = int(d[:4])
        boundaries[year] = i + 1
    return boundaries


def sync_device():
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def get_lr_multiplier(progress):
    """Dual-cosine LR schedule with warmup."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 0.5:
        t = (progress - WARMUP_RATIO) / (0.5 - WARMUP_RATIO)
    else:
        t = (progress - 0.5) / 0.5
    return max(FINAL_LR_FRAC, 0.5 * (1 + math.cos(math.pi * t)))


def train_single_model(features, targets, train_start, train_end,
                       num_features, seed, time_budget, dev):
    """Train one GRU model. Returns (state_dict, best_val_acc, steps)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GRUModel(
        num_features=num_features,
        hidden_dim=MODEL_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
    ).to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_dataloader(
        features, targets,
        start_idx=train_start, end_idx=train_end,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, shuffle=True,
    )

    total_training_time = 0
    step = 0
    smooth_loss = 0.0
    smooth_acc = 0.0
    debiased_loss = 0.0
    debiased_acc = 0.0

    best_val_acc = 0.0
    best_state = None
    last_val_time = 0.0

    gc.enable()
    gc.collect()

    X_batch, y_batch, epoch = next(train_loader)

    while True:
        sync_device()
        t0 = time.time()

        model.train()
        X_batch = X_batch.to(dev)
        y_batch = y_batch.to(dev)

        if INPUT_NOISE > 0:
            X_noisy = X_batch + torch.randn_like(X_batch) * INPUT_NOISE
        else:
            X_noisy = X_batch

        logits1 = model(X_noisy).squeeze(-1)
        logits2 = model(X_noisy).squeeze(-1)
        y_smooth = y_batch * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
        loss_ce = 0.5 * (F.binary_cross_entropy_with_logits(logits1, y_smooth)
                       + F.binary_cross_entropy_with_logits(logits2, y_smooth))

        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)
        kl = 0.5 * (p1 * (p1 / (p2 + 1e-8)).log() + (1-p1) * ((1-p1) / (1-p2 + 1e-8)).log()
                  + p2 * (p2 / (p1 + 1e-8)).log() + (1-p2) * ((1-p2) / (1-p1 + 1e-8)).log())
        avg_p = 0.5 * (p1 + p2)
        entropy = -(avg_p * (avg_p + 1e-8).log() + (1 - avg_p) * (1 - avg_p + 1e-8).log())
        rdrop_w = RDROP_ALPHA
        conf_w = CONF_PENALTY * min(1.0, total_training_time / (time_budget * 0.3)) if time_budget > 0 else CONF_PENALTY
        loss = loss_ce + rdrop_w * kl.mean() - conf_w * entropy.mean()
        logits = logits1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        progress = min(total_training_time / time_budget, 1.0) if time_budget > 0 else 1.0
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * lrm

        optimizer.step()
        X_batch, y_batch, epoch = next(train_loader)

        train_loss = loss.item()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_batch.to(dev)).float().mean().item()

        sync_device()
        dt = time.time() - t0

        if step > 5:
            total_training_time += dt

        # EMA metrics (computed before validation to avoid undefined variable bug)
        ema_c = 0.95
        smooth_loss = ema_c * smooth_loss + (1 - ema_c) * train_loss
        smooth_acc = ema_c * smooth_acc + (1 - ema_c) * acc
        debiased_loss = smooth_loss / (1 - ema_c ** (step + 1))
        debiased_acc = smooth_acc / (1 - ema_c ** (step + 1))

        # Periodic validation
        if total_training_time - last_val_time >= VAL_INTERVAL and step > 10:
            val_metrics = evaluate(model, features, targets, train_end, dev, SEQ_LEN)
            val_acc = val_metrics["accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            last_val_time = total_training_time

        pct_done = 100 * progress
        remaining = max(0, time_budget - total_training_time)
        print(f"\r  step {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | acc: {debiased_acc:.3f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

        step += 1
        if step > 5 and total_training_time >= time_budget:
            break

    gc.enable()
    print()

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    del model, optimizer
    return best_state, best_val_acc, step


def ensemble_predict(model_states, num_features, features, targets,
                     start_idx, end_idx, dev):
    """Run inference with N model states. Returns (mean_probs, y_true)."""
    X_seqs, y_seqs = make_sequences(features, targets, start_idx, end_idx, SEQ_LEN)
    X_tensor = torch.from_numpy(X_seqs)
    all_probs = []

    for state_dict in model_states:
        model = GRUModel(
            num_features=num_features,
            hidden_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
            seq_len=SEQ_LEN,
        ).to(dev)
        model.load_state_dict(state_dict)
        model.eval()

        probs = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 256):
                batch = X_tensor[i:i+256].to(dev)
                logits = model(batch).squeeze(-1)
                probs.append(torch.sigmoid(logits).cpu().numpy())

        all_probs.append(np.concatenate(probs))
        del model

    mean_probs = np.stack(all_probs).mean(axis=0)
    return mean_probs, y_seqs


def evaluate_selective(mean_probs, targets, features, val_start, n_samples,
                       threshold=CONFIDENCE_THRESHOLD,
                       cost_per_trade=COST_PER_TRADE,
                       kelly_fraction=KELLY_FRACTION):
    """Evaluate with confidence filtering, costs, position sizing, and drawdown."""
    # Standard accuracy
    all_preds = (mean_probs > 0.5).astype(np.float32)
    accuracy = float((all_preds == targets).mean())

    # Selective trading mask
    trade_mask = (mean_probs > threshold) | (mean_probs < (1.0 - threshold))
    coverage = float(trade_mask.mean())

    if trade_mask.sum() > 0:
        sel_preds = (mean_probs[trade_mask] > 0.5).astype(np.float32)
        selective_accuracy = float((sel_preds == targets[trade_mask]).mean())
    else:
        selective_accuracy = 0.0

    # Trading returns: prediction at sample j corresponds to features index (val_start + j).
    # The return from acting on that prediction is the next day's ret_1d = features[val_start + j + 1, 0].
    returns = np.zeros(n_samples, dtype=np.float32)
    for j in range(n_samples):
        idx = val_start + j + 1
        if idx < len(features):
            returns[j] = features[idx, 0]

    # Standard Sharpe (all positions, no costs)
    positions = np.where(all_preds == 1, 1.0, -1.0)
    strategy_returns = positions * returns
    if len(strategy_returns) > 1 and strategy_returns.std() > 1e-10:
        sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Position sizing: fractional Kelly
    # edge = 2 * max(p, 1-p) - 1; ranges from 0 (p=0.5) to 1 (p=0 or 1)
    edge = 2.0 * np.maximum(mean_probs, 1.0 - mean_probs) - 1.0
    position_size = edge * kelly_fraction
    direction = np.where(mean_probs > 0.5, 1.0, -1.0)

    # Apply trade mask: position = 0 on skip days
    sized_positions = np.where(trade_mask, direction * position_size, 0.0)

    # Transaction costs on position changes
    position_changes = np.abs(np.diff(np.concatenate([[0.0], sized_positions])))
    costs = cost_per_trade * position_changes

    # Net returns
    net_returns = sized_positions * returns - costs

    # Net Sharpe (only on days with positions)
    traded_net = net_returns[trade_mask]
    if len(traded_net) > 1 and traded_net.std() > 1e-10:
        net_sharpe = float(traded_net.mean() / traded_net.std() * np.sqrt(252))
    else:
        net_sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "selective_accuracy": selective_accuracy,
        "sharpe": sharpe,
        "net_sharpe": net_sharpe,
        "coverage": coverage,
        "max_drawdown": max_drawdown,
        "high_risk": max_drawdown > MAX_DRAWDOWN_LIMIT,
        "n_trades": int(trade_mask.sum()),
        "n_val": n_samples,
    }


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

t_start = time.time()

print(f"Target ticker: {TARGET_TICKER}")
features_raw, targets, meta = download_and_cache(TARGET_TICKER)
features_raw = features_raw.copy()
num_features = meta["num_features"]
train_size = meta["train_size"]

print(f"Num features: {num_features}")
print(f"Baseline accuracy: {max(targets[:train_size].mean(), 1-targets[:train_size].mean()):.4f}")

_tmp = GRUModel(num_features=num_features, hidden_dim=MODEL_DIM, n_layers=N_LAYERS, dropout=DROPOUT, seq_len=SEQ_LEN)
num_params = sum(p.numel() for p in _tmp.parameters())
del _tmp
print(f"Model parameters: {num_params:,} ({num_params/1000:.1f}K)")

if not WALK_FORWARD:
    # -----------------------------------------------------------------------
    # Default Mode: Single-window multi-seed ensemble
    # -----------------------------------------------------------------------
    features = normalize_features(features_raw, train_size)
    per_model_budget = (TIME_BUDGET - 30) / N_SEEDS

    print(f"Ensemble training: {N_SEEDS} models x {per_model_budget:.0f}s each")
    print(f"Time budget: {TIME_BUDGET}s (reserved 30s for eval overhead)")

    run_name = f"{TARGET_TICKER}_ensemble_{N_SEEDS}seeds_d={MODEL_DIM}_l={N_LAYERS}"
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
            "n_seeds": N_SEEDS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "cost_per_trade": COST_PER_TRADE,
        })

        ensemble_states = []
        best_single_acc = 0.0
        best_single_state = None
        total_steps = 0

        for i, seed in enumerate(SEEDS[:N_SEEDS]):
            print(f"\n--- Model {i+1}/{N_SEEDS} (seed={seed}) ---")
            state, val_acc, steps = train_single_model(
                features, targets, 0, train_size,
                num_features, seed, per_model_budget, device,
            )
            ensemble_states.append(state)
            total_steps += steps
            print(f"  Model {i+1}: val_acc={val_acc:.4f}, steps={steps}")
            if val_acc > best_single_acc:
                best_single_acc = val_acc
                best_single_state = state

        # Backward-compatible single-model evaluation
        best_model = GRUModel(
            num_features=num_features, hidden_dim=MODEL_DIM,
            n_layers=N_LAYERS, dropout=DROPOUT, seq_len=SEQ_LEN,
        ).to(device)
        best_model.load_state_dict(best_single_state)
        single_metrics = evaluate(best_model, features, targets, train_size, device, SEQ_LEN)
        del best_model

        # Ensemble evaluation
        print(f"\n--- Ensemble Evaluation ({N_SEEDS} models) ---")
        mean_probs, y_true = ensemble_predict(
            ensemble_states, num_features, features, targets,
            train_size, len(features), device,
        )

        ens_metrics = evaluate_selective(
            mean_probs, y_true, features, train_size, len(mean_probs),
        )

        t_end = time.time()
        total_training_time = t_end - t_start

        # Standard metrics (backward compatible)
        print("---")
        print(f"ticker:           {TARGET_TICKER}")
        print(f"val_accuracy:     {single_metrics['accuracy']:.6f}")
        print(f"val_sharpe:       {single_metrics['sharpe']:.4f}")
        print(f"val_loss:         {single_metrics['avg_loss']:.6f}")
        print(f"val_samples:      {single_metrics['n_val']}")
        print(f"training_seconds: {total_training_time:.1f}")
        print(f"total_seconds:    {t_end - t_start:.1f}")
        print(f"num_steps:        {total_steps}")
        print(f"num_params_K:     {num_params / 1000:.1f}")
        print(f"model_dim:        {MODEL_DIM}")
        print(f"n_layers:         {N_LAYERS}")

        # Ensemble metrics
        print(f"ensemble_size:    {N_SEEDS}")
        print(f"ens_accuracy:     {ens_metrics['accuracy']:.6f}")
        print(f"ens_sel_accuracy: {ens_metrics['selective_accuracy']:.6f}")
        print(f"ens_sharpe:       {ens_metrics['sharpe']:.4f}")
        print(f"ens_net_sharpe:   {ens_metrics['net_sharpe']:.4f}")
        print(f"ens_coverage:     {ens_metrics['coverage']:.4f}")
        print(f"ens_max_drawdown: {ens_metrics['max_drawdown']:.4f}")
        print(f"ens_high_risk:    {ens_metrics['high_risk']}")
        print(f"ens_n_trades:     {ens_metrics['n_trades']}")

        mlflow.log_metrics({
            "final_val_accuracy": single_metrics['accuracy'],
            "final_val_sharpe": single_metrics['sharpe'],
            "final_val_loss": single_metrics['avg_loss'],
            "best_val_accuracy": best_single_acc,
            "ens_accuracy": ens_metrics['accuracy'],
            "ens_selective_accuracy": ens_metrics['selective_accuracy'],
            "ens_sharpe": ens_metrics['sharpe'],
            "ens_net_sharpe": ens_metrics['net_sharpe'],
            "ens_coverage": ens_metrics['coverage'],
            "ens_max_drawdown": ens_metrics['max_drawdown'],
            "ens_n_trades": float(ens_metrics['n_trades']),
            "training_seconds": total_training_time,
            "total_seconds": t_end - t_start,
            "num_steps": total_steps,
        })
        mlflow.log_param("status", "completed")
        print("MLflow run:", mlflow.active_run().info.run_id)

else:
    # -----------------------------------------------------------------------
    # Walk-Forward Validation Mode
    # -----------------------------------------------------------------------
    print("=== WALK-FORWARD VALIDATION MODE ===")

    boundaries = find_year_boundaries(meta['dates'])
    years = sorted(boundaries.keys())

    # Build windows: train through year Y-1, validate on year Y
    windows = []
    for year in range(2021, max(years) + 1):
        train_end = boundaries.get(year - 1)
        val_end = boundaries.get(year, len(features_raw))
        if train_end is None or train_end < SEQ_LEN + 100:
            continue
        if val_end - train_end < 20:
            continue
        windows.append({
            "label": f"val_{year}",
            "train_end": train_end,
            "val_start": train_end,
            "val_end": val_end,
        })

    wf_n_seeds = min(3, N_SEEDS)
    per_model_budget = max(20, (TIME_BUDGET - 30) / (len(windows) * wf_n_seeds))

    print(f"Windows: {len(windows)}, Seeds per window: {wf_n_seeds}")
    print(f"Per-model budget: {per_model_budget:.0f}s")

    run_name = f"{TARGET_TICKER}_walkforward_{len(windows)}win"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "ticker": TARGET_TICKER,
            "mode": "walk_forward",
            "n_windows": len(windows),
            "n_seeds": wf_n_seeds,
            "per_model_budget": per_model_budget,
            "model_dim": MODEL_DIM,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
        })

        all_window_metrics = []

        for wi, window in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"Window {wi+1}/{len(windows)}: {window['label']}")
            print(f"  Train: [0, {window['train_end']}), Val: [{window['val_start']}, {window['val_end']})")

            features_w = normalize_features(features_raw, window['train_end'])

            window_states = []
            for si, seed in enumerate(SEEDS[:wf_n_seeds]):
                print(f"\n  Model {si+1}/{wf_n_seeds} (seed={seed})")
                state, val_acc, steps = train_single_model(
                    features_w, targets, 0, window['train_end'],
                    num_features, seed, per_model_budget, device,
                )
                window_states.append(state)

            mean_probs, y_true = ensemble_predict(
                window_states, num_features, features_w, targets,
                window['val_start'], window['val_end'], device,
            )

            wm = evaluate_selective(
                mean_probs, y_true, features_w, window['val_start'], len(mean_probs),
            )
            wm['window'] = window['label']
            all_window_metrics.append(wm)

            print(f"\n  {window['label']}: acc={wm['accuracy']:.4f} sel_acc={wm['selective_accuracy']:.4f} "
                  f"sharpe={wm['sharpe']:.4f} net_sharpe={wm['net_sharpe']:.4f} "
                  f"coverage={wm['coverage']:.4f} drawdown={wm['max_drawdown']:.4f}")

        t_end = time.time()

        print(f"\n{'='*60}")
        print("WALK-FORWARD SUMMARY")
        print(f"{'='*60}")
        print(f"{'Window':<12} {'Acc':>6} {'SelAcc':>7} {'Sharpe':>7} {'NetShp':>7} {'Cover':>6} {'MaxDD':>6}")
        print("-" * 60)

        net_sharpes = []
        for wm in all_window_metrics:
            print(f"{wm['window']:<12} {wm['accuracy']:>6.4f} {wm['selective_accuracy']:>7.4f} "
                  f"{wm['sharpe']:>7.4f} {wm['net_sharpe']:>7.4f} {wm['coverage']:>6.4f} {wm['max_drawdown']:>6.4f}")
            net_sharpes.append(wm['net_sharpe'])

        print("-" * 60)
        mean_ns = float(np.mean(net_sharpes))
        std_ns = float(np.std(net_sharpes))
        min_ns = float(np.min(net_sharpes))
        print(f"Net Sharpe: mean={mean_ns:.4f} std={std_ns:.4f} min={min_ns:.4f}")

        production_ready = min_ns > 0.5
        print(f"\nPRODUCTION READY: {'YES' if production_ready else 'NO'}")
        if not production_ready:
            print(f"  Requires min(net_sharpe) > 0.5 across all windows (got {min_ns:.4f})")

        # Standard format for compatibility
        print("---")
        print(f"ticker:           {TARGET_TICKER}")
        print(f"val_accuracy:     {np.mean([w['accuracy'] for w in all_window_metrics]):.6f}")
        print(f"val_sharpe:       {mean_ns:.4f}")
        print(f"total_seconds:    {t_end - t_start:.1f}")
        print(f"num_params_K:     {num_params / 1000:.1f}")
        print(f"model_dim:        {MODEL_DIM}")
        print(f"n_layers:         {N_LAYERS}")

        mlflow.log_metrics({
            "wf_mean_net_sharpe": mean_ns,
            "wf_std_net_sharpe": std_ns,
            "wf_min_net_sharpe": min_ns,
            "wf_production_ready": 1.0 if production_ready else 0.0,
            "total_seconds": t_end - t_start,
        })
        mlflow.log_param("status", "completed")
        print("MLflow run:", mlflow.active_run().info.run_id)
