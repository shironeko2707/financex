"""
Data preparation, feature engineering, and evaluation for Vietnam stock prediction.
This file is READ-ONLY — the agent must not modify it.

Usage:
  uv run prepare.py          # download data and cache features
  (imported by train.py)     # provides dataloader + evaluation

Supports two modes via TICKER:
  - "VNINDEX"  — predict VN-Index direction (default)
  - "VNM", "FPT", etc. — predict individual stock direction
"""

import os
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 400       # 5-minute training budget (wall clock)
SEQ_LEN = 60            # 60 trading days of history per sample
TRAIN_RATIO = 0.8       # 80% train, 20% validation (walk-forward)
TICKER = "VNINDEX"      # Change to stock symbol for individual stocks
DATA_START = "2016-01-01"
CACHE_DIR = Path.home() / ".cache" / "autoresearch-vn"

# VN30 components (updated periodically — these are representative)
VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
    "HDB", "HPG", "KDH", "MBB", "MSN", "MWG", "PLX", "POW",
    "SAB", "SHB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
    "VIC", "VJC", "VNM", "VPB", "VRE",
]

VNSTOCK_SOURCE = "VCI"  # Data source: "VCI" or "KBS"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from OHLCV data. All features use only past data."""
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns at multiple horizons
    for d in [1, 5, 10, 20]:
        feat[f"ret_{d}d"] = close.pct_change(d)

    # Log return (1-day)
    feat["log_ret_1d"] = np.log(close / close.shift(1))

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd"] = macd / close  # normalize by price
    feat["macd_signal"] = signal / close
    feat["macd_hist"] = (macd - signal) / close

    # Bollinger Bands (%B — position within bands)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_pct_b"] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)

    # Moving average ratios
    for w in [10, 20, 50]:
        sma = close.rolling(w).mean()
        feat[f"price_sma{w}_ratio"] = close / sma - 1

    # Volatility
    for w in [10, 20]:
        feat[f"volatility_{w}d"] = close.pct_change().rolling(w).std()

    # ATR (14-day, normalized by price)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / close

    # Volume features
    feat["volume_change_1d"] = volume.pct_change()
    feat["volume_sma10_ratio"] = volume / volume.rolling(10).mean() - 1

    # High-Low range (normalized)
    feat["hl_range"] = (high - low) / close

    # Day of week (cyclical encoding)
    dow = df.index.dayofweek
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 5)  # 5 trading days
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    return feat


def compute_target(df: pd.DataFrame) -> pd.Series:
    """Target: 1 if next-day return > 0, else 0."""
    next_ret = df["Close"].pct_change().shift(-1)  # next day's return
    return (next_ret > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Data download via vnstock
# ---------------------------------------------------------------------------

def _download_ticker(symbol: str, start: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker using vnstock."""
    from vnstock import Vnstock

    print(f"  Downloading {symbol}...")
    stock = Vnstock().stock(symbol=symbol, source=VNSTOCK_SOURCE)
    df = stock.quote.history(start=start, end=pd.Timestamp.now().strftime("%Y-%m-%d"), interval="1D")

    # Rename columns to standard format (vnstock uses lowercase)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("time", "date", "trading_date"):
            col_map[col] = "Date"
        elif cl == "open":
            col_map[col] = "Open"
        elif cl == "high":
            col_map[col] = "High"
        elif cl == "low":
            col_map[col] = "Low"
        elif cl == "close":
            col_map[col] = "Close"
        elif cl == "volume":
            col_map[col] = "Volume"

    df = df.rename(columns=col_map)

    # Set Date as index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # Keep only OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} for {symbol}. Available: {list(df.columns)}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Drop rows with zero volume or zero close (bad data)
    df = df[(df["Volume"] > 0) & (df["Close"] > 0)]

    print(f"  {symbol}: {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def download_and_cache(ticker: str = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Download VN stock data, compute features, cache result.

    Args:
        ticker: Override the default TICKER. Use "VNINDEX" for index,
                or a stock symbol like "VNM" for individual stocks.

    Returns:
        features: np.ndarray of shape (N, num_features), float32
        targets: np.ndarray of shape (N,), float32 (0 or 1)
        meta: dict with feature_names, dates, train_size, val_size
    """
    if ticker is None:
        ticker = TICKER

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker.lower()}_features.pkl"

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Downloading {ticker} data from {DATA_START}...")
    df = _download_ticker(ticker, DATA_START)
    print(f"Downloaded {len(df)} trading days")

    # Compute features and target
    features_df = compute_features(df)
    targets_series = compute_target(df)

    # Drop rows with NaN (from rolling windows)
    valid_mask = features_df.notna().all(axis=1) & targets_series.notna()
    features_df = features_df[valid_mask]
    targets_series = targets_series[valid_mask]

    feature_names = list(features_df.columns)
    dates = features_df.index.strftime("%Y-%m-%d").tolist()

    features = features_df.values.astype(np.float32)
    targets = targets_series.values.astype(np.float32)

    # Replace any remaining inf/nan with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    train_size = int(len(features) * TRAIN_RATIO)
    val_size = len(features) - train_size

    meta = {
        "ticker": ticker,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "dates": dates,
        "train_size": train_size,
        "val_size": val_size,
        "total_samples": len(features),
    }

    print(f"Features: {meta['num_features']}")
    print(f"Total samples: {meta['total_samples']}")
    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Feature names: {feature_names}")

    result = (features, targets, meta)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached to {cache_file}")

    return result


def download_vn30_data() -> dict:
    """Download data for all VN30 components. Returns dict of ticker -> (features, targets, meta)."""
    results = {}
    for ticker in VN30_TICKERS:
        try:
            results[ticker] = download_and_cache(ticker)
        except Exception as e:
            print(f"  WARNING: Failed to download {ticker}: {e}")
    return results


# ---------------------------------------------------------------------------
# Normalization (using train stats only — no leakage)
# ---------------------------------------------------------------------------

def normalize_features(features: np.ndarray, train_size: int) -> np.ndarray:
    """Z-score normalize using only training data statistics."""
    train_feats = features[:train_size]
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return ((features - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def make_sequences(features: np.ndarray, targets: np.ndarray,
                   start_idx: int, end_idx: int, seq_len: int = SEQ_LEN):
    """Create (sequence, target) pairs using a sliding window.

    Each sequence is features[i-seq_len:i], target is targets[i].
    """
    X, y = [], []
    for i in range(max(start_idx, seq_len), end_idx):
        X.append(features[i - seq_len:i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_dataloader(features: np.ndarray, targets: np.ndarray,
                    start_idx: int, end_idx: int,
                    batch_size: int, seq_len: int = SEQ_LEN, shuffle: bool = True):
    """Yields (X_batch, y_batch) as torch tensors. Loops forever."""
    import torch

    X, y = make_sequences(features, targets, start_idx, end_idx, seq_len)
    n = len(X)
    if n == 0:
        raise ValueError(f"No sequences available for range [{start_idx}, {end_idx})")

    print(f"Dataloader: {n} sequences, batch_size={batch_size}")
    indices = np.arange(n)
    epoch = 0

    while True:
        if shuffle:
            np.random.shuffle(indices)
        epoch += 1
        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = torch.from_numpy(X[batch_idx])
            y_batch = torch.from_numpy(y[batch_idx])
            yield X_batch, y_batch, epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, features: np.ndarray, targets: np.ndarray,
             train_size: int, device, seq_len: int = SEQ_LEN,
             batch_size: int = 256, use_tta: bool = False,
             num_tta: int = 5) -> dict:
    """Evaluate model on validation set. Returns metrics dict.

    Metrics:
        accuracy: directional accuracy (% correct up/down predictions)
        sharpe: annualized Sharpe ratio of a long/short strategy
        avg_loss: average binary cross-entropy loss on val set
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    X_val, y_val = make_sequences(features, targets, train_size, len(features), seq_len)

    all_probs = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(X_val), batch_size):
        X_batch = torch.from_numpy(X_val[start:start + batch_size]).to(device)
        y_batch = torch.from_numpy(y_val[start:start + batch_size]).to(device)

        if use_tta:
            model.train()
            probs_list = []
            with torch.no_grad():
                for _ in range(num_tta):
                    logits = model(X_batch).squeeze(-1)
                    probs = torch.sigmoid(logits)
                    probs_list.append(probs)
            probs_avg = torch.stack(probs_list).mean(dim=0)
            preds = (probs_avg > 0.5).float()
            loss = F.binary_cross_entropy_with_logits(torch.logit(probs_avg), y_batch)
            model.eval()
        else:
            with torch.no_grad():
                logits = model(X_batch).squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)

        total_loss += loss.item()
        n_batches += 1

        all_probs.append(probs.cpu().numpy() if use_tta else probs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_preds = (all_probs > 0.5).astype(float)

    # Directional accuracy
    accuracy = (all_preds == all_targets).mean()

    # Sharpe ratio: go long when predict up, short when predict down
    val_start = train_size
    val_returns = np.diff(features[val_start:val_start + len(all_preds) + 1, 0])  # ret_1d feature
    if len(val_returns) > len(all_preds):
        val_returns = val_returns[:len(all_preds)]
    elif len(val_returns) < len(all_preds):
        all_preds = all_preds[:len(val_returns)]

    positions = np.where(all_preds == 1, 1.0, -1.0)
    strategy_returns = positions * val_returns
    sharpe = 0.0
    if len(strategy_returns) > 1 and strategy_returns.std() > 1e-10:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    avg_loss = total_loss / max(n_batches, 1)

    return {
        "accuracy": float(accuracy),
        "sharpe": float(sharpe),
        "avg_loss": float(avg_loss),
        "n_val": len(all_preds),
    }


# ---------------------------------------------------------------------------
# Main: download and prepare data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else TICKER

    if ticker.upper() == "VN30":
        print("Downloading all VN30 components...")
        data = download_vn30_data()
        print(f"\nSuccessfully downloaded {len(data)}/{len(VN30_TICKERS)} stocks")
        for t, (f, tgt, m) in data.items():
            print(f"  {t}: {m['total_samples']} samples, pos_rate={tgt.mean():.3f}")
    else:
        features, targets, meta = download_and_cache(ticker)
        norm_features = normalize_features(features, meta["train_size"])

        print(f"\nData ready for {ticker}!")
        print(f"  Features shape: {norm_features.shape}")
        print(f"  Targets shape:  {targets.shape}")
        print(f"  Positive rate:  {targets.mean():.3f} (baseline accuracy = {max(targets.mean(), 1-targets.mean()):.3f})")
        print(f"  Train samples:  {meta['train_size']}")
        print(f"  Val samples:    {meta['val_size']}")
        print(f"\nCached at: {CACHE_DIR}")
