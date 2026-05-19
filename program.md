# autoresearch-vn

Autonomous ML research for Vietnam stock market direction prediction.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr24`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context (if exists).
   - `prepare.py` — fixed: data download, features, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Run `uv run prepare.py` to download and cache VN-Index data. Run `uv run prepare.py VN30` to download all VN30 stocks.
5. **Initialize results.tsv**: Create with header row and baseline entry.
6. **Confirm and go**.

## The Task

Predict the **next-day direction** (up/down) of Vietnamese stocks from 60 days of historical features. The primary metric is **val_accuracy** (directional accuracy on walk-forward validation set). Secondary metric is **val_sharpe** (annualized Sharpe ratio of long/short strategy).

**Targets:**
- **VN-Index**: `uv run train.py VNINDEX` (default)
- **Individual stocks**: `uv run train.py VNM`, `uv run train.py FPT`, etc.

**Baseline**: A random/majority classifier gets ~50-52% accuracy. Anything consistently above 53% is meaningful. Above 55% is excellent.

## Experimentation

Each experiment runs for a **fixed time budget of 5 minutes**. Launch: `uv run train.py [TICKER]`.

**What you CAN do:**
- Modify `train.py` — model architecture, optimizer, hyperparameters, training loop, features selection, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages beyond what's in `pyproject.toml`.
- Modify the evaluation function.

**The goal: maximize val_accuracy.** Since training time is fixed at 5 minutes, everything is fair game: architecture, optimizer, hyperparameters, batch size, model size, regularization.

**val_sharpe** is a secondary goal. A model with 54% accuracy and Sharpe 1.5 is better than 55% accuracy and Sharpe 0.2.

## Output format

The script prints a summary:

```
---
ticker:           VNINDEX
val_accuracy:     0.534500      # from single best model (backward compat)
val_sharpe:       0.85          # from single best model (backward compat)
val_loss:         0.692100
val_samples:      800
training_seconds: 370.0
total_seconds:    400.0
num_steps:        12500
num_params_K:     150.3
model_dim:        192
n_layers:         3
ensemble_size:    5             # new (default mode only)
ens_accuracy:     0.5912        # new
ens_sel_accuracy: 0.6820        # new
ens_sharpe:       0.6800        # new
ens_net_sharpe:   0.4200        # new
ens_coverage:     0.2800        # new
ens_max_drawdown: 0.0850        # new
ens_high_risk:    False         # new
ens_n_trades:     148           # new
```

Extract key metrics (backward compatible):
```
grep "^val_accuracy:\|^val_sharpe:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated):

```
ticker	commit	val_accuracy	val_sharpe	status	description
```

1. ticker (e.g. VNINDEX, VNM)
2. git commit hash (short, 7 chars)
3. val_accuracy (e.g. 0.534500) — use 0.000000 for crashes
4. val_sharpe (e.g. 0.85) — use 0.00 for crashes
5. status: `keep`, `discard`, or `crash`
6. short description

Example:
```
ticker	commit	val_accuracy	val_sharpe	status	description
VNINDEX	a1b2c3d	0.520000	0.10	keep	baseline transformer
VNM	b2c3d4e	0.545000	0.95	keep	LSTM with attention
FPT	c3d4e5f	0.510000	-0.30	discard	too much dropout
```

## The experiment loop

LOOP FOREVER:

1. Look at git state
2. Modify `train.py` with an experimental idea
3. git commit
4. Run: `uv run train.py [TICKER] > run.log 2>&1`
5. Read results: `grep "^val_accuracy:\|^val_sharpe:" run.log`
6. If grep empty -> crash. Run `tail -n 50 run.log` for traceback.
7. Record in TSV
8. If val_accuracy improved -> keep commit
9. If equal or worse -> git reset back

**NEVER STOP**: Continue indefinitely until manually stopped.

## Ensemble Pipeline (default mode)

The default `uv run train.py VNINDEX` runs a **5-seed ensemble** with confidence-based selective trading and cost-aware evaluation.

### How it works
- Train 5 independent GRU models (seeds: 42, 137, 256, 1024, 2049), each with ~74s budget
- At inference, average probabilities across all 5 models
- **Only trade** when mean probability > 0.60 (buy) or < 0.40 (sell) — otherwise hold
- Apply 0.3% transaction cost per trade, fractional Kelly position sizing
- Report metrics both for the raw model and the cost-aware ensemble

### New output fields
```
ensemble_size:    5
ens_accuracy:     0.591200   # ensemble mean-probability accuracy
ens_sel_accuracy: 0.682000   # accuracy on days with high-confidence signals only
ens_sharpe:       0.6800     # long/short Sharpe (no costs)
ens_net_sharpe:   0.4200     # Sharpe after transaction costs
ens_coverage:     0.2800     # fraction of days traded (high-confidence signal)
ens_max_drawdown: 0.0850     # peak-to-trough drawdown of strategy
ens_high_risk:    False      # True if max_drawdown > 15%
ens_n_trades:     148        # number of active trading days
```

### Keep criteria (updated)
- **keep**: `val_sharpe > 0.3` AND `ens_net_sharpe > 0.2` AND `ens_coverage > 0.15`
- **discard**: anything else
- **crash**: script failed to complete

## Walk-Forward Validation Mode

For rigorous out-of-sample testing, use `WALK_FORWARD=1`:

```bash
WALK_FORWARD=1 uv run train.py VNINDEX > run.log 2>&1
```

### How it works
- Dynamically builds year-boundary windows (train through Y-1, validate on Y)
- Trains 3-seed ensembles for each window
- Per-model budget: `max(20s, (TIME_BUDGET - 30s) / (n_windows × 3))`
- Re-normalizes features per window (using only that window's training data)

### Output
Prints per-window metrics table and aggregate summary:
- Mean/std/min of `net_sharpe` across all windows
- `PRODUCTION READY: YES` only if `min(net_sharpe) > 0.5` across ALL windows

### When to use
- After finding a good config in default ensemble mode
- Before considering real capital deployment
- Run walk-forward at the end of each experiment batch

## Architecture ideas to explore

- LSTM / GRU (classic for time series)
- Temporal CNN (1D convolutions)
- Transformer variants (attention patterns, pooling strategies)
- Feature selection (which features matter most?)
- Ensemble methods (average predictions from multiple heads)
- Different sequence lengths (override SEQ_LEN via model's internal windowing)
- Regularization: dropout, weight decay, early stopping
- Data augmentation: add noise to features, time warping
- Multi-task: predict return magnitude alongside direction
- Different optimizers: Adam, AdamW, SGD with momentum
- Cross-stock features (VN-Index as feature for individual stocks)
