"""ConvLSTM on sequences of per-channel features, cross-validated like the other models.

    .venv/bin/python -m modeling.run_convlstm                                       # all folds, both sets
    .venv/bin/python -m modeling.run_convlstm --folds 0 --sets set2 --max-epochs 5 --out-dir results/test

Same folds (results/folds.csv), training mask and output format as
modeling/run_experiments.py: rows with model="ConvLSTM" in cv_auroc.csv and
predictions_ConvLSTM_set<N>.parquet. Also writes convlstm_history.csv (loss and
validation AUROC per epoch) and convlstm_curves.png.

Input. Needs data/processed/channel_features.npy from
`scripts/build_feature_table.py --channel-features`: Leila's per-channel features,
(n_windows, 18 bipolar channels, 58 features). Set 1 uses the first 55 (Temko),
set 2 all 58. Each target window becomes a sequence of SEQ_PAST earlier windows of
the same subject plus the target itself (6 s apart), oldest first. Only past
windows are used, and a sequence never crosses subjects.

Missing context (documented rule):
  * Before the first window of a recording there is nothing: those steps are
    padding, all zeros with mask 0.
  * A channel without features (recording gap, or QC-bad per §6) is set to 0,
    which after the rank transform below is the training-typical value, and its
    mask input is 0. Real channels have mask 1. Ineligible windows therefore
    appear as context but carry mask 0 on every gap channel.
  * Ambiguous (label -1) windows have real features and are used as context like
    any other window; labels are never model inputs.
  Targets for training and validation are only training_mask windows. Every
  window of the held-out patients is predicted.

Scaling. Per feature, a QuantileTransformer (rank -> normal distribution) is fit on
the valid channel-windows of the training patients only, then applied to every
window. Many features (powers, energies) span orders of magnitude; a rank
transform makes them comparable without being dominated by outliers.

Training. Weighted BCE (pos_weight = negatives/positives among training targets);
Adam; early stopping on the AUROC of a validation split of the TRAINING patients
(VAL_FRACTION of the seizure and of the non-seizure subjects), never the test fold.

PyTorch note: the newest PyTorch for Intel Macs (2.2.2) was built for numpy 1.x and
cannot exchange arrays with numpy 2 directly, so arrays go through raw bytes
(to_torch) and results come back as lists (to_numpy).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

# PyTorch 2.2.2 prints a NumPy 1.x/2.x notice on import; to_torch / to_numpy avoid the affected interop.
with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
    warnings.simplefilter("ignore", UserWarning)
    import torch
    from torch import nn

from modeling.common import DATA_ROOT, N_SPLITS, RESULTS_DIR, add_folds, load_feature_table, training_mask
from modeling.run_experiments import PREDICTION_COLUMNS, evaluate, save_cv_rows

PROCESSED = DATA_ROOT / "processed"
N_TEMKO = 55
SET_FEATURES = {"set1": slice(0, N_TEMKO), "set2": slice(0, None)}

SEQ_PAST = 5            # previous windows per sequence (30 s of history at a 6 s stride)
HIDDEN = 16             # ConvLSTM hidden channels per EEG channel
KERNEL = 3              # convolution across neighbouring bipolar channels
DROPOUT = 0.3
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5            # epochs without validation AUROC improvement before stopping
VAL_FRACTION = 0.2
RANDOM_STATE = 0


def to_torch(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    np_dtype = {torch.float32: np.float32, torch.int64: np.int64}[dtype]
    a = np.ascontiguousarray(a, dtype=np_dtype)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=dtype).reshape(a.shape)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return np.asarray(t.detach().tolist())


# ---- data -----------------------------------------------------------------------

def load_data():
    """Window table (with folds) and row-aligned per-channel features (NaN where a channel is invalid)."""
    df = add_folds(load_feature_table())
    index = pd.read_parquet(PROCESSED / "channel_index.parquet")
    if not (index.subject_id.to_numpy() == df.subject_id.to_numpy()).all() or \
            not (index.window_index.to_numpy() == df.window_index.to_numpy()).all():
        raise ValueError("channel_features.npy is not aligned with feature_table.parquet; rebuild both")
    order = df.sort_values(["subject_id", "window_index"]).index.to_numpy()
    if not (order == np.arange(len(df))).all():
        raise ValueError("feature table must be sorted by subject_id, window_index")
    feats = np.load(PROCESSED / "channel_features.npy")
    valid = np.load(PROCESSED / "channel_valid.npy")
    feats[~valid] = np.nan
    return df, feats, valid


def sequence_index(df: pd.DataFrame) -> np.ndarray:
    """(n_windows, SEQ_PAST + 1) row indices, oldest first, target last; len(df) marks padding.

    Step k of row i is row i - (SEQ_PAST - k) if it is the same subject's window
    exactly that many windows earlier, else padding. Never looks forward, never crosses subjects.
    """
    n = len(df)
    subj, widx = df.subject_id.to_numpy(), df.window_index.to_numpy()
    rows = np.arange(n)
    seq = np.full((n, SEQ_PAST + 1), n)
    for k in range(SEQ_PAST + 1):
        back = SEQ_PAST - k
        j = rows - back
        ok = j >= 0
        ok[ok] &= (subj[j[ok]] == subj[ok]) & (widx[j[ok]] == widx[ok] - back)
        seq[ok, k] = j[ok]
    return seq


def scaled_inputs(feats: np.ndarray, valid: np.ndarray, fit_rows: np.ndarray) -> torch.Tensor:
    """(n_windows + 1, 18, n_features + 1) model inputs; last row is the padding row of zeros.

    Rank transform fit on valid channels of `fit_rows` only; missing -> 0; last input = channel mask.
    """
    n, n_ch, n_f = feats.shape
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                             random_state=RANDOM_STATE)
    qt.fit(feats[fit_rows][valid[fit_rows]])
    x = qt.transform(feats.reshape(-1, n_f)).reshape(n, n_ch, n_f)
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    out = np.zeros((n + 1, n_ch, n_f + 1), np.float32)
    out[:n, :, :n_f] = x
    out[:n, :, n_f] = valid
    return to_torch(out)


# ---- model ----------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """LSTM whose gates are 1-D convolutions across the 18 bipolar channels."""

    def __init__(self, n_in: int, n_hidden: int, kernel: int):
        super().__init__()
        self.n_hidden = n_hidden
        self.gates = nn.Conv1d(n_in + n_hidden, 4 * n_hidden, kernel, padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """One ConvLSTM layer over the sequence, then mean and max over channels -> one logit."""

    def __init__(self, n_in: int):
        super().__init__()
        self.cell = ConvLSTMCell(n_in, HIDDEN, KERNEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(2 * HIDDEN, 1)

    def forward(self, x):                       # x: (batch, time, channels, inputs)
        b, t, n_ch, _ = x.shape
        h = x.new_zeros(b, HIDDEN, n_ch)
        c = x.new_zeros(b, HIDDEN, n_ch)
        for step in range(t):
            h, c = self.cell(x[:, step].transpose(1, 2), (h, c))
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(self.dropout(pooled)).squeeze(1)


# ---- training -------------------------------------------------------------------

def split_validation(df: pd.DataFrame, train_subjects: np.ndarray, seed: int) -> np.ndarray:
    """VAL_FRACTION of the seizure and of the non-seizure training subjects (deterministic)."""
    has_seizure = df[df.label == 1].subject_id.unique()
    rng = np.random.default_rng(seed)
    val = []
    for group in (np.intersect1d(train_subjects, has_seizure), np.setdiff1d(train_subjects, has_seizure)):
        val.extend(rng.choice(group, max(1, round(VAL_FRACTION * len(group))), replace=False))
    return np.sort(val)


def predict(model, x_all, seq, rows, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            idx = to_torch(seq[rows[i:i + batch]], torch.int64)
            out.append(torch.sigmoid(model(x_all[idx])))
    return to_numpy(torch.cat(out)) if out else np.empty(0)


def run_fold(df, feats, valid, seq, set_name: str, fold: int, max_epochs: int):
    """Train on the other folds' patients, return (held-out predictions, history rows, fit seconds)."""
    torch.manual_seed(RANDOM_STATE + fold)
    f = feats[:, :, SET_FEATURES[set_name]]
    train_ok = training_mask(df).to_numpy()
    test_rows = np.flatnonzero(df.fold.to_numpy() == fold)
    train_subjects = np.unique(df.subject_id[df.fold != fold])
    val_subjects = split_validation(df, train_subjects, seed=RANDOM_STATE + fold)
    in_val = df.subject_id.isin(val_subjects).to_numpy()
    in_train = (df.fold != fold).to_numpy() & ~in_val
    tr_rows, va_rows = np.flatnonzero(in_train & train_ok), np.flatnonzero(in_val & train_ok)
    y = df.label.to_numpy()

    t0 = time.time()
    x_all = scaled_inputs(f, valid, tr_rows)
    y_t = to_torch(y.astype(np.float32))
    n_pos = int((y[tr_rows] == 1).sum())
    pos_weight = torch.tensor((len(tr_rows) - n_pos) / n_pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = ConvLSTM(x_all.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    seq_t = to_torch(seq, torch.int64)
    tr_t, va_t = to_torch(tr_rows, torch.int64), to_torch(va_rows, torch.int64)
    gen = torch.Generator().manual_seed(RANDOM_STATE + fold)
    print(f"  {set_name} fold {fold}: train {len(tr_rows)} targets ({n_pos} pos) from {len(np.unique(df.subject_id.to_numpy()[tr_rows]))} "
          f"patients | val {len(va_rows)} targets from {len(val_subjects)} patients | test {len(test_rows)} windows",
          flush=True)

    history, best_auc, best_state, stale = [], -np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for b in tr_t[torch.randperm(len(tr_t), generator=gen)].split(BATCH):
            logits = model(x_all[seq_t[b]])
            loss = loss_fn(logits, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            count += len(b)
        model.eval()
        with torch.no_grad():
            va_logits = torch.cat([model(x_all[seq_t[b]]) for b in va_t.split(2048)])
            va_loss = loss_fn(va_logits, y_t[va_t]).item()
        va_auc = roc_auc_score(y[va_rows], to_numpy(va_logits))
        history.append({"feature_set": set_name, "fold": fold, "epoch": epoch, "train_loss": total / count,
                        "val_loss": va_loss, "val_auroc": va_auc})
        print(f"    epoch {epoch:2d}: train loss {total / count:.4f}  val loss {va_loss:.4f}  val AUROC {va_auc:.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
        if va_auc > best_auc:
            best_auc, stale = va_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    t_fit = time.time() - t0

    held_out = df.iloc[test_rows].assign(score=predict(model, x_all, seq, test_rows))
    return held_out, history, t_fit


def save_history(history: list[dict], out_dir) -> pd.DataFrame:
    path = out_dir / "convlstm_history.csv"
    new = pd.DataFrame(history)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old.set_index(["feature_set", "fold"]).index.isin(new.set_index(["feature_set", "fold"]).index)]
        new = pd.concat([old, new])
    new = new.sort_values(["feature_set", "fold", "epoch"])
    new.to_csv(path, index=False)
    return new


def plot_history(history: pd.DataFrame, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11, 4))
    for (set_name, fold), h in history.groupby(["feature_set", "fold"]):
        line, = ax_loss.plot(h.epoch, h.train_loss, label=f"{set_name} fold {fold} train")
        ax_loss.plot(h.epoch, h.val_loss, "--", color=line.get_color(), label=f"{set_name} fold {fold} val")
        ax_auc.plot(h.epoch, h.val_auroc, color=line.get_color(), label=f"{set_name} fold {fold}")
    ax_loss.set(xlabel="epoch", ylabel="weighted BCE loss", title="Loss (solid: train, dashed: validation)")
    ax_auc.set(xlabel="epoch", ylabel="AUROC", title="Validation AUROC (training patients held out)")
    for ax in (ax_loss, ax_auc):
        ax.grid(alpha=0.3)
    if history.groupby(["feature_set", "fold"]).ngroups <= 4:
        ax_loss.legend(fontsize=7)
        ax_auc.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sets", nargs="+", default=list(SET_FEATURES), choices=list(SET_FEATURES))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_SPLITS)))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df, feats, valid = load_data()
    seq = sequence_index(df)
    print(f"{len(df)} windows, {df.subject_id.nunique()} subjects | sequences of {SEQ_PAST + 1} windows "
          f"| {int((seq[:, 0] == len(df)).sum())} windows have padded history\n", flush=True)

    for set_name in args.sets:
        rows, predictions, history = [], [], []
        t_set = time.time()
        for fold in args.folds:
            t0 = time.time()
            held_out, hist, t_fit = run_fold(df, feats, valid, seq, set_name, fold, args.max_epochs)
            t_pred = time.time() - t0 - t_fit
            row = evaluate(held_out, set_name, "ConvLSTM", fold)
            rows.append({**row, "fit_s": round(t_fit, 1), "predict_s": round(t_pred, 1)})
            predictions.append(held_out[PREDICTION_COLUMNS])
            history.extend(hist)
            best = max(hist, key=lambda h: h["val_auroc"])
            print(f"ConvLSTM {set_name} fold {fold}: AUROC {row['auroc']:.3f}  AUPRC {row['auprc']:.3f}  "
                  f"(best epoch {best['epoch']}, val AUROC {best['val_auroc']:.3f})  "
                  f"fold time {(time.time() - t0) / 60:.1f} min\n", flush=True)
        pd.concat(predictions).sort_values(["subject_id", "start_s"]).to_parquet(
            args.out_dir / f"predictions_ConvLSTM_{set_name}.parquet", index=False)
        save_cv_rows(rows, args.out_dir)
        plot_history(save_history(history, args.out_dir), args.out_dir / "convlstm_curves.png")
        print(f"ConvLSTM {set_name} done in {(time.time() - t_set) / 60:.1f} min\n", flush=True)


if __name__ == "__main__":
    main()
