"""Multi-branch neural network for Phase 2.

    static branch:  embeddings (carrier, origin, dest_grouped[, flight])
                    + standardised Phase 1 numeric features -> dense
    origin branch:  sequence encoder over the origin history window
    route branch:   sequence encoder over the route history window
    head:           concat -> dense + dropout -> logit

Sequence encoders (Stage 4): GRU, 1D CNN, or a flatten + dense control.

Any history branch can be switched off (Stage 1). Training uses binary
cross-entropy, AdamW (weight decay) and early stopping on validation AP
(rule R10), restoring the best epoch.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import P1_NUMERIC
from .metrics import validation_metrics

MIN_CATEGORY_FREQ = 20  # same rare-category rule as the Phase 1 one-hot encoder


# ----------------------------------------------------------------- inputs
@dataclass
class StaticEncoder:
    """Category vocabularies and numeric scaling, fitted on train only."""
    categorical: list[str]
    numeric: list[str] = field(default_factory=lambda: list(P1_NUMERIC))
    vocab: dict = field(default_factory=dict)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "StaticEncoder":
        for c in self.categorical:
            counts = train[c].value_counts()
            keep = counts[counts >= MIN_CATEGORY_FREQ].index
            self.vocab[c] = {v: i + 1 for i, v in enumerate(sorted(keep))}  # 0 = rare/unseen
        x = train[self.numeric].to_numpy(dtype=np.float64)
        self.mean, self.std = x.mean(0), x.std(0)
        self.std[self.std == 0] = 1.0
        return self

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.categorical]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat = np.stack([df[c].map(self.vocab[c]).fillna(0).astype(np.int64).to_numpy()
                        for c in self.categorical], axis=1)
        num = ((df[self.numeric].to_numpy(dtype=np.float64) - self.mean) / self.std).astype(np.float32)
        return cat, num


# ----------------------------------------------------------------- model
class GRUEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


class CNNEncoder(nn.Module):
    """1D CNN over days: two conv layers, then a readout.

    pool="meanmax": mean- and max-pool over days (forgets which day a
    pattern came from). pool="last": take the output at the last day (D-1),
    whose receptive field covers the most recent days in order.
    """
    def __init__(self, n_in: int, hidden: int, kernel: int = 3, pool: str = "meanmax"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, hidden, kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.ReLU(),
        )
        self.pool = pool
        self.out_dim = 2 * hidden if pool == "meanmax" else hidden

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        if self.pool == "last":
            return h[..., -1]
        return torch.cat([h.mean(-1), h.amax(-1)], dim=-1)


class CNNLastEncoder(CNNEncoder):
    def __init__(self, n_in: int, hidden: int):
        super().__init__(n_in, hidden, pool="last")


class FlatEncoder(nn.Module):
    """Control with no notion of order over days: flatten the window and
    apply one dense layer. Needs a fixed window length."""
    def __init__(self, n_in: int, hidden: int, window: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_in * window, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


ENCODERS = {"gru": GRUEncoder, "cnn": CNNEncoder, "flat": FlatEncoder,
            "cnn_last": CNNLastEncoder}


class FlightNet(nn.Module):
    def __init__(self, cards: list[int], n_num: int, seq_dims: dict[str, int],
                 encoder: str = "gru", hidden: int = 32, static_hidden: int = 64,
                 head_hidden: int = 64, dropout: float = 0.2, window: int | None = None,
                 one_hot: bool = False):
        super().__init__()
        if one_hot:  # Stage 5 control: fixed identity vectors instead of learned embeddings
            self.embs = nn.ModuleList(nn.Embedding.from_pretrained(torch.eye(c), freeze=True)
                                      for c in cards)
        else:
            self.embs = nn.ModuleList(nn.Embedding(c, min(16, (c + 1) // 2)) for c in cards)
        emb_dim = sum(e.embedding_dim for e in self.embs)
        self.static = nn.Sequential(nn.Linear(emb_dim + n_num, static_hidden), nn.ReLU(),
                                    nn.Dropout(dropout))
        extra = {"window": window} if encoder == "flat" else {}
        self.encoders = nn.ModuleDict({k: ENCODERS[encoder](d, hidden, **extra)
                                       for k, d in seq_dims.items()})
        fused = static_hidden + sum(e.out_dim for e in self.encoders.values())
        self.head = nn.Sequential(nn.Linear(fused, head_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(head_hidden, 1)

    def representation(self, cat, num, seqs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Penultimate-layer features (used by the Stage 7 hybrid)."""
        e = torch.cat([emb(cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        parts = [self.static(torch.cat([e, num], dim=1))]
        parts += [enc(seqs[k]) for k, enc in self.encoders.items()]
        return self.head(torch.cat(parts, dim=1))

    def forward(self, cat, num, seqs):
        return self.out(self.representation(cat, num, seqs)).squeeze(-1)


# ----------------------------------------------------------------- training
@dataclass
class Batchable:
    cat: torch.Tensor
    num: torch.Tensor
    seqs: dict[str, torch.Tensor]
    y: torch.Tensor | None = None

    def __len__(self):
        return len(self.cat)

    def take(self, idx):
        return (self.cat[idx], self.num[idx], {k: v[idx] for k, v in self.seqs.items()})


def make_batchable(cat, num, seqs: dict[str, np.ndarray], y=None) -> Batchable:
    return Batchable(torch.from_numpy(cat), torch.from_numpy(num),
                     {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in seqs.items()},
                     None if y is None else torch.from_numpy(np.asarray(y, dtype=np.float32)))


@torch.no_grad()
def predict(model: FlightNet, data: Batchable, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(*data.take(slice(i, i + batch_size))))
           for i in range(0, len(data), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def represent(model: FlightNet, data: Batchable, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = [model.representation(*data.take(slice(i, i + batch_size)))
           for i in range(0, len(data), batch_size)]
    return torch.cat(out).numpy()


def train_model(model: FlightNet, train: Batchable, val: Batchable, *, seed: int,
                lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 1024,
                max_epochs: int = 25, patience: int = 4, verbose: bool = True,
                early_stopping: bool = True) -> dict:
    """Train with AdamW + BCE. With early_stopping (default) training stops
    after `patience` epochs without a better validation AP and the best epoch
    is restored. Without it, all `max_epochs` run and the last epoch is kept
    (Stage 6 control)."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"val_ap": -1.0, "epoch": 0, "state": None}
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(train), generator=gen)
        total = 0.0
        for i in range(0, len(train), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(*train.take(idx)), train.y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        prob = predict(model, val)
        ap = validation_metrics(val.y.numpy(), prob)["val_ap"]
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_ap": ap})
        if verbose:
            print(f"    epoch {epoch:2d}  loss={total / len(train):.4f}  val_AP={ap:.4f}")
        if ap > best["val_ap"]:
            best = {"val_ap": ap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        elif early_stopping and epoch - best["epoch"] >= patience:
            break
    if early_stopping:
        model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"] if early_stopping else len(history), "epochs_run": len(history),
            "seconds": time.time() - t0, "history": history}


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
