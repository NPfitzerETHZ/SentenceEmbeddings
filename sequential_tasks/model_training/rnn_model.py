"""Event RNN (v2) – hidden == predicted h
===========================================================
This revision unifies the GRU hidden state and the target vector **hₜ** so
`prev_h` *is literally the GRU hidden*.  Concretely, the GRU hidden size
is now **h_dim = 1024**, and the projection layer `next_state_head` has
been dropped.

Public interface (constructor arguments, loss breakdown, training loop)
remains the same, but the internal tensor shapes are simpler:

```
prev_h ∈ ℝ^{B×h_dim}
gru_hidden ≡ prev_h
```

So the autoregressive update is now just

```python
gru_hidden = rnn_cell(fused_t, gru_hidden)  # new hidden == new hₜ
next_h     = gru_hidden
```

The rest of the file is a complete, runnable script.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

###############################################################################
# Constants – unchanged
###############################################################################
MAX_SEQ_LEN = 10

###############################################################################
# Dataset + DataLoader helpers – identical to v1 (see comments there)
###############################################################################

def make_loaders(
    json_path: str,
    batch_size: int = 128,
    train_frac: float = 0.8,
    *,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    full_ds = EventSequenceDataset.from_json(json_path)
    n_total = len(full_ds)
    n_train = math.floor(n_total * train_frac)
    n_val = n_total - n_train

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, (n_train, n_val), generator=gen)

    kwargs = dict(
        batch_size=batch_size,
        collate_fn=EventSequenceDataset.collate,
        num_workers=0,
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=True, **kwargs),
    )


class EventSequenceDataset(Dataset):
    """Parses raw JSON into tensor dicts."""

    def __init__(self, raw: List[Dict]):
        self.samples: List[Dict[str, Tensor]] = []
        for ix, s in enumerate(raw):
            e_key = "events" if "events" in s else "e"
            if e_key not in s or not all(k in s for k in ("y", "h", "success")):
                raise KeyError(f"Sample {ix} missing required keys")

            e = torch.tensor(s[e_key], dtype=torch.float32)
            h = torch.tensor(s["h"], dtype=torch.float32)
            y = torch.tensor(s["y"], dtype=torch.float32)
            if y.ndim != 1:
                raise ValueError("'y' must be 1-D vector")
            label = torch.tensor(float(s["success"]), dtype=torch.float32)

            self.samples.append({"e": e, "y": y, "h": h, "label": label})

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "EventSequenceDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError("JSON must be an array of samples")
        return cls(raw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    @staticmethod
    def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        lengths = torch.tensor([s["e"].size(0) for s in batch])
        max_len = MAX_SEQ_LEN
        event_dim = batch[0]["e"].size(1)
        h_dim = batch[0]["h"].size(1)

        e_out, h_out, y_out, labels = [], [], [], []
        for s in batch:
            T = s["e"].size(0)
            pad_e = torch.zeros((max_len - T, event_dim))
            pad_h = torch.zeros((max_len - T, h_dim))

            e_out.append(torch.cat([s["e"], pad_e]))
            h_out.append(torch.cat([s["h"], pad_h]))
            y_rep = s["y"].unsqueeze(0).expand(max_len, -1)
            y_out.append(y_rep)
            labels.append(s["label"])

        return {
            "e": torch.stack(e_out),
            "h": torch.stack(h_out),
            "y": torch.stack(y_out),
            "lengths": lengths,
            "label": torch.stack(labels),
        }

###############################################################################
# EventRNN – hidden == h
###############################################################################

class EventRNN(pl.LightningModule):
    """RNN variant where the GRU hidden state *is* the predicted hₜ."""

    def __init__(
        self,
        *,
        event_dim: int,
        y_dim: int,
        h_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-4,
        cls_loss_weight: float = 1.0,
        recon_loss: str = "mse",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        D = h_dim  # hidden size now equals target size

        # ─── Embeddings ──────────────────────────────────────────────
        self.e_proj = nn.Linear(event_dim, D)
        self.y_proj = nn.Linear(y_dim, D) if y_dim != D else nn.Identity()
        self.h_proj = nn.Identity()  # no change in dimensionality
        self.fuse = nn.Linear(3 * D, D)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ─── GRU stack ───────────────────────────────────────────────
        if num_layers == 1:
            self.rnn = nn.GRUCell(D, D)
        else:
            self.rnn = nn.GRU(D, D, num_layers=num_layers, batch_first=True)
            # we will still step manually to keep parity with TBPTT logic

        # ─── Classification head ────────────────────────────────────
        hidden_dim = max(D // 2, 4)
        self.cls_head = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.recon_loss = recon_loss.lower()
        if self.recon_loss not in {"mse", "cosine"}:
            raise ValueError("recon_loss must be 'mse' or 'cosine'")
        self.bce = nn.BCEWithLogitsLoss()
        self.cls_w = cls_loss_weight
        self.lr = lr

    # ------------------------------------------------------------------
    def _prepare_step(self, e_t: Tensor, y_t: Tensor, h_prev: Tensor) -> Tensor:
        emb_e = self.e_proj(e_t)
        emb_y = self.y_proj(y_t)
        fused = self.fuse(torch.cat([emb_e, emb_y, h_prev], dim=-1))
        return self.dropout(fused)

    # ------------------------------------------------------------------
    def _rollout(self, e: Tensor, y: Tensor) -> Tensor:
        B, T, _ = e.shape
        D = self.hparams.h_dim

        preds: List[Tensor] = []
        h = e.new_zeros(B, D)  # h₀ = 0

        for t in range(T):
            fused = self._prepare_step(e[:, t], y[:, t], h.detach())
            h = self.rnn(fused, h) if isinstance(self.rnn, nn.GRUCell) else self.rnn(fused.unsqueeze(1), h.unsqueeze(0))[0].squeeze(1)
            preds.append(h)

        return torch.stack(preds, dim=1)  # (B,T,D) now D=h_dim

    # ------------------------------------------------------------------
    def _step(self, batch):
        e, y, target_h = batch["e"], batch["y"], batch["h"]
        lengths = batch["lengths"]

        pred_h = self._rollout(e, y)
        tgt_h = target_h[:, :MAX_SEQ_LEN]

        if self.recon_loss == "mse":
            loss_recon = F.mse_loss(pred_h, tgt_h)
        else:
            cos = F.cosine_similarity(pred_h, tgt_h, dim=-1)
            loss_recon = (1 - cos).mean()

        # classification on the *true* last timestep
        label = batch["label"]
        last_hidden = pred_h.gather(
            1,
            (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, pred_h.size(-1)),
        ).squeeze(1)
        pred_cls = self.cls_head(last_hidden)
        loss_cls = self.bce(pred_cls, label.unsqueeze(1))

        loss = loss_recon + self.cls_w * loss_cls
        return loss, loss_recon, loss_cls

    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, rec, cls = self._step(batch)
        self.log_dict({"train_loss": loss, "train_recon": rec, "train_cls": cls}, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, rec, cls = self._step(batch)
        self.log_dict({"val_loss": loss, "val_recon": rec, "val_cls": cls}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

###############################################################################
# CLI entrypoint
###############################################################################
if __name__ == "__main__":
    train_loader, val_loader = make_loaders(
        "sequential_tasks/data/dataset.json", batch_size=64
    )

    model = EventRNN(event_dim=3, y_dim=1024, h_dim=1024)

    trainer = pl.Trainer(max_epochs=100, accelerator="auto", log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
