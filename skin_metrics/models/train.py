"""Phase 2 training loop.

Supports two label regimes:

- ``mode="regression"``: Huber (Smooth-L1) loss against instrument labels,
  balanced across tasks by homoscedastic uncertainty weighting.
- ``mode="ranking"``: pairwise margin ranking loss ("A drier than B") for when
  only relative comparisons -- not absolute values -- are available.

A domain-adversarial illumination classifier (via gradient reversal) is trained
jointly to encourage illumination-invariant features. Validation reports MAE,
Pearson r, and Spearman rho, **broken down by Fitzpatrick type**.

Requires the ``dl`` extra (``uv sync --extra dl``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import TARGET_NAMES


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    from scipy.stats import rankdata

    return _pearson(rankdata(x), rankdata(y))


def _pairwise_ranking_loss(preds: "Any", targets: "Any", margin: float = 1.0):
    """Margin ranking loss over all valid ordered pairs in a batch (per task)."""
    import torch
    import torch.nn.functional as F

    total = preds.new_zeros(())
    n_tasks = preds.shape[1]
    for t in range(n_tasks):
        p = preds[:, t]
        y = targets[:, t]
        valid = ~torch.isnan(y)
        if valid.sum() < 2:
            continue
        p, y = p[valid], y[valid]
        # All i<j pairs.
        i, j = torch.triu_indices(p.numel(), p.numel(), offset=1)
        sign = torch.sign(y[i] - y[j])
        keep = sign != 0
        if keep.sum() == 0:
            continue
        total = total + F.margin_ranking_loss(
            p[i][keep], p[j][keep], sign[keep], margin=margin
        )
    return total


def _run_epoch(model, loader, optimizer, mode, grl_lambda, device, train: bool):
    import torch
    import torch.nn.functional as F

    from .network import uncertainty_weighted_loss

    model.train(train)
    torch.set_grad_enabled(train)

    agg_pred: list[np.ndarray] = []
    agg_true: list[np.ndarray] = []
    agg_fitz: list[np.ndarray] = []
    running = 0.0
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        physics = batch["physics"].to(device)
        targets = batch["targets"].to(device)
        illumination = batch["illumination"].to(device)

        out = model(image, physics, grl_lambda=grl_lambda)
        preds = out["predictions"]

        # --- task loss ---
        if mode == "ranking":
            task_loss = _pairwise_ranking_loss(preds, targets)
            reg_total = task_loss
        else:
            per_task = []
            for t in range(preds.shape[1]):
                y = targets[:, t]
                valid = ~torch.isnan(y)
                if valid.sum() == 0:
                    per_task.append(preds.new_zeros(()))
                else:
                    per_task.append(F.smooth_l1_loss(preds[valid, t], y[valid]))
            reg_total = uncertainty_weighted_loss(torch.stack(per_task), model.log_vars)

        # --- domain-adversarial loss ---
        domain_loss = F.cross_entropy(out["domain_logits"], illumination)
        loss = reg_total + domain_loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running += float(loss.detach().cpu())
        n_batches += 1
        agg_pred.append(preds.detach().cpu().numpy())
        agg_true.append(targets.detach().cpu().numpy())
        agg_fitz.append(batch["fitzpatrick"].cpu().numpy())

    torch.set_grad_enabled(True)
    return {
        "loss": running / max(n_batches, 1),
        "pred": np.concatenate(agg_pred) if agg_pred else np.zeros((0, len(TARGET_NAMES))),
        "true": np.concatenate(agg_true) if agg_true else np.zeros((0, len(TARGET_NAMES))),
        "fitz": np.concatenate(agg_fitz) if agg_fitz else np.zeros((0,), dtype=int),
    }


def _metrics_block(pred: np.ndarray, true: np.ndarray) -> dict[str, dict[str, float]]:
    """MAE / Pearson / Spearman per task, ignoring NaN targets."""
    out: dict[str, dict[str, float]] = {}
    for t, name in enumerate(TARGET_NAMES):
        p, y = pred[:, t], true[:, t]
        valid = ~np.isnan(y)
        p, y = p[valid], y[valid]
        if p.size == 0:
            out[name] = {"mae": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
            continue
        out[name] = {
            "mae": float(np.mean(np.abs(p - y))),
            "pearson": _pearson(p, y),
            "spearman": _spearman(p, y),
        }
    return out


def evaluate_report(pred: np.ndarray, true: np.ndarray, fitz: np.ndarray) -> dict[str, Any]:
    """Build an overall + per-Fitzpatrick metrics report."""
    report: dict[str, Any] = {"overall": _metrics_block(pred, true), "by_fitzpatrick": {}}
    for f in sorted(set(int(x) for x in fitz)):
        sel = fitz == f
        if sel.sum() == 0:
            continue
        report["by_fitzpatrick"][str(f)] = {
            "n": int(sel.sum()),
            "metrics": _metrics_block(pred[sel], true[sel]),
        }
    return report


def run_training(
    data_csv: str | Path | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "regression",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    grl_lambda: float = 0.5,
    use_dummy: bool = False,
    pretrained: bool = False,
    n_dummy: int = 128,
    image_size: int = 64,
    device: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train the Phase 2 model and return a metrics report.

    Parameters
    ----------
    data_csv : str or pathlib.Path, optional
        Instrument-label CSV. If ``None`` or ``use_dummy`` is ``True``, a
        synthetic dummy dataset is used so the loop runs with no labels.
    config : dict, optional
        Loaded configuration (currently unused by training but accepted for a
        uniform CLI signature).
    mode : {"regression", "ranking"}, optional
        Loss regime (see module docstring).
    epochs, batch_size, lr, grl_lambda : optional
        Standard training hyper-parameters; ``grl_lambda`` scales the
        domain-adversarial gradient reversal.
    use_dummy : bool, optional
        Force the dummy dataset.
    pretrained : bool, optional
        Load pretrained backbone weights (needs network access).
    n_dummy, image_size, seed : optional
        Dummy-dataset controls.
    device : str, optional
        Torch device string; auto-selected if ``None``.

    Returns
    -------
    dict
        ``{"train_losses": [...], "val_report": {...}, "mode": ...}``.
    """
    import torch
    from torch.utils.data import DataLoader

    from .dataset import SkinDataset
    from .network import SkinNet

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if data_csv is not None and not use_dummy:
        full = SkinDataset.from_csv(data_csv, train=True)
        n_val = max(1, len(full) // 5)
        train_ds = SkinDataset(full.samples[n_val:], train=True)
        val_ds = SkinDataset(full.samples[:n_val], train=False)
    else:
        train_ds = SkinDataset.dummy(n=n_dummy, image_size=image_size, train=True, seed=seed)
        val_ds = SkinDataset.dummy(n=max(32, n_dummy // 4), image_size=image_size, train=False, seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SkinNet(pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    for epoch in range(epochs):
        tr = _run_epoch(model, train_loader, optimizer, mode, grl_lambda, device, train=True)
        train_losses.append(tr["loss"])
        print(f"[epoch {epoch + 1}/{epochs}] mode={mode} train_loss={tr['loss']:.4f}")

    val = _run_epoch(model, val_loader, optimizer, mode, grl_lambda, device, train=False)
    report = evaluate_report(val["pred"], val["true"], val["fitz"])

    print("\n=== Validation report (per-Fitzpatrick) ===")
    for name, m in report["overall"].items():
        print(f"  [overall] {name:12s} MAE={m['mae']:.3f} r={m['pearson']:.3f} rho={m['spearman']:.3f}")
    for ftype, block in report["by_fitzpatrick"].items():
        print(f"  -- Fitzpatrick {ftype} (n={block['n']}) --")
        for name, m in block["metrics"].items():
            print(f"     {name:12s} MAE={m['mae']:.3f} r={m['pearson']:.3f} rho={m['spearman']:.3f}")

    return {"train_losses": train_losses, "val_report": report, "mode": mode}
