"""The network, and the target it is asked to predict.

WHY LOG SPACE. Demand density spans four orders of magnitude — 0.1 trips/km²/day
in outer Queens against 16,000 in Times Square. Trained on the raw number, a
regression spends all its capacity on the handful of enormous zones and learns
nothing about the other 250; the loss barely moves whether it gets a quiet
residential zone right or wrong. Trained on log10, an error is proportional,
which is also how anyone actually reasons about demand: "twice as busy", not
"4,000 trips more".

WHY SO SMALL. 260 training zones is a tiny dataset by any vision standard. A
ResNet would memorise it before the second epoch. This is four convolutional
blocks, ~200k parameters, heavy augmentation, and it still needs watching for
overfit — the honest constraint of a problem where the label comes from a city
and there are only so many cities.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CHIP_PX = 128


def to_target(density: float) -> float:
    """Trips/km²/day -> log10, floored so empty zones do not become -inf."""
    return float(np.log10(max(density, 0.1)))


def from_target(y: float) -> float:
    return float(10.0 ** y)


class DemandNet(nn.Module):
    """Satellite chip -> log10 demand density.

    Deliberately plain: stride-2 convolutions, batch norm, global average pool.
    No pretrained backbone, because the interesting question is what *this*
    imagery carries, and an ImageNet backbone would blur that with whatever it
    already knows about photographs of streets.
    """

    def __init__(self, in_ch: int = 3, width: int = 32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_ch, width),        # 128 -> 64
            block(width, width * 2),    # 64  -> 32
            block(width * 2, width * 4),  # 32 -> 16
            block(width * 4, width * 4),  # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(width * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def normalise(chips: np.ndarray) -> np.ndarray:
    """uint8 CHW -> float32 in [-1, 1]."""
    return (chips.astype(np.float32) / 127.5) - 1.0


def augment(batch: torch.Tensor) -> torch.Tensor:
    """Flips and 90° rotations only.

    A satellite chip has no canonical orientation — north is a convention, not a
    feature — so the eight-fold dihedral group is free, label-preserving data.
    Colour jitter is deliberately left out: brightness here is atmosphere and
    sun angle, and teaching the model to ignore it would throw away the signal
    that separates a bright bare roof from dark asphalt.
    """
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[-1])
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        batch = torch.rot90(batch, k, dims=[-2, -1])
    return batch


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Scored in log space, where the model works and the reasoning happens."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.abs(y_true - y_pred).mean())

    # Spearman without scipy: rank correlation says whether the *ordering* of
    # zones survives, which is what a market-entry decision actually uses.
    def ranks(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    rt, rp = ranks(y_true), ranks(y_pred)
    rho = float(np.corrcoef(rt, rp)[0, 1]) if len(y_true) > 2 else float("nan")

    return {"r2": r2, "mae_log10": mae, "spearman": rho,
            "median_ratio_error": float(10 ** np.median(np.abs(y_true - y_pred)))}
