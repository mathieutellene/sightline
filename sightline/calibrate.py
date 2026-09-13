"""How much local ground truth does the prediction need to become useful?

The transfer result splits cleanly in two. Rank survives the move to a new city
almost untouched; absolute level does not, and it is off by a factor of two or
three. That is not a broken model, it is a model being asked something pixels
cannot answer: how enthusiastically a particular city has adopted ride-hailing.
New York runs about six times denser than Chicago at the median, and no amount
of looking at rooftops reveals that.

Which makes the operational question concrete. Entering a city, you can buy or
run a handful of zones' worth of real demand. How few is enough to fix the
level, given the shape is already right?

This fits a single scalar offset in log space -- one number, the city's overall
intensity -- on k randomly chosen zones and scores the rest. Repeated over many
draws, because which k zones you happen to get matters when k is small.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import metrics

RUN = Path(__file__).resolve().parents[1] / "data" / "runs" / "result.json"


def curve(trials: int = 400, seed: int = 0) -> list[dict]:
    r = json.loads(RUN.read_text())
    keys = sorted(r["chicago_true"])
    y = np.log10(np.array([r["chicago_true"][k] for k in keys]))
    p = np.log10(np.array([r["chicago_pred"][k] for k in keys]))
    rng = np.random.default_rng(seed)

    rows = [{"k": 0, **metrics(y, p)}]
    for k in (1, 2, 3, 5, 8, 13, 21):
        if k >= len(keys) - 5:
            break
        got = []
        for _ in range(trials):
            idx = rng.choice(len(keys), size=k, replace=False)
            rest = np.setdiff1d(np.arange(len(keys)), idx)
            # One scalar: the median gap between truth and prediction on the
            # calibration zones. Deliberately not a slope as well -- with k=1
            # a two-parameter fit is meaningless, and the whole point is how
            # little information is needed.
            shift = float(np.median(y[idx] - p[idx]))
            got.append(metrics(y[rest], p[rest] + shift))
        rows.append({
            "k": k,
            "r2": float(np.median([g["r2"] for g in got])),
            "spearman": float(np.median([g["spearman"] for g in got])),
            "median_ratio_error": float(np.median([g["median_ratio_error"] for g in got])),
        })
    return rows


if __name__ == "__main__":
    rows = curve()
    print("zonas conocidas ->  R2      rho     error mediano")
    for row in rows:
        tag = "ninguna" if row["k"] == 0 else f"{row['k']:>7}"
        print(f"  {tag}      {row['r2']:+.3f}   {row['spearman']:.3f}   "
              f"x{row['median_ratio_error']:.2f}")
    out = RUN.parent / "calibration.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nguardado en {out.name}")
