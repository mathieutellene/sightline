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


def ceiling() -> dict:
    """Two questions the curve above raises but cannot answer on its own.

    HOW GOOD COULD THIS GET? The curve flattens, and a flat curve is only good
    news if it flattens near the best achievable. So: fit the correction with
    every one of Chicago's answers in hand. That is not a result — it reads the
    test set — it is the bound the honest numbers should be read against.

    IS ONE SCALAR THE RIGHT MODEL? The scatter looks squeezed towards the
    middle, which invites a second parameter to stretch it back. Whether that
    helps is a measurable question, not a matter of taste, and it is measured
    here rather than assumed either way.
    """
    r = json.loads(RUN.read_text())
    keys = sorted(r["chicago_true"])
    y = np.log10(np.array([r["chicago_true"][k] for k in keys]))
    p = np.log10(np.array([r["chicago_pred"][k] for k in keys]))
    all_idx = np.arange(len(keys))

    def err(a, b, idx):
        return float(10 ** np.median(np.abs((a * p[idx] + b) - y[idx])))

    # Search the slope against the metric actually reported. Fitting by least
    # squares instead answers a different question and lands somewhere worse.
    grid = [(a, float(np.median(y - a * p))) for a in np.arange(0.2, 2.001, 0.01)]
    best_a, best_b = min(grid, key=lambda ab: err(*ab, all_idx))

    rng = np.random.default_rng(0)
    two = {}
    for k in (2, 3, 5, 8, 13, 21):
        got = []
        for _ in range(400):
            fit = rng.choice(len(keys), size=k, replace=False)
            rest = np.setdiff1d(all_idx, fit)
            a, b = np.polyfit(p[fit], y[fit], 1)
            got.append(err(a, b, rest))
        two[k] = float(np.median(got))

    return {
        "offset_only_ceiling": err(1.0, float(np.median(y - p)), all_idx),
        "affine_ceiling": err(best_a, best_b, all_idx),
        "affine_ceiling_slope": round(float(best_a), 3),
        "two_parameter_curve": two,
        "spread_decades": {"measured": round(float(y.max() - y.min()), 2),
                           "predicted": round(float(p.max() - p.min()), 2)},
        "within": {"x1.5": round(float((np.abs(y - p) < np.log10(1.5)).mean()), 3),
                   "x2": round(float((np.abs(y - p) < np.log10(2)).mean()), 3)},
        "ratio_percentiles": {str(q): round(float(10 ** np.percentile(np.abs(y - p), q)), 2)
                              for q in (50, 75, 90, 100)},
    }


if __name__ == "__main__":
    rows = curve()
    print("zonas conocidas ->  R2      rho     error mediano")
    for row in rows:
        tag = "ninguna" if row["k"] == 0 else f"{row['k']:>7}"
        print(f"  {tag}      {row['r2']:+.3f}   {row['spearman']:.3f}   "
              f"x{row['median_ratio_error']:.2f}")

    c = ceiling()
    print(f"\ncon las 77 respuestas a la vista (cota, no resultado):")
    print(f"  solo desplazamiento  x{c['offset_only_ceiling']:.2f}")
    print(f"  afín completo        x{c['affine_ceiling']:.2f}  "
          f"(pendiente óptima {c['affine_ceiling_slope']})")
    print("\najustar también la pendiente, con k zonas:")
    for k, v in c["two_parameter_curve"].items():
        base = next(r["median_ratio_error"] for r in rows if r["k"] == k)
        print(f"  k={k:>2}: x{v:.2f}   frente a x{base:.2f} con solo desplazamiento"
              f"   ({'peor' if v > base else 'mejor'})")
    print(f"\nzonas dentro de x2: {c['within']['x2'] * 100:.0f}%   "
          f"dentro de x1.5: {c['within']['x1.5'] * 100:.0f}%")

    out = RUN.parent / "calibration.json"
    out.write_text(json.dumps({"curve": rows, "ceiling": c}, indent=1))
    print(f"\nguardado en {out.name}")
