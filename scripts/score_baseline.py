"""Score the OpenStreetMap baseline against the same cities, the same way.

The README has been carrying an unanswered objection: that imagery is a
roundabout way of measuring urban density and OSM hands it over directly. This
answers it with a number instead of a paragraph.

The comparison is deliberately like-for-like. Same zones, same 1.28 km squares,
same log10 target, same train-on-New-York / test-on-Chicago split, same four
metrics. The only difference is what the model is shown: four numbers about the
road network, or 49,152 pixels.

The baseline model is ridge regression on log1p features plus an intercept. It
is the right shape of model for four correlated, heavy-tailed density features
and 255 training rows — anything larger would overfit and would also be a
strawman in the other direction, since the point is to find out what the
features carry, not how well a regressor can be tuned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sightline import baseline, demand, zones as Z         # noqa: E402
from sightline.model import metrics, to_target             # noqa: E402
from sightline.tiles import CITY_BBOX                      # noqa: E402

OUT = ROOT / "data" / "runs" / "baseline.json"


def design(feats: dict, keys: list[str]) -> np.ndarray:
    """Features -> log1p, which is the scale density actually lives on."""
    X = np.array([[feats[k][f] for f in baseline.FEATURES] for k in keys], dtype=np.float64)
    return np.log1p(X)


def ridge(X, y, lam: float):
    """Closed form, with the intercept left unpenalised."""
    Xc, mu = X - X.mean(0), X.mean(0)
    yc, ybar = y - y.mean(), y.mean()
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    return w, ybar - mu @ w


def main():
    run = json.loads((ROOT / "data" / "runs" / "result.json").read_text())

    data = {}
    for city in ("nyc", "chicago"):
        zs = Z.load(city)
        feats = baseline.features(city, zs, CITY_BBOX[city])
        dens = demand.density(city, zs)
        keys = [z.key for z in zs if z.key in feats and z.key in dens
                and feats[z.key]["road_m"] > 0]
        data[city] = (keys, design(feats, keys),
                      np.array([to_target(dens[k]) for k in keys]))

    (ktr, Xtr, ytr), (kte, Xte, yte) = data["nyc"], data["chicago"]
    print(f"\nOSM: entrenamiento {Xtr.shape} | prueba {Xte.shape}")

    # The penalty is chosen on New York alone, by leaving out a fifth at a time.
    # Choosing it on Chicago would hand the baseline an advantage the network
    # was never given, and the comparison would stop meaning anything.
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(Xtr)) % 5
    best = (None, 1e9)
    for lam in (0.01, 0.1, 1, 3, 10, 30, 100, 300):
        err = []
        for f in range(5):
            tr, va = fold != f, fold == f
            w, b = ridge(Xtr[tr], ytr[tr], lam)
            err.append(np.abs(Xtr[va] @ w + b - ytr[va]).mean())
        if np.mean(err) < best[1]:
            best = (lam, float(np.mean(err)))
    lam = best[0]
    print(f"penalización elegida en NYC: lambda = {lam} (MAE {best[1]:.3f})")

    w, b = ridge(Xtr, ytr, lam)
    m_nyc = metrics(ytr, Xtr @ w + b)
    m_chi = metrics(yte, Xte @ w + b)

    print("\ncoeficientes (log1p de cada feature):")
    for name, coef in sorted(zip(baseline.FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"  {name:12} {coef:+.3f}")

    net = run["chicago"]
    print(f"\n{'':26} {'R2':>8} {'rho':>8} {'error':>8}")
    for label, m in (("OSM en NYC (ajuste)", m_nyc),
                     ("OSM en Chicago", m_chi),
                     ("Red neuronal en Chicago", net)):
        print(f"{label:26} {m['r2']:>+8.3f} {m['spearman']:>8.3f} "
              f"x{m['median_ratio_error']:>7.2f}")

    OUT.write_text(json.dumps({
        "features": baseline.FEATURES, "lambda": lam,
        "coef": {f: float(c) for f, c in zip(baseline.FEATURES, w)},
        "intercept": float(b),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "nyc_fit": m_nyc, "chicago": m_chi,
        "chicago_pred": {k: float(10 ** p) for k, p in zip(kte, Xte @ w + b)},
    }, indent=1))
    print(f"\nguardado en {OUT}")


if __name__ == "__main__":
    main()
