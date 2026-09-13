"""Train on one city, test on another.

The split that matters is not inside New York. Adjacent zones share streets,
share traffic and share whatever the satellite sees, so a random hold-out
inside one city measures interpolation and flatters the model badly. New York
is used only to fit and to stop early; the reported result is Chicago, a city
the network never saw in any form.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import demand, tiles, zones as Z
from .model import DemandNet, augment, metrics, normalise, to_target

OUT = Path(__file__).resolve().parents[1] / "data" / "runs"


def dataset(city: str, month="2022-06"):
    zs = Z.load(city)
    chips = tiles.build(city, zs, month, verbose=False)
    dens = demand.density(city, zs, month)
    keys = [z.key for z in zs if z.key in chips and z.key in dens]
    X = normalise(np.stack([chips[k] for k in keys]))
    y = np.array([to_target(dens[k]) for k in keys], dtype=np.float32)
    return keys, X, y, {z.key: z for z in zs}


def run(epochs=60, seed=0, lr=2e-3, batch=32, val_frac=0.15):
    torch.manual_seed(seed); np.random.seed(seed)

    tr_keys, Xtr, ytr, _ = dataset("nyc")
    te_keys, Xte, yte, te_zones = dataset("chicago")
    print(f"entrenamiento NYC {Xtr.shape} | prueba Chicago {Xte.shape}")

    idx = np.random.permutation(len(Xtr))
    n_val = int(len(idx) * val_frac)
    va, tr = idx[:n_val], idx[n_val:]

    # Standardise the target on the training split only; letting the test city
    # touch these two numbers would be leakage, small but real.
    mu, sd = float(ytr[tr].mean()), float(ytr[tr].std())
    norm = lambda a: (a - mu) / sd
    denorm = lambda a: a * sd + mu

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(norm(ytr))
    Xte_t = torch.from_numpy(Xte)

    net = DemandNet()
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.SmoothL1Loss()
    print(f"parámetros: {n_par:,}")

    best = {"val": 1e9, "state": None, "epoch": -1}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb = augment(Xtr_t[b]); yb = ytr_t[b]
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()

        net.eval()
        with torch.no_grad():
            pv = denorm(net(Xtr_t[va]).numpy())
        vm = metrics(ytr[va], pv)
        if vm["mae_log10"] < best["val"]:
            best = {"val": vm["mae_log10"], "epoch": ep,
                    "state": {k: v.clone() for k, v in net.state_dict().items()}}
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:>3}  train {tot/len(perm):.4f}  "
                  f"val R2 {vm['r2']:+.3f}  MAE {vm['mae_log10']:.3f}")

    net.load_state_dict(best["state"])
    net.eval()
    print(f"mejor época {best['epoch']} ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        pred_nyc = denorm(net(Xtr_t[va]).numpy())
        pred_chi = denorm(net(Xte_t).numpy())

    m_nyc = metrics(ytr[va], pred_nyc)
    m_chi = metrics(yte, pred_chi)
    print(f"\nNYC (validación, misma ciudad)  R2 {m_nyc['r2']:+.3f}  "
          f"rho {m_nyc['spearman']:.3f}  error x{m_nyc['median_ratio_error']:.2f}")
    print(f"CHICAGO (nunca vista)           R2 {m_chi['r2']:+.3f}  "
          f"rho {m_chi['spearman']:.3f}  error x{m_chi['median_ratio_error']:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], OUT / "demandnet.pt")
    json.dump({
        "params": n_par, "best_epoch": best["epoch"],
        "target_mu": mu, "target_sd": sd,
        "nyc_val": m_nyc, "chicago": m_chi,
        "chicago_pred": {k: float(10 ** p) for k, p in zip(te_keys, pred_chi)},
        "chicago_true": {k: float(10 ** t) for k, t in zip(te_keys, yte)},
    }, open(OUT / "result.json", "w"), indent=1)
    print(f"guardado en {OUT}")
    return m_chi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    run(**vars(ap.parse_args()))
