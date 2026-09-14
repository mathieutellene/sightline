"""Pack the trained network into something a browser can run by itself.

The page already shows activations captured in Python. That proves what the
network did once, on a machine nobody can see. Shipping the weights lets the
page do the arithmetic in front of the reader instead — same numbers, no server,
no inference API, nothing to trust.

Two things happen here.

BATCH NORM IS FOLDED INTO THE CONVOLUTION. At eval time a BatchNorm is an affine
map with fixed constants, so it can be pushed into the preceding convolution's
weights and bias exactly:

    W' = W * gamma / sqrt(var + eps)
    b' = beta - gamma * mean / sqrt(var + eps)

That removes a whole layer type from the JavaScript, which then only needs
convolution, ReLU, an average and two matrix multiplies. Fewer kinds of thing to
get wrong, and the result is bit-comparable rather than approximately similar.

THE WEIGHTS GO OUT AS RAW FLOAT32. 591k parameters is 2.3 MB as binary and about
14 MB as JSON, and the browser can map the binary straight into a Float32Array
with no parsing at all.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sightline.model import DemandNet                      # noqa: E402

OUT = ROOT / "docs" / "net"


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    """Conv + BatchNorm at eval -> one convolution with a bias."""
    w = conv.weight.detach().numpy().astype(np.float32)
    gamma = bn.weight.detach().numpy().astype(np.float32)
    beta = bn.bias.detach().numpy().astype(np.float32)
    mean = bn.running_mean.detach().numpy().astype(np.float32)
    var = bn.running_var.detach().numpy().astype(np.float32)
    scale = gamma / np.sqrt(var + bn.eps)
    return w * scale[:, None, None, None], beta - mean * scale


def main():
    run = json.loads((ROOT / "data" / "runs" / "result.json").read_text())
    net = DemandNet()
    net.load_state_dict(torch.load(ROOT / "data" / "runs" / "demandnet.pt",
                                   weights_only=True))
    net.eval()

    layers, blob = [], bytearray()

    def put(kind: str, arr: np.ndarray, **meta):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        layers.append({"kind": kind, "shape": list(a.shape),
                       "offset": len(blob) // 4, "count": a.size, **meta})
        blob.extend(a.tobytes())

    for bi in range(4):                       # four convolutional blocks
        block = net.features[bi]
        for ci, stride in ((0, 2), (3, 1)):   # each is conv, bn, relu
            w, b = fold(block[ci], block[ci + 1])
            put("conv", w, stride=stride, pad=1, block=bi + 1,
                cin=int(w.shape[1]), cout=int(w.shape[0]))
            put("bias", b, block=bi + 1)

    put("pool", np.zeros(0, dtype=np.float32))            # global average
    for li in (2, 4):                                     # head: 128->64->1
        lin = net.head[li]
        put("linear", lin.weight.detach().numpy(), relu=(li == 2))
        put("bias", lin.bias.detach().numpy())

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "demandnet.bin").write_bytes(bytes(blob))
    (OUT / "demandnet.json").write_text(json.dumps({
        "input": {"px": 128, "channels": 3, "scale": 1 / 127.5, "shift": -1.0},
        "target": {"mu": run["target_mu"], "sd": run["target_sd"], "base10": True},
        "params": sum(l["count"] for l in layers),
        "layers": layers,
    }, indent=1))

    # A reference forward pass, so the JavaScript has something to be checked
    # against rather than merely looking plausible.
    chips = np.load(ROOT / "data" / "chips" / "chicago_2022-06.npz")
    probe = {}
    for key in ("chicago-8", "chicago-32", "chicago-55"):
        if key not in chips:
            continue
        x = torch.from_numpy((chips[key].astype(np.float32) / 127.5) - 1.0)[None]
        with torch.no_grad():
            y = float(net(x).item())
        probe[key] = round(y * run["target_sd"] + run["target_mu"], 6)
    (OUT / "probe.json").write_text(json.dumps(probe, indent=1))

    mb = len(blob) / 1e6
    print(f"{sum(l['count'] for l in layers):,} parámetros -> {mb:.2f} MB")
    print(f"{len([l for l in layers if l['kind'] == 'conv'])} convoluciones, "
          f"BatchNorm plegada en cada una")
    for k, v in probe.items():
        print(f"  referencia {k}: log10 = {v:.6f}  ->  {10 ** v:,.1f} viajes/km²/día")


if __name__ == "__main__":
    main()
