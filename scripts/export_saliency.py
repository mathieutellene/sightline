"""Where the network was looking, measured by taking the view away.

The feature maps show what the network computed. They do not show what it
*used*: a channel can light up brilliantly over a car park that made no
difference to the answer. Occlusion answers the harder question directly —
cover a patch of the chip with flat grey, run the whole network again, and see
how far the estimate moves. A patch that matters leaves a hole in the answer.

This is Zeiler and Fergus's occlusion sensitivity, and its virtue is that it
needs no gradients, no assumptions and no extra training: it is the model's own
arithmetic, run 196 times per zone with one square of the world hidden each
time. What comes out is honest in a way a saliency heuristic is not — every
pixel of the map is a real prediction the network made.

READ THE SIGN. Red means hiding that patch *lowered* the estimate: the network
was reading demand there. Blue means hiding it *raised* the estimate: the patch
was evidence against, which given what the deep blocks key on is usually
greenery or open ground. White means covering it changed nothing.

The scale is shared across all 77 zones, for the same reason the feature maps
are: a per-zone scale would make every map look equally decisive, including the
ones where the network barely reacted to anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sightline import demand, tiles, zones as Z          # noqa: E402
from sightline.model import DemandNet, normalise         # noqa: E402

OUT = ROOT / "docs" / "viz" / "saliency"
PATCH = 24          # side of the occluding square, in pixels (240 m of ground)
STRIDE = 8          # 80 m between probes: 14x14 = 196 forward passes per zone
GREY = 0.0          # flat mid-grey once normalised to [-1, 1]
RENDER_PX = 256     # 128 upscaled by 2, so the map lines up with the chip


def ramp_diverging(t: np.ndarray) -> np.ndarray:
    """-1..+1 -> blue .. white .. red, as RGB uint8.

    Diverging rather than the project's usual ramp, because this quantity has a
    meaningful zero: patches that changed nothing must read as nothing, not as
    the bottom of a scale.
    """
    blue = np.array([26, 115, 232]); white = np.array([255, 255, 255])
    red = np.array([217, 48, 37])
    t = np.clip(t, -1, 1)[..., None]
    neg = white + (blue - white) * np.clip(-t, 0, 1)
    pos = white + (red - white) * np.clip(t, 0, 1)
    return np.where(t < 0, neg, pos).astype(np.uint8)


def main():
    run = json.loads((ROOT / "data" / "runs" / "result.json").read_text())
    net = DemandNet()
    net.load_state_dict(torch.load(ROOT / "data" / "runs" / "demandnet.pt",
                                   weights_only=True))
    net.eval()
    mu, sd = run["target_mu"], run["target_sd"]

    zs = Z.load("chicago")
    chips = tiles.build("chicago", zs, verbose=False)
    dens = demand.density("chicago", zs)
    keys = [z.key for z in zs if z.key in chips and z.key in dens]
    by = {z.key: z for z in zs}

    offsets = list(range(0, 128 - PATCH + 1, STRIDE))
    print(f"{len(keys)} zonas x {len(offsets) ** 2} oclusiones "
          f"(parche {PATCH}px = {PATCH * 10} m, paso {STRIDE * 10} m)")

    maps: dict[str, np.ndarray] = {}
    for n, key in enumerate(keys, 1):
        base = normalise(chips[key][None])                # (1,3,128,128)
        batch = np.repeat(base, len(offsets) ** 2, axis=0)
        for i, (oy, ox) in enumerate((y, x) for y in offsets for x in offsets):
            batch[i, :, oy:oy + PATCH, ox:ox + PATCH] = GREY

        with torch.no_grad():
            ref = float(net(torch.from_numpy(base)).item())
            out = net(torch.from_numpy(batch)).numpy()

        # Accumulate each probe's effect over the square it covered, then divide
        # by how many probes touched each pixel: the patches overlap, and the
        # edges are covered by fewer of them.
        acc = np.zeros((128, 128), dtype=np.float64)
        hits = np.zeros((128, 128), dtype=np.float64)
        for i, (oy, ox) in enumerate((y, x) for y in offsets for x in offsets):
            acc[oy:oy + PATCH, ox:ox + PATCH] += (ref - out[i]) * sd
            hits[oy:oy + PATCH, ox:ox + PATCH] += 1
        maps[key] = acc / np.maximum(hits, 1)             # in log10 units
        if n % 20 == 0:
            print(f"  {n}/{len(keys)}", flush=True)

    # One scale for every zone, anchored on the 98th percentile of magnitude so
    # a single extreme patch cannot flatten all the others.
    scale = float(np.percentile(np.abs(np.stack(list(maps.values()))), 98))
    print(f"escala compartida: +/- {scale:.4f} en log10 "
          f"(x{10 ** scale:.2f} de cambio en viajes)")

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for key, m in maps.items():
        img = Image.fromarray(ramp_diverging(m / scale)).resize(
            (RENDER_PX, RENDER_PX), Image.BILINEAR)
        img.save(OUT / f"{key}.png", optimize=True)
        meta[key] = {"max_drop": round(float(m.max()), 4),
                     "max_rise": round(float(-m.min()), 4)}

    (OUT / "index.json").write_text(json.dumps({
        "patch_px": PATCH, "stride_px": STRIDE, "metres_per_px": 10,
        "scale_log10": round(scale, 5),
        "probes_per_zone": len(offsets) ** 2,
        "zones": meta,
    }, indent=1))

    size = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e6
    top = max(meta.items(), key=lambda kv: kv[1]["max_drop"])
    print(f"{len(meta)} mapas -> {size:.1f} MB en docs/viz/saliency")
    print(f"mayor caída al tapar: {by[top[0]].name.title()} "
          f"(x{10 ** top[1]['max_drop']:.2f})")


if __name__ == "__main__":
    main()
