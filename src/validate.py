#!/usr/bin/env python3
"""Validate generated islands vs real Japan terrain: statistical + visual.

  python src/validate.py --config configs/phase1.yaml \
      --gen-dir outputs/generated --n-real 32 --out reports/phase1

Outputs (to --out):
  - metrics.json            full numeric comparison (metrics.compare_all)
  - summary.txt             human-readable one-line-per-metric
  - real_montage.png        grid of real crops (shaded relief)
  - fake_montage.png        grid of generated islands (shaded relief)
  - psd_curve.png           radial PSD log-log, real vs fake
  - hypsometric.png         hypsometric curves, real vs fake
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import TerrainMosaic  # noqa
import metrics as M  # noqa
from render import grid_montage, save_png  # noqa


def sample_real(mosaic, size, n, lo, hi, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        out.append(mosaic.sample_window(size, lo, hi, rng=rng).astype(np.float32))
    return out


def load_gen(gen_dir):
    dems = []
    for p in sorted(Path(gen_dir).glob("*.tif")):
        with rasterio.open(p) as ds:
            dems.append(ds.read(1).astype(np.float32))
    return dems


def plot_curves(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # PSD
    p = res["psd"]
    fig, ax = plt.subplots(figsize=(6, 4))
    k = np.array(p["k"]);
    ax.plot(np.log10(k + 1e-12), p["logP_real"], label=f"real (beta={p['beta_real']:.2f})")
    ax.plot(np.log10(k + 1e-12), p["logP_fake"], label=f"fake (beta={p['beta_fake']:.2f})")
    ax.set_xlabel("log10 k (cycles/m)"); ax.set_ylabel("log P(k)")
    ax.set_title("Radially-averaged power spectrum"); ax.legend()
    fig.tight_layout(); fig.savefig(Path(out_dir) / "psd_curve.png", dpi=110); plt.close(fig)
    # hypsometric
    h = res["hypsometric"]
    fig, ax = plt.subplots(figsize=(6, 4))
    a = np.linspace(0, 1, len(h["curve_real"]))
    ax.plot(h["curve_real"], a, label=f"real (HI={h['HI_real']:.2f})")
    ax.plot(h["curve_fake"], a, label=f"fake (HI={h['HI_fake']:.2f})")
    ax.set_xlabel("relative area (>= h)"); ax.set_ylabel("relative height")
    ax.set_title("Hypsometric curve"); ax.legend()
    fig.tight_layout(); fig.savefig(Path(out_dir) / "hypsometric.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--n-real", type=int, default=32)
    ap.add_argument("--real-lo", type=float, default=0.40)
    ap.add_argument("--real-hi", type=float, default=0.80)
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    vmax = cfg["data"]["vmax"]; res = cfg["data"]["res_m"]
    sea = cfg["data"]["sea_thresh"]

    fake = load_gen(args.gen_dir)
    if not fake:
        raise SystemExit(f"no *.tif in {args.gen_dir}")
    size = fake[0].shape[0]
    print(f"[validate] {len(fake)} generated DEMs at {size}px")

    mosaic = TerrainMosaic(cfg["data"]["mosaic"], cfg["data"]["mask"], sea)
    real = sample_real(mosaic, size, args.n_real, args.real_lo, args.real_hi, args.seed)
    print(f"[validate] {len(real)} real crops at {size}px")

    res_metrics = M.compare_all(real, fake, res=res, vmax=vmax, sea_thresh=sea)
    (out_dir / "metrics.json").write_text(json.dumps(res_metrics, indent=2))
    summary = M.summarize(res_metrics)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)

    # visual montages (cap count for readability)
    nshow = min(12, len(real), len(fake))
    save_png(grid_montage(real[:nshow], res=res, vmax=vmax, cols=4, mode="shaded"),
             out_dir / "real_montage.png")
    save_png(grid_montage(fake[:nshow], res=res, vmax=vmax, cols=4, mode="shaded"),
             out_dir / "fake_montage.png")
    plot_curves(res_metrics, out_dir)
    print(f"[validate] wrote report to {out_dir}")


if __name__ == "__main__":
    main()
