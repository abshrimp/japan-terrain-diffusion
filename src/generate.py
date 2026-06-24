#!/usr/bin/env python3
"""Generate fictional-island DEMs with the trained cascade.

Pipeline (NO spatial tiling, NO seam blending):
  1. Coarse EDM samples the FULL canvas at coarse resolution (one un-tiled pass/step).
  2. Coarse map is bicubically upsampled to the final canvas size -> SR condition.
  3. SR EDM refines the ENTIRE canvas in a single fully-convolutional pass/step.

Three modes:
  * default        coarse -> post-select by land area -> SR -> save final islands.
  * --coarse-only  mass-produce coarse drafts (fast, no SR): saves each coarse DEM
                   (.npy) + a preview PNG + a labelled contact_sheet.png for browsing.
  * --complete DIR finish chosen coarse drafts from a --coarse-only run through SR.

  # browse: make 100 coarse drafts
  python src/generate.py --config configs/phase1.yaml \
      --coarse-ckpt checkpoints/phase1/coarse/latest.pt \
      --coarse-only --n 100 --out outputs/drafts
  # complete the ones you like (by id substring) at full resolution
  python src/generate.py --config configs/phase1.yaml \
      --sr-ckpt checkpoints/phase1/sr/latest.pt \
      --complete outputs/drafts --pick 003 041 --out outputs/final --hydro
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml
from affine import Affine

import sys
sys.path.insert(0, str(Path(__file__).parent))
from networks import UNet  # noqa
from edm import EDM  # noqa
from dataset import denormalize, normalize  # noqa
from render import shaded_relief, hillshade, color_relief, save_png  # noqa
from hydro import fill_depressions, flow_accumulation, drainage_overlay  # noqa

DST_CRS = ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=38 +lon_0=137 "
           "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def load_stage(ckpt_path, device, use_ema=True):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]
    stage = ck["stage"]
    m = cfg[stage]["model"]
    net = UNet(in_ch=1, cond_ch=m.get("cond_ch", 0), out_ch=1, base=m["base"],
               ch_mult=tuple(m["ch_mult"]), num_res=m["num_res"],
               use_attn=m.get("attn", True))
    sd = ck["ema"] if use_ema else ck["net"]
    net.load_state_dict(sd)
    edm = EDM(net, sigma_data=m.get("sigma_data", 0.5)).to(device).eval()
    return edm, cfg, ck["step"]


def land_km2(dem_m, res, sea_thresh=0.5):
    return float((dem_m > sea_thresh).sum()) * res * res / 1e6


def largest_component_km2(dem_m, res, sea_thresh=0.5):
    """Area (km2) of the LARGEST connected land mass, and its fraction of all land."""
    from scipy.ndimage import label
    land = dem_m > sea_thresh
    lab, n = label(land)
    if n == 0:
        return 0.0, 0.0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = sizes.max()
    total = land.sum()
    km2 = float(biggest) * res * res / 1e6
    frac = float(biggest) / float(total) if total else 0.0
    return km2, frac


def save_geotiff(dem_m, path, res):
    h, w = dem_m.shape
    transform = Affine(res, 0, 0, 0, -res, 0)
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": h,
            "width": w, "crs": DST_CRS, "transform": transform,
            "compress": "deflate", "predictor": 2}
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(dem_m.astype(np.float32), 1)


def grid_params(cfg, args):
    g = cfg["geometry"]
    res = cfg["data"]["res_m"]
    canvas_px = args.canvas_px or g["canvas_px"]
    coarse_px = canvas_px // (g["canvas_px"] // g["coarse_px"])
    coarse_res = res * (canvas_px // coarse_px)
    return canvas_px, coarse_px, coarse_res


# ----------------------------- coarse drafting ----------------------------- #
def gen_coarse_candidates(coarse, cfg, args, coarse_px, coarse_res):
    """Sample args.n coarse drafts; return list of dicts (dem_c + land stats)."""
    vmax, nm = cfg["data"]["vmax"], cfg["data"].get("norm", "sqrt")
    cstep = args.coarse_steps or cfg["coarse"]["sample"]["steps"]
    cands = []
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(args.n):
            x = coarse.sample((1, 1, coarse_px, coarse_px), "cuda",
                              steps=cstep, seed=args.seed + i)
            dem_c = denormalize(x[0, 0].float().cpu().numpy(), vmax, nm)
            lcc, frac = largest_component_km2(dem_c, coarse_res)
            cands.append({"idx": i, "seed": args.seed + i, "dem_c": dem_c,
                          "lcc_km2": lcc, "frac": frac,
                          "total_km2": land_km2(dem_c, coarse_res)})
            if (i + 1) % 10 == 0 or i + 1 == args.n:
                print(f"  [coarse] {i+1}/{args.n}", flush=True)
    return cands


def save_coarse_draft(cand, out, coarse_res, vmax):
    """Save a coarse draft: .npy DEM (to complete later) + shaded-relief preview PNG."""
    tag = f"coarse_{cand['idx']:03d}_seed{cand['seed']}_{round(cand['lcc_km2'])}km2"
    np.save(out / f"{tag}.npy", cand["dem_c"].astype(np.float32))
    save_png(shaded_relief(cand["dem_c"], res=coarse_res, vmax=vmax),
             out / f"{tag}.png")
    return {"tag": tag, "idx": cand["idx"], "seed": cand["seed"],
            "lcc_km2": round(cand["lcc_km2"], 1), "frac": round(cand["frac"], 2),
            "total_km2": round(cand["total_km2"], 1), "npy": f"{tag}.npy"}


def contact_sheet(recs, out, coarse_res, vmax, cols=8):
    """Labelled montage of all coarse drafts for quick visual browsing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(recs)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1), squeeze=False)
    axs = axs.ravel()
    for ax, rec in zip(axs, recs):
        dem = np.load(out / rec["npy"])
        ax.imshow(shaded_relief(dem, res=coarse_res, vmax=vmax))
        ax.set_title(f"{rec['idx']:03d}  {round(rec['lcc_km2'])}km²  f{rec['frac']:.2f}",
                     fontsize=6)
        ax.axis("off")
    for ax in axs[n:]:
        ax.axis("off")
    fig.suptitle("coarse drafts — id / largest-island area / single-island fraction",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "contact_sheet.png", dpi=120)
    plt.close(fig)


# ----------------------------- SR finishing ----------------------------- #
def finish_island(sr, dem_c, seed, rank, cfg, args, out, canvas_px):
    """Upsample a coarse DEM -> SR refine (one full-canvas pass) -> optional hydro -> save."""
    vmax, nm = cfg["data"]["vmax"], cfg["data"].get("norm", "sqrt")
    res, sea = cfg["data"]["res_m"], cfg["data"].get("sea_thresh", 0.5)
    sstep = args.sr_steps or cfg["sr"]["sample"]["steps"]
    device = "cuda"
    xc = torch.from_numpy(normalize(dem_c, vmax, nm).astype(np.float32))[None, None].to(device)
    cond = F.interpolate(xc, size=(canvas_px, canvas_px), mode="bicubic", align_corners=False)
    if sr is not None:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x = sr.sample((1, 1, canvas_px, canvas_px), device, cond=cond,
                          steps=sstep, seed=seed)
        dem = denormalize(x[0, 0].float().cpu().numpy(), vmax, nm)
    else:
        dem = denormalize(cond[0, 0].float().cpu().numpy(), vmax, nm)
    hstats = None
    if args.hydro:
        dem, hstats = fill_depressions(dem, sea_level=sea, epsilon=args.hydro_epsilon)
        print(f"      [hydro] filled {hstats['cells_raised']} cells "
              f"({hstats['frac_land_raised']*100:.1f}% land, mean {hstats['mean_fill_m']:.1f} m); "
              f"strict sinks {hstats['sinks_before']}->{hstats['sinks_after']}")
    total_km2 = land_km2(dem, res)
    lcc_km2, frac = largest_component_km2(dem, res)
    tag = f"island_{rank:02d}_seed{seed}_{round(lcc_km2)}km2"
    save_geotiff(dem, out / f"{tag}.tif", res)
    save_png(shaded_relief(dem, res=res, vmax=vmax), out / f"{tag}_shaded.png")
    save_png(hillshade(dem, res=res), out / f"{tag}_hillshade.png")
    save_png(color_relief(dem, vmax=vmax), out / f"{tag}_color.png")
    if args.hydro_drainage:
        acc = flow_accumulation(dem, sea_level=sea)
        save_png(drainage_overlay(dem, acc, sea_level=sea), out / f"{tag}_drainage.png")
    rec = {"tag": tag, "seed": seed, "island_km2": round(lcc_km2, 1),
           "total_land_km2": round(total_km2, 1), "single_island_frac": round(frac, 2),
           "elev_max_m": round(float(dem.max()), 1)}
    if hstats:
        rec["hydro"] = hstats
    print(f"  [{rank}] {tag}: island {lcc_km2:.0f} km2 "
          f"(total land {total_km2:.0f}, frac {frac:.2f}), max {dem.max():.0f} m")
    return rec


def load_drafts(draft_dir, picks):
    """Load coarse drafts from a --coarse-only run, optionally filtered by --pick tokens."""
    draft_dir = Path(draft_dir)
    man_path = draft_dir / "coarse_manifest.json"
    if man_path.exists():
        recs = json.loads(man_path.read_text())
    else:  # fall back to globbing
        recs = []
        for p in sorted(draft_dir.glob("coarse_*.npy")):
            recs.append({"tag": p.stem, "npy": p.name,
                         "seed": int(p.stem.split("seed")[1].split("_")[0])})
    if picks:
        recs = [r for r in recs if any(tok in r["tag"] for tok in picks)]
    out = []
    for r in recs:
        out.append((np.load(draft_dir / r["npy"]).astype(np.float32), int(r["seed"]), r["tag"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--coarse-ckpt", default="")
    ap.add_argument("--sr-ckpt", default="")
    ap.add_argument("--n", type=int, default=16, help="coarse candidates / drafts")
    ap.add_argument("--keep", type=int, default=4, help="how many to SR + save (default mode)")
    ap.add_argument("--target-km2", type=float, default=5000.0)
    ap.add_argument("--tol-km2", type=float, default=1000.0)
    ap.add_argument("--canvas-px", type=int, default=0, help="override final canvas")
    ap.add_argument("--coarse-steps", type=int, default=0)
    ap.add_argument("--sr-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/generated")
    ap.add_argument("--no-ema", action="store_true")
    # two-phase browsing workflow
    ap.add_argument("--coarse-only", action="store_true",
                    help="STAGE 1 ONLY: mass-produce coarse drafts (no SR) for browsing")
    ap.add_argument("--complete", default="",
                    help="STAGE 2: finish coarse drafts from this --coarse-only dir via SR")
    ap.add_argument("--pick", nargs="+", default=None,
                    help="with --complete: id substrings to finish (e.g. 003 041); all if omitted")
    # hydrological conditioning
    ap.add_argument("--hydro", action="store_true",
                    help="fill depressions so the island drains to the sea")
    ap.add_argument("--hydro-epsilon", type=float, default=1e-3,
                    help="m of drainage gradient across flats (0 = flat fill, faster)")
    ap.add_argument("--hydro-drainage", action="store_true",
                    help="also save a D8 river-network overlay (implies --hydro)")
    args = ap.parse_args()
    if args.hydro_drainage:
        args.hydro = True

    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    vmax = cfg["data"]["vmax"]
    canvas_px, coarse_px, coarse_res = grid_params(cfg, args)

    # ---------------- STAGE 2: complete chosen drafts ---------------- #
    if args.complete:
        if not args.sr_ckpt:
            raise SystemExit("--complete needs --sr-ckpt")
        drafts = load_drafts(args.complete, args.pick)
        if not drafts:
            raise SystemExit(f"no drafts matched in {args.complete} (pick={args.pick})")
        sr, _, ss = load_stage(args.sr_ckpt, device, use_ema=not args.no_ema)
        print(f"[complete] finishing {len(drafts)} drafts from {args.complete} "
              f"with SR (step {ss}) at {canvas_px}px in ONE pass each")
        results = [finish_island(sr, dem_c, seed, rank, cfg, args, out, canvas_px)
                   for rank, (dem_c, seed, tag) in enumerate(drafts)]
        (out / "manifest.json").write_text(json.dumps(results, indent=2))
        print(f"[done] wrote {len(results)} islands to {out}")
        return

    # ---------------- STAGE 1: coarse drafts (browse or full pipeline) ---------------- #
    if not args.coarse_ckpt:
        raise SystemExit("need --coarse-ckpt (or use --complete with --sr-ckpt)")
    coarse, _, cs = load_stage(args.coarse_ckpt, device, use_ema=not args.no_ema)
    print(f"[coarse] loaded (step {cs}); sampling {args.n} drafts at "
          f"{coarse_px}px ({coarse_res:.0f} m/px)")
    cands = gen_coarse_candidates(coarse, cfg, args, coarse_px, coarse_res)
    lccs = sorted(c["lcc_km2"] for c in cands)
    print(f"[coarse] largest-island km2 dist: min={lccs[0]:.0f} "
          f"med={lccs[len(lccs)//2]:.0f} max={lccs[-1]:.0f}")

    if args.coarse_only:
        recs = [save_coarse_draft(c, out, coarse_res, vmax) for c in cands]
        recs.sort(key=lambda r: r["idx"])
        (out / "coarse_manifest.json").write_text(json.dumps(recs, indent=2))
        contact_sheet(recs, out, coarse_res, vmax)
        print(f"[coarse-only] wrote {len(recs)} drafts + contact_sheet.png to {out}\n"
              f"  browse contact_sheet.png, then finish picks with:\n"
              f"  python src/generate.py --config {args.config} --sr-ckpt <sr.pt> "
              f"--complete {out} --pick <ids...> --out outputs/final")
        return

    # default: post-select by largest-island area, then SR the keepers
    for c in cands:
        c["score"] = abs(c["lcc_km2"] - args.target_km2) + (1 - c["frac"]) * args.target_km2 * 0.3
    cands.sort(key=lambda c: c["score"])
    keepers = [c for c in cands
               if abs(c["lcc_km2"] - args.target_km2) <= args.tol_km2][:args.keep]
    if not keepers:
        keepers = cands[:args.keep]
        print("[warn] none within tolerance; taking best-scored")
    print(f"[select] kept {len(keepers)}: largest-island km2="
          f"{[round(c['lcc_km2']) for c in keepers]}")
    sr = None
    if args.sr_ckpt:
        sr, _, ss = load_stage(args.sr_ckpt, device, use_ema=not args.no_ema)
        print(f"[sr] loaded (step {ss}); refining full {canvas_px}px canvas in ONE pass")
    results = [finish_island(sr, c["dem_c"], c["seed"], rank, cfg, args, out, canvas_px)
               for rank, c in enumerate(keepers)]
    (out / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {len(results)} islands to {out}")


if __name__ == "__main__":
    main()
