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
                   [Feature: Resumable, Batched & Reproducible. Optimized for high-speed drafting.]
  * --complete DIR finish chosen coarse drafts from a --coarse-only run through SR.

  # browse: make 100 coarse drafts (FAST mode)
  python src/generate.py --config configs/phase1.yaml \
      --coarse-ckpt checkpoints/phase1/coarse/latest.pt \
      --coarse-only --n 100 --batch-size 16 --out outputs/drafts
  # complete the ones you like (by id substring) at full resolution
  python src/generate.py --config configs/phase1.yaml \
      --sr-ckpt checkpoints/phase1/sr/latest.pt \
      --complete outputs/drafts --pick 003 041 --out outputs/final --hydro
"""
import argparse
import json
import math
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

DST_CRS = "EPSG:3857"


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


def save_geotiff(dem_m, path, res, args):
    h, w = dem_m.shape

    R = 6378137.0
    center_x = math.radians(args.center_coords[1]) * R
    center_y = math.log(math.tan(math.pi / 4.0 + math.radians(args.center_coords[0]) / 2.0)) * R
    top_left_x = center_x - (w * res) / 2.0
    top_left_y = center_y + (h * res) / 2.0
    
    transform = Affine(res, 0, top_left_x, 0, -res, top_left_y)
    
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": h,
            "width": w, "crs": DST_CRS, "transform": transform,
            "compress": "deflate", "predictor": 2}
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(dem_m.astype(np.float32), 1)


def grid_params(cfg, args):
    g = cfg["geometry"]
    res = cfg["data"]["res_m"]
    canvas_px = args.canvas_px or g["canvas_px"]
    coarse_px = args.coarse_px or canvas_px // (g["canvas_px"] // g["coarse_px"])
    coarse_res = res * (canvas_px // coarse_px)
    return canvas_px, coarse_px, coarse_res


# ----------------------------- coarse drafting ----------------------------- #
def gen_coarse_candidates(coarse, cfg, args, coarse_px, coarse_res, out_dir=None):
    """Sample args.n coarse drafts in batches; return list of dicts (dem_c + land stats)."""
    vmax, nm = cfg["data"]["vmax"], cfg["data"].get("norm", "sqrt")
    
    cstep = args.coarse_steps
    if cstep == 0:
        if args.coarse_only:
            cstep = 15
            print(f"  [coarse] auto-reduced steps to {cstep} for fast drafting.")
        else:
            cstep = cfg["coarse"]["sample"]["steps"]

    cands = []
    existing_seeds = {}
    if out_dir is not None and out_dir.exists():
        for p in out_dir.glob("coarse_*_seed*.npy"):
            try:
                s_str = [x for x in p.stem.split("_") if x.startswith("seed")][0]
                seed_val = int(s_str.replace("seed", ""))
                existing_seeds[seed_val] = p
            except Exception:
                pass

    missing_tasks = []
    for i in range(args.n):
        current_seed = args.seed + i
        current_idx = args.seed + i
        if current_seed in existing_seeds:
            p = existing_seeds[current_seed]
            dem_c = np.load(p).astype(np.float32)
            lcc, frac = largest_component_km2(dem_c, coarse_res)
            cands.append({"idx": current_idx, "seed": current_seed, "dem_c": dem_c,
                          "lcc_km2": lcc, "frac": frac,
                          "total_km2": land_km2(dem_c, coarse_res),
                          "tag": p.stem})
        else:
            missing_tasks.append((current_idx, current_seed))

    if missing_tasks:
        print(f"  [coarse] loaded {len(existing_seeds)} existing. generating {len(missing_tasks)} new drafts in batches of {args.batch_size}...")

    batch_size = args.batch_size
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for b_start in range(0, len(missing_tasks), batch_size):
            batch_tasks = missing_tasks[b_start : b_start + batch_size]
            b_len = len(batch_tasks)
            
            batch_seeds = [task[1] for task in batch_tasks]
            x = coarse.sample((b_len, 1, coarse_px, coarse_px), "cuda",
                              steps=cstep, seed=batch_seeds)
            
            for k, (idx, seed) in enumerate(batch_tasks):
                dem_c = denormalize(x[k, 0].float().cpu().numpy(), vmax, nm)
                lcc, frac = largest_component_km2(dem_c, coarse_res)
                cand = {"idx": idx, "seed": seed, "dem_c": dem_c,
                        "lcc_km2": lcc, "frac": frac,
                        "total_km2": land_km2(dem_c, coarse_res)}
                cands.append(cand)
                
                if out_dir is not None:
                    save_coarse_draft(cand, out_dir, coarse_res, vmax)
                    
            print(f"  [coarse] processed batch {b_start//batch_size + 1}/{(len(missing_tasks)+batch_size-1)//batch_size}", flush=True)
            
    return cands


def save_coarse_draft(cand, out, coarse_res, vmax):
    tag = cand.get("tag", f"coarse_{cand['idx']:03d}_seed{cand['seed']}_{round(cand['lcc_km2'])}km2")
    npy_path = out / f"{tag}.npy"
    png_path = out / f"{tag}.png"
    
    if not npy_path.exists():
        np.save(npy_path, cand["dem_c"].astype(np.float32))
    if not png_path.exists():
        save_png(shaded_relief(cand["dem_c"], res=coarse_res, vmax=vmax), png_path)
        
    return {"tag": tag, "idx": cand["idx"], "seed": cand["seed"],
            "lcc_km2": round(cand["lcc_km2"], 1), "frac": round(cand["frac"], 2),
            "total_km2": round(cand["total_km2"], 1), "npy": f"{tag}.npy"}


def contact_sheet(recs, out, coarse_res, vmax, cols=8, batch_size=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    for i in range(0, len(recs), batch_size):
        chunk = recs[i:i + batch_size]
        n = len(chunk)
        current_cols = min(cols, n)
        rows = (n + current_cols - 1) // current_cols
        
        fig, axs = plt.subplots(rows, current_cols, figsize=(current_cols * 1.9, rows * 2.1), squeeze=False)
        axs = axs.ravel()
        
        for ax, rec in zip(axs, chunk):
            dem = np.load(out / rec["npy"])
            ax.imshow(shaded_relief(dem, res=coarse_res, vmax=vmax))
            ax.set_title(f"{rec['idx']:03d}  {round(rec['lcc_km2'])}km²  f{rec['frac']:.2f}",
                         fontsize=6)
            ax.axis("off")
            
        for ax in axs[n:]:
            ax.axis("off")
            
        start_idx = i
        end_idx = i + n - 1
        fig.suptitle(f"coarse drafts ({start_idx}-{end_idx}) — id / largest-island area / single-island fraction",
                     fontsize=9)
        fig.tight_layout()
        
        if len(recs) <= batch_size:
            filename = "contact_sheet.png"
        else:
            filename = f"contact_sheet_{i // batch_size:03d}.png"
            
        fig.savefig(out / filename, dpi=120)
        plt.close(fig)


# ----------------------------- SR finishing ----------------------------- #
def finish_island(sr, dem_c, seed, rank, cfg, args, out, canvas_px):
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
    save_geotiff(dem, out / f"{tag}.tif", res, args)
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
    draft_dir = Path(draft_dir)
    man_path = draft_dir / "coarse_manifest.json"
    if man_path.exists():
        recs = json.loads(man_path.read_text())
    else:
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
    
    ap.add_argument("--batch-size", type=int, default=1, help="batch size for coarse generation (speed up)")
    ap.add_argument("--coarse-px", type=int, default=0, help="override coarse resolution for faster drafting")
    
    ap.add_argument("--keep", type=int, default=4, help="how many to SR + save (default mode)")
    ap.add_argument("--target-km2", type=float, default=5000.0)
    ap.add_argument("--tol-km2", type=float, default=1000.0)
    ap.add_argument("--canvas-px", type=int, default=0, help="override final canvas")
    ap.add_argument("--coarse-steps", type=int, default=0)
    ap.add_argument("--sr-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/generated")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--coarse-only", action="store_true",
                    help="STAGE 1 ONLY: mass-produce coarse drafts (no SR) for browsing")
    ap.add_argument("--complete", default="",
                    help="STAGE 2: finish coarse drafts from this --coarse-only dir via SR")
    ap.add_argument("--pick", nargs="+", default=None,
                    help="with --complete: id substrings to finish (e.g. 003 041); all if omitted")
    ap.add_argument("--hydro", action="store_true",
                    help="fill depressions so the island drains to the sea")
    ap.add_argument("--hydro-epsilon", type=float, default=1e-3,
                    help="m of drainage gradient across flats (0 = flat fill, faster)")
    ap.add_argument("--hydro-drainage", action="store_true",
                    help="also save a D8 river-network overlay (implies --hydro)")
    ap.add_argument("--center-coords", nargs=2, type=float, default=(32.234853, 129.372722),
                    help="Override the center coordinates (lat, lon) for the generated island")
    args = ap.parse_args()
    if args.hydro_drainage:
        args.hydro = True

    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    vmax = cfg["data"]["vmax"]
    canvas_px, coarse_px, coarse_res = grid_params(cfg, args)

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

    if not args.coarse_ckpt:
        raise SystemExit("need --coarse-ckpt (or use --complete with --sr-ckpt)")
    coarse, _, cs = load_stage(args.coarse_ckpt, device, use_ema=not args.no_ema)
    print(f"[coarse] loaded (step {cs}); sampling {args.n} drafts at "
          f"{coarse_px}px ({coarse_res:.0f} m/px)")
          
    out_dir_for_coarse = out if args.coarse_only else None
    cands = gen_coarse_candidates(coarse, cfg, args, coarse_px, coarse_res, out_dir=out_dir_for_coarse)
    
    lccs = sorted(c["lcc_km2"] for c in cands)
    print(f"[coarse] largest-island km2 dist: min={lccs[0]:.0f} "
          f"med={lccs[len(lccs)//2]:.0f} max={lccs[-1]:.0f}")

    if args.coarse_only:
        recs = []
        for c in cands:
            tag = c.get("tag") or f"coarse_{c['idx']:03d}_seed{c['seed']}_{round(c['lcc_km2'])}km2"
            recs.append({
                "tag": tag, "idx": c["idx"], "seed": c["seed"],
                "lcc_km2": round(c["lcc_km2"], 1), "frac": round(c["frac"], 2),
                "total_km2": round(c["total_km2"], 1), "npy": f"{tag}.npy"
            })
            
        recs.sort(key=lambda r: r["idx"])
        (out / "coarse_manifest.json").write_text(json.dumps(recs, indent=2))
        contact_sheet(recs, out, coarse_res, vmax, batch_size=100)
        
        sheet_name = "contact_sheet.png" if len(recs) <= 100 else "contact_sheet_*.png"
        print(f"[coarse-only] wrote {len(recs)} drafts + {sheet_name} to {out}\n"
              f"  browse {sheet_name}, then finish picks with:\n"
              f"  python src/generate.py --config {args.config} --sr-ckpt <sr.pt> "
              f"--complete {out} --pick <ids...> --out outputs/final")
        return

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