#!/usr/bin/env python3
"""Generate fictional-island DEMs with the trained cascade.

Pipeline (NO spatial tiling, NO seam blending):
  1. Coarse EDM samples the FULL canvas at coarse resolution (one un-tiled pass/step).
  2. Coarse map is bicubically upsampled to the final canvas size -> SR condition.
  3. SR EDM refines the ENTIRE canvas in a single fully-convolutional pass/step
     (translation-equivariant UNet trained on patches, run on the whole canvas).

Post-selection: generate N coarse candidates, keep those whose land area is within
tolerance of the target (~5000 km2), then run SR only on the keepers.

  python src/generate.py --config configs/phase1.yaml \
      --coarse-ckpt checkpoints/phase1/coarse/latest.pt \
      --sr-ckpt checkpoints/phase1/sr/latest.pt --n 16 --target-km2 5000
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
    """Area (km2) of the LARGEST connected land mass, and its fraction of all land.
    Used so post-selection picks a single coherent island, not an archipelago."""
    from scipy.ndimage import label
    land = dem_m > sea_thresh
    lab, n = label(land)
    if n == 0:
        return 0.0, 0.0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--coarse-ckpt", required=True)
    ap.add_argument("--sr-ckpt", default="")
    ap.add_argument("--n", type=int, default=16, help="coarse candidates")
    ap.add_argument("--keep", type=int, default=4, help="how many to SR + save")
    ap.add_argument("--target-km2", type=float, default=5000.0)
    ap.add_argument("--tol-km2", type=float, default=1000.0)
    ap.add_argument("--canvas-px", type=int, default=0, help="override final canvas")
    ap.add_argument("--coarse-steps", type=int, default=0)
    ap.add_argument("--sr-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/generated")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--hydro", action="store_true",
                    help="hydrological conditioning: fill depressions so the island "
                         "drains to the sea (removes spurious sinks)")
    ap.add_argument("--hydro-epsilon", type=float, default=1e-3,
                    help="m of drainage gradient imposed across flats (priority-flood). "
                         "Default 1e-3 (fully drainable). Set 0 for flat fill (faster).")
    ap.add_argument("--hydro-drainage", action="store_true",
                    help="also compute D8 flow accumulation and save a river overlay "
                         "(implies --hydro)")
    args = ap.parse_args()
    if args.hydro_drainage:
        args.hydro = True

    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    g = cfg["geometry"]
    vmax = cfg["data"]["vmax"]
    res = cfg["data"]["res_m"]
    nm = cfg["data"].get("norm", "sqrt")
    sea = cfg["data"].get("sea_thresh", 0.5)
    canvas_px = args.canvas_px or g["canvas_px"]
    coarse_px = canvas_px // (g["canvas_px"] // g["coarse_px"])
    coarse_res = res * (canvas_px // coarse_px)
    cstep = args.coarse_steps or cfg["coarse"]["sample"]["steps"]
    sstep = args.sr_steps or cfg["sr"]["sample"]["steps"]

    coarse, _, cs = load_stage(args.coarse_ckpt, device, use_ema=not args.no_ema)
    print(f"[coarse] loaded (step {cs}); generating {args.n} candidates "
          f"at {coarse_px}px ({coarse_res:.0f} m/px)")

    # 1. coarse candidates — score by LARGEST connected island area (single-island
    #    goal), preferring compact islands (largest-cc / total land high).
    cands = []
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(args.n):
            x = coarse.sample((1, 1, coarse_px, coarse_px), device,
                              steps=cstep, seed=args.seed + i)
            dem_c = denormalize(x[0, 0].float().cpu().numpy(), vmax, nm)
            lcc_km2, frac = largest_component_km2(dem_c, coarse_res)
            # score: distance of largest island to target, penalize fragmentation
            score = abs(lcc_km2 - args.target_km2) + (1.0 - frac) * args.target_km2 * 0.3
            cands.append({"score": score, "lcc_km2": lcc_km2, "frac": frac,
                          "total_km2": land_km2(dem_c, coarse_res),
                          "seed": args.seed + i, "dem_c": dem_c})
    cands.sort(key=lambda c: c["score"])
    lccs = sorted(c["lcc_km2"] for c in cands)
    print(f"[coarse] largest-island km2 dist: min={lccs[0]:.0f} "
          f"med={lccs[len(lccs)//2]:.0f} max={lccs[-1]:.0f}")
    keepers = [c for c in cands
               if abs(c["lcc_km2"] - args.target_km2) <= args.tol_km2][:args.keep]
    if not keepers:
        keepers = cands[:args.keep]
        print("[warn] none within tolerance; taking best-scored")
    print(f"[select] kept {len(keepers)}: largest-island km2="
          f"{[round(c['lcc_km2']) for c in keepers]} single-island-frac="
          f"{[round(c['frac'],2) for c in keepers]}")

    sr = None
    if args.sr_ckpt:
        sr, _, ss = load_stage(args.sr_ckpt, device, use_ema=not args.no_ema)
        print(f"[sr] loaded (step {ss}); refining full {canvas_px}px canvas in ONE pass")

    results = []
    for rank, cand in enumerate(keepers):
        km2c, seed, dem_c = cand["lcc_km2"], cand["seed"], cand["dem_c"]
        # 2. upsample coarse -> condition (in normalized space, matching training)
        xc = torch.from_numpy(normalize(dem_c, vmax, nm).astype(np.float32))[None, None].to(device)
        cond = F.interpolate(xc, size=(canvas_px, canvas_px), mode="bicubic",
                             align_corners=False)
        if sr is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                x = sr.sample((1, 1, canvas_px, canvas_px), device, cond=cond,
                              steps=sstep, seed=seed)
            dem = denormalize(x[0, 0].float().cpu().numpy(), vmax, nm)
        else:
            dem = denormalize(cond[0, 0].float().cpu().numpy(), vmax, nm)
        # optional hydrological conditioning: fill depressions so the island drains to sea
        hstats = None
        if args.hydro:
            dem, hstats = fill_depressions(dem, sea_level=sea, epsilon=args.hydro_epsilon)
            print(f"      [hydro] filled {hstats['cells_raised']} cells "
                  f"({hstats['frac_land_raised']*100:.1f}% of land, mean {hstats['mean_fill_m']:.1f} m, "
                  f"max {hstats['max_fill_m']:.0f} m); strict sinks {hstats['sinks_before']}->{hstats['sinks_after']}")
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
               "total_land_km2": round(total_km2, 1),
               "single_island_frac": round(frac, 2),
               "coarse_island_km2": round(km2c, 1),
               "elev_max_m": round(float(dem.max()), 1)}
        if hstats:
            rec["hydro"] = hstats
        results.append(rec)
        print(f"  [{rank}] {tag}: island {lcc_km2:.0f} km2 "
              f"(total land {total_km2:.0f}, frac {frac:.2f}), max {dem.max():.0f} m")

    (out / "manifest.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {len(results)} islands to {out}")


if __name__ == "__main__":
    main()
