#!/usr/bin/env python3
"""Preprocess Copernicus DEM tiles -> a single Japan mosaic at ~60 m/px.

Steps:
  1. Reproject each raw 1deg tile (EPSG:4326, ~30 m) into a common metric grid
     (Lambert Conformal Conic centered on Japan) at TARGET_RES m, snapped to a
     shared global grid so tiles are pixel-aligned (seamless merge).
     Downsampling uses Resampling.average (anti-aliasing).
  2. Merge the grid-aligned reprojected tiles into one mosaic GeoTIFF (int16 m,
     DEFLATE-compressed; ocean=0 compresses to near nothing).
  3. Compute a land mask, elevation stats/percentiles, and total land area.

Rationale for reprojection (vs pure downsampling): terrain STATISTICS (slope,
roughness, radial power spectrum) require square pixels in METERS. EPSG:4326
pixels are anisotropic by cos(lat) (14-31% across 31-46N). A conformal projection
preserves local shape/scale -> correct stats. Downsampling to 60 m is done as
part of the same warp.

Vertical datum: EGM2008 geoid (orthometric). Ocean is encoded as 0 by Copernicus.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, reproject, transform_bounds

# Japan-centered Lambert Conformal Conic (metric, conformal -> preserves shape).
DST_CRS = ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=38 +lon_0=137 "
           "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
TARGET_RES = 60.0  # meters/pixel
# Union bbox of the 4 main islands (excl. Nansei): lon 128.5-146.5, lat 30.5-46.5
SRC_BBOX_4326 = (128.5, 30.5, 146.5, 46.5)


def compute_master_grid():
    """Define the global LCC grid origin and size covering all of Japan."""
    l, b, r, t = SRC_BBOX_4326
    xs, ys = [], []
    # densely sample the bbox edges and transform to LCC to get true extent
    for lon in np.linspace(l, r, 50):
        for lat in (b, t):
            xs.append(lon); ys.append(lat)
    for lat in np.linspace(b, t, 50):
        for lon in (l, r):
            xs.append(lon); ys.append(lat)
    X, Y = rasterio.warp.transform("EPSG:4326", DST_CRS, xs, ys)
    x_min = math.floor(min(X) / TARGET_RES) * TARGET_RES
    x_max = math.ceil(max(X) / TARGET_RES) * TARGET_RES
    y_min = math.floor(min(Y) / TARGET_RES) * TARGET_RES
    y_max = math.ceil(max(Y) / TARGET_RES) * TARGET_RES
    width = int(round((x_max - x_min) / TARGET_RES))
    height = int(round((y_max - y_min) / TARGET_RES))
    return x_min, y_max, width, height


def merge_4326(tiles):
    """Seamless merge of raw tiles in their native EPSG:4326 grid (they share the
    1deg grid, so placement is exact -> no internal seams). Returns (array, transform)."""
    srcs = [rasterio.open(t) for t in tiles]
    arr, transform = rio_merge(srcs, nodata=0.0, dtype="float32",
                               resampling=Resampling.nearest)
    for s in srcs:
        s.close()
    return arr[0], transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--sea-thresh", type=float, default=0.5,
                    help="meters above which a pixel counts as land")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(raw_dir.glob("*.tif"))
    print(f"[preprocess] {len(tiles)} raw tiles")
    x_min, y_max, W, H = compute_master_grid()
    out_transform = Affine(TARGET_RES, 0, x_min, 0, -TARGET_RES, y_max)
    print(f"[grid] LCC origin=({x_min:.0f},{y_max:.0f}) size={W}x{H} "
          f"({W*H/1e6:.0f} Mpx, {W*60/1000:.0f}x{H*60/1000:.0f} km)")

    # 1. seamless merge in native 4326 (single continuous source -> no tile seams)
    print("[merge] merging raw tiles in EPSG:4326 ...", flush=True)
    big, big_transform = merge_4326(tiles)
    big = np.nan_to_num(big, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[merge] 4326 mosaic {big.shape} {big.nbytes/1e9:.1f}GB", flush=True)

    # 2. SINGLE reproject of the continuous mosaic -> LCC @60m (no internal edges)
    print("[reproject] 4326 -> LCC @60m (average) ...", flush=True)
    dst = np.zeros((H, W), dtype=np.float32)
    reproject(source=big, destination=dst,
              src_transform=big_transform, src_crs="EPSG:4326",
              dst_transform=out_transform, dst_crs=DST_CRS,
              src_nodata=0.0, dst_nodata=0.0, resampling=Resampling.average,
              num_threads=8)
    del big
    mosaic = np.clip(np.nan_to_num(dst, nan=0.0), -500, 9000).astype(np.int16)
    del dst
    print(f"[reproject] mosaic shape={mosaic.shape} "
          f"{mosaic.nbytes/1e9:.2f}GB in-mem", flush=True)

    mosaic_path = out_dir / "japan_dem_60m.tif"
    profile = {
        "driver": "GTiff", "dtype": "int16", "count": 1,
        "height": mosaic.shape[0], "width": mosaic.shape[1],
        "crs": DST_CRS, "transform": out_transform, "nodata": -32768,
        "compress": "deflate", "predictor": 2, "tiled": True,
        "blockxsize": 512, "blockysize": 512,
    }
    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(mosaic, 1)
    print(f"[write] {mosaic_path} ({mosaic_path.stat().st_size/1e9:.2f}GB on disk)")

    # land mask + stats
    land = mosaic > args.sea_thresh
    land_px = int(land.sum())
    land_km2 = land_px * (TARGET_RES ** 2) / 1e6
    land_vals = mosaic[land].astype(np.float32)
    pct = np.percentile(land_vals, [0, 1, 50, 90, 99, 99.9, 100]).tolist()
    stats = {
        "crs": DST_CRS, "res_m": TARGET_RES,
        "mosaic_shape": list(mosaic.shape),
        "transform": list(out_transform)[:6],
        "sea_thresh_m": args.sea_thresh,
        "land_pixels": land_px,
        "total_land_km2": round(land_km2, 1),
        "land_fraction": round(land_px / mosaic.size, 4),
        "elev_percentiles_m": {k: round(v, 1) for k, v in
                               zip(["min","p1","p50","p90","p99","p99.9","max"], pct)},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    # save land mask as compact uint8
    with rasterio.open(out_dir / "japan_landmask_60m.tif", "w",
                       **{**profile, "dtype": "uint8", "nodata": 0}) as dst:
        dst.write(land.astype(np.uint8), 1)
    print("[stats]", json.dumps(stats, indent=2))
    print("[done] preprocessing complete")


if __name__ == "__main__":
    main()
