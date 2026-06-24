#!/usr/bin/env python3
"""Download Copernicus DEM GLO-30 tiles covering Japan's 4 main islands.

Data source: Copernicus DEM GLO-30 (30 m) on AWS Open Data, public bucket
`copernicus-dem-30m` (no auth). Tiles are 1deg x 1deg COG GeoTIFFs named by their
SW corner, e.g. Copernicus_DSM_COG_10_N35_00_E138_00_DEM. Vertical datum: EGM2008
geoid (orthometric, sea ~ 0). Horizontal: WGS84 / EPSG:4326.

Selection: per-island integer-degree bounding boxes covering Honshu, Hokkaido,
Shikoku, Kyushu, EXCLUDING the Nansei/Ryukyu islands (everything south of ~31N
and the SW arc). Only land-touching tiles present in the bucket's tileList.txt
are kept (pure-ocean tiles are absent from the list).

License: Copernicus DEM is free to use with attribution
("Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus
Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union
and ESA; all rights reserved").
"""
import argparse
import concurrent.futures as cf
import os
import re
import sys
import time
from pathlib import Path

import requests

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
TILELIST_URL = f"{BUCKET}/tileList.txt"

# Per-island SW-corner integer-degree tile boxes (lat range, lon range), inclusive.
ISLAND_BOXES = {
    "Hokkaido": (range(41, 46), range(139, 146)),  # 41-46N, 139-146E
    "Honshu":   (range(33, 42), range(130, 143)),  # 33-42N, 130-143E
    "Shikoku":  (range(32, 35), range(132, 135)),  # 32-35N, 132-135E
    "Kyushu":   (range(31, 34), range(129, 132)),  # 31-34N, 129-132E  (excludes Nansei < 31N)
}


def fetch_tilelist(cache: Path) -> set:
    if cache.exists() and cache.stat().st_size > 1000:
        return set(l.strip() for l in cache.read_text().splitlines() if l.strip())
    print(f"[tilelist] downloading {TILELIST_URL}")
    r = requests.get(TILELIST_URL, timeout=60)
    r.raise_for_status()
    cache.write_text(r.text)
    return set(l.strip() for l in r.text.splitlines() if l.strip())


def select_tiles(available: set) -> list:
    sel = set()
    for _, (lats, lons) in ISLAND_BOXES.items():
        for la in lats:
            for lo in lons:
                t = f"Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM"
                if t in available:
                    sel.add(t)
    return sorted(sel)


def tile_url(name: str) -> str:
    # objects live under <name>/<name>.tif
    return f"{BUCKET}/{name}/{name}.tif"


def download_one(name: str, out_dir: Path, retries: int = 4) -> tuple:
    dst = out_dir / f"{name}.tif"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return (name, "skip", dst.stat().st_size)
    url = tile_url(name)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dst.with_suffix(".tif.part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dst)
            return (name, "ok", dst.stat().st_size)
        except Exception as e:  # noqa
            if attempt == retries - 1:
                return (name, f"FAIL:{e}", 0)
            time.sleep(2 * (attempt + 1))
    return (name, "FAIL", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="limit number of tiles (PoC)")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir.parent / "tileList.txt"

    available = fetch_tilelist(cache)
    tiles = select_tiles(available)
    if args.limit:
        # For PoC: take a spread (every Nth) so we cover varied terrain, not one corner.
        step = max(1, len(tiles) // args.limit)
        tiles = tiles[::step][: args.limit]
    print(f"[select] {len(tiles)} tiles, est {len(tiles)*40/1024:.1f} GB raw")
    (out_dir.parent / "selected_tiles.txt").write_text("\n".join(tiles) + "\n")
    if args.list_only:
        for t in tiles:
            print(t)
        return

    t0 = time.time()
    done = ok = skip = fail = 0
    total_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, t, out_dir): t for t in tiles}
        for fut in cf.as_completed(futs):
            name, status, size = fut.result()
            done += 1
            total_bytes += size
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  [{done}/{len(tiles)}] {name}: {status}", file=sys.stderr)
            if done % 10 == 0 or done == len(tiles):
                print(f"  [{done}/{len(tiles)}] ok={ok} skip={skip} fail={fail} "
                      f"{total_bytes/1e9:.2f}GB {time.time()-t0:.0f}s", flush=True)
    print(f"[done] ok={ok} skip={skip} fail={fail} total={total_bytes/1e9:.2f}GB "
          f"in {time.time()-t0:.0f}s")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
