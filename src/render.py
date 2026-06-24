#!/usr/bin/env python3
"""Render DEMs to hillshade and color-relief PNGs for visual inspection.

Used both for visual self-critique (Claude opens these PNGs) and reports.
Works on elevation arrays in METERS (float), with ocean ~ 0.
"""
from pathlib import Path

import numpy as np


def hillshade(dem_m, res=60.0, azimuth=315.0, altitude=45.0, z_factor=1.0):
    """Standard Horn hillshade. dem_m in meters. Returns uint8 [0,255]."""
    dem = dem_m.astype(np.float32)
    dy, dx = np.gradient(dem * z_factor, res, res)  # d/drow, d/dcol
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)  # GIS aspect convention
    az = np.deg2rad(360.0 - azimuth + 90.0)
    alt = np.deg2rad(altitude)
    shaded = (np.sin(alt) * np.sin(slope) +
              np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    shaded = np.clip(shaded, 0, 1)
    return (shaded * 255).astype(np.uint8)


def _terrain_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # sea -> coast -> green lowlands -> brown -> white peaks
    colors = [
        (0.00, (0.10, 0.22, 0.42)),   # deep sea
        (0.04, (0.22, 0.42, 0.62)),   # shallow sea
        (0.0401, (0.18, 0.50, 0.28)), # coast/land start (sharp)
        (0.18, (0.36, 0.62, 0.30)),   # lowland green
        (0.42, (0.78, 0.74, 0.45)),   # hills tan
        (0.68, (0.58, 0.44, 0.30)),   # mountain brown
        (0.88, (0.80, 0.78, 0.76)),   # rock grey
        (1.00, (1.00, 1.00, 1.00)),   # snow
    ]
    stops = [c[0] for c in colors]
    cols = [c[1] for c in colors]
    return LinearSegmentedColormap.from_list("jp_terrain", list(zip(stops, cols)))


def color_relief(dem_m, vmax=3776.0, sea_level=0.0):
    """Color-relief RGB uint8. Maps sea (<=sea_level) to blue band, land by elevation."""
    cmap = _terrain_cmap()
    dem = dem_m.astype(np.float32)
    norm = np.empty_like(dem)
    sea = dem <= sea_level
    # sea band occupies [0, 0.04]; land [0.04, 1]
    sea_floor = -50.0
    norm[sea] = 0.04 * (1 - np.clip((sea_level - dem[sea]) / (sea_level - sea_floor), 0, 1))
    norm[~sea] = 0.04 + 0.96 * np.clip(dem[~sea] / vmax, 0, 1)
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return rgb


def shaded_relief(dem_m, res=60.0, vmax=3776.0, **hs):
    """Blend color-relief with hillshade for a natural look. Returns RGB uint8."""
    rgb = color_relief(dem_m, vmax=vmax).astype(np.float32) / 255.0
    hs_arr = hillshade(dem_m, res=res, **hs).astype(np.float32) / 255.0
    # multiply blend, lifted so it doesn't go fully black
    shade = 0.45 + 0.55 * hs_arr
    out = np.clip(rgb * shade[..., None], 0, 1)
    return (out * 255).astype(np.uint8)


def save_png(arr, path):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
    return path


def grid_montage(dems, res=60.0, vmax=3776.0, cols=4, pad=4, mode="shaded"):
    """Tile multiple DEMs into a single montage image (RGB uint8)."""
    tiles = []
    for d in dems:
        if mode == "hillshade":
            t = np.stack([hillshade(d, res=res)] * 3, axis=-1)
        elif mode == "color":
            t = color_relief(d, vmax=vmax)
        else:
            t = shaded_relief(d, res=res, vmax=vmax)
        tiles.append(t)
    n = len(tiles)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    canvas = np.full((rows * h + (rows + 1) * pad,
                      cols * w + (cols + 1) * pad, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = t
    return canvas


if __name__ == "__main__":
    # smoke test on synthetic terrain
    import numpy as np
    yy, xx = np.mgrid[0:256, 0:256]
    dem = (np.sin(xx / 20) * np.cos(yy / 25) * 400 + 600
           - np.hypot(xx - 128, yy - 128) * 3)
    dem = np.maximum(dem, 0)
    save_png(shaded_relief(dem), "outputs/_render_smoke.png")
    print("wrote outputs/_render_smoke.png")
