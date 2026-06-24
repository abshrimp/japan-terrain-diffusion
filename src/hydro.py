#!/usr/bin/env python3
"""Optional hydrological conditioning of a generated DEM.

`fill_depressions` removes spurious interior sinks/pits so that every land cell
drains to the sea (the outlet) — the standard "depression filling" used to make a
DEM hydrologically consistent (e.g. HydroSHEDS-style conditioning). Uses grayscale
morphological reconstruction (Robinson/Vincent), with the SEA (elevation <= sea_level)
and the grid border as outlets. Fast and exact (no internal sinks remain).

`flow_accumulation` computes D8 flow accumulation on the conditioned DEM — used to
visualize/verify the drainage network (rivers concentrate downslope to the coast).

`count_sinks` counts remaining interior local minima — used to verify conditioning.
"""
import numpy as np


def count_sinks(dem, sea_level=0.0):
    """Number of interior land local-minima (cells strictly below all 8 neighbours).
    On a hydrologically consistent surface this should be ~0 over land."""
    d = dem.astype(np.float64)
    H, W = d.shape
    is_min = np.ones((H, W), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.full((H, W), np.inf)
            ys = slice(max(0, dy), H + min(0, dy)); xs = slice(max(0, dx), W + min(0, dx))
            yd = slice(max(0, -dy), H + min(0, -dy)); xd = slice(max(0, -dx), W + min(0, -dx))
            sh[yd, xd] = d[ys, xs]
            is_min &= d < sh           # strictly lower than this neighbour
    land = d > sea_level
    return int((is_min & land).sum())


def _priority_flood_epsilon(base, sea_level, epsilon):
    """Barnes (2014) priority-flood with an epsilon gradient: fills depressions AND
    imposes a tiny monotonic slope across flats -> surface drains to the sea everywhere
    (no sinks, no flats). Sea (<= sea_level) + grid border are outlets."""
    import heapq
    from scipy.ndimage import binary_dilation
    H, W = base.shape
    filled = base.copy()
    closed = np.zeros((H, W), bool)
    land = base > sea_level
    sea = ~land
    closed |= sea                       # sea cells are outlets (kept, never filled)
    closed[0, :] = closed[-1, :] = closed[:, 0] = closed[:, -1] = True
    # seed only the active frontier: coastline (sea touching land) + grid border
    coast = sea & binary_dilation(land)
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    pq = [(float(base[y, x]), int(y), int(x)) for y, x in zip(*np.where(coast | border))]
    heapq.heapify(pq)
    push, pop = heapq.heappush, heapq.heappop
    NB = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while pq:
        e, y, x = pop(pq)
        for dy, dx in NB:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not closed[ny, nx]:
                closed[ny, nx] = True
                ne = base[ny, nx]
                if ne <= e + epsilon:
                    ne = e + epsilon       # raise into flat/pit -> drainable gradient
                    filled[ny, nx] = ne
                push(pq, (ne, ny, nx))
    return filled


def fill_depressions(dem, sea_level=0.0, epsilon=0.0):
    """Hydrological conditioning: fill depressions so every land cell drains to the sea.
    epsilon=0  -> flat fill via morphological reconstruction (fast; flats remain).
    epsilon>0  -> priority-flood with an epsilon (m) gradient (slower; fully drainable,
                  no sinks AND no flats -> clean D8 drainage networks). Try ~1e-3.
    Sea (<= sea_level) and the grid border are outlets. Returns (filled_dem, stats)."""
    base = np.asarray(dem, np.float64)
    if epsilon and epsilon > 0:
        filled = _priority_flood_epsilon(base, sea_level, float(epsilon))
    else:
        from skimage.morphology import reconstruction
        seed = np.full_like(base, base.max() + 1.0)
        outlet = base <= sea_level
        seed[outlet] = base[outlet]
        seed[0, :] = base[0, :]; seed[-1, :] = base[-1, :]
        seed[:, 0] = base[:, 0]; seed[:, -1] = base[:, -1]
        filled = reconstruction(seed, base, method="erosion")
    diff = filled - base
    raised = diff > 1e-6
    land = base > sea_level
    nland = max(int(land.sum()), 1)
    stats = {
        "cells_raised": int(raised.sum()),
        "frac_land_raised": round(float(raised.sum()) / nland, 4),
        "max_fill_m": round(float(diff.max()), 2),
        "mean_fill_m": round(float(diff[raised].mean()) if raised.any() else 0.0, 3),
        "sinks_before": count_sinks(base, sea_level),
        "sinks_after": count_sinks(filled, sea_level),
    }
    return filled.astype(np.float32), stats


def flow_accumulation(dem, sea_level=0.0):
    """D8 flow accumulation (cells drained through each cell). Returns float array.
    Best run on a depression-filled DEM. Flats (no lower neighbour) terminate flow."""
    d = dem.astype(np.float64)
    H, W = d.shape
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = [2 ** 0.5, 1, 2 ** 0.5, 1, 1, 2 ** 0.5, 1, 2 ** 0.5]
    best_slope = np.zeros((H, W))
    best_dir = np.full((H, W), -1, np.int8)
    for k, (dy, dx) in enumerate(nb):
        sh = np.full((H, W), np.inf)
        ys = slice(max(0, dy), H + min(0, dy)); xs = slice(max(0, dx), W + min(0, dx))
        yd = slice(max(0, -dy), H + min(0, -dy)); xd = slice(max(0, -dx), W + min(0, -dx))
        sh[yd, xd] = d[ys, xs]
        slope = (d - sh) / dist[k]
        better = slope > best_slope
        best_slope[better] = slope[better]
        best_dir[better] = k
    acc = np.ones(H * W, np.float64)
    bd = best_dir.ravel()
    order = np.argsort(d.ravel(), kind="stable")[::-1]   # high -> low
    for idx in order:
        k = bd[idx]
        if k < 0:
            continue
        dy, dx = nb[k]
        y, x = divmod(int(idx), W)
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            acc[ny * W + nx] += acc[idx]
    return acc.reshape(H, W)


def drainage_overlay(dem, acc, sea_level=0.0, river_pct=96.0):
    """RGB uint8 visual: shaded relief with high-accumulation cells drawn as rivers.
    Rivers are dilated 1px and tinted blue (stronger for larger streams) for legibility."""
    from render import shaded_relief
    from scipy.ndimage import binary_dilation
    rgb = shaded_relief(dem, vmax=3776.0).astype(np.float32)
    land = dem > sea_level
    la = np.log1p(acc)
    thr = np.percentile(la[land], river_pct) if land.any() else la.max()
    rivers = (la >= thr) & land
    rivers = binary_dilation(rivers, iterations=1) & land     # 1px width for visibility
    inten = np.clip((la - thr) / (la.max() - thr + 1e-9), 0.45, 1.0)
    rgb[rivers] = ((1 - inten[rivers, None]) * rgb[rivers]
                   + inten[rivers, None] * np.array([20, 80, 230], np.float32))
    return np.clip(rgb, 0, 255).astype(np.uint8)
