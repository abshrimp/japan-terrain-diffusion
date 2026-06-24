#!/usr/bin/env python3
"""On-the-fly terrain crop dataset sampled from the in-RAM Japan mosaic.

No per-crop disk cost; infinite augmentation. Land-aware sampling uses an integral
image of the land mask for O(1) land-fraction queries (rejection sampling).

Modes:
  - 'coarse': sample a `canvas_px` window, average-pool to `coarse_px`. Returns the
    normalized coarse full-canvas crop. Prefers mixed land/sea windows (island-like).
  - 'sr': sample a `patch_px` high-res (60 m) window with enough land; returns
    (hi, cond) where cond = window downsampled by `sr_factor` then upsampled back
    (mimics the coarse stage output the SR model sees at inference).

Normalization: norm = clip(elev, 0, vmax)/vmax * 2 - 1  -> sea(0) maps to -1.
Augmentation: random dihedral (flip + 90 deg rotations); terrain is ~orientation-free.
"""
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def normalize(elev_m, vmax=3776.0, mode="sqrt"):
    """Map elevation (m) -> [-1, 1]. sea(0) -> -1.
    sqrt expands the low-elevation band (Japan's hypsometry is right-skewed)."""
    h = np.clip(elev_m, 0.0, vmax) / vmax
    if mode == "sqrt":
        return 2.0 * np.sqrt(h) - 1.0
    elif mode == "log":
        return 2.0 * np.log1p(h * (np.e - 1)) - 1.0  # log1p scaled, h in[0,1]->[0,1]
    else:  # linear
        return h * 2.0 - 1.0


def denormalize(x, vmax=3776.0, mode="sqrt"):
    x = np.asarray(x, dtype=np.float32)
    u = (x + 1.0) * 0.5  # -> [0,1]
    u = np.clip(u, 0.0, 1.0)
    if mode == "sqrt":
        h = u * u
    elif mode == "log":
        h = (np.expm1(u) ) / (np.e - 1)
    else:
        h = u
    return h * vmax


def _avgpool2d(a, f):
    h, w = a.shape
    h2, w2 = h // f, w // f
    return a[:h2 * f, :w2 * f].reshape(h2, f, w2, f).mean(axis=(1, 3))


class TerrainMosaic:
    """Loads the mosaic + land mask into RAM and builds an integral image."""
    def __init__(self, mosaic_path, mask_path, sea_thresh=0.5):
        with rasterio.open(mosaic_path) as ds:
            self.dem = ds.read(1).astype(np.float32)  # meters, (H,W)
            self.dem[self.dem == ds.nodata] = 0.0
        try:
            with rasterio.open(mask_path) as ds:
                self.mask = (ds.read(1) > 0).astype(np.uint8)
        except Exception:
            self.mask = (self.dem > sea_thresh).astype(np.uint8)
        self.dem = np.maximum(self.dem, 0.0)  # ocean/below-sea -> 0
        self.H, self.W = self.dem.shape
        # integral image for O(1) land-fraction queries.
        # NOTE: cumsum of uint8 would overflow; force int32 (land px < 2.1e9).
        ii = np.zeros((self.H + 1, self.W + 1), np.int32)
        ii[1:, 1:] = self.mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
        self.ii = ii
        print(f"[mosaic] {self.H}x{self.W} land_frac={self.mask.mean():.3f}")

    def land_fraction(self, y, x, s):
        ii = self.ii
        total = (ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x])
        return total / float(s * s)

    def sample_window(self, size, lo=0.1, hi=0.9, max_try=200, rng=None):
        rng = rng or np.random
        for _ in range(max_try):
            y = rng.randint(0, self.H - size)
            x = rng.randint(0, self.W - size)
            frac = self.land_fraction(y, x, size)
            if lo <= frac <= hi:
                return self.dem[y:y + size, x:x + size]
        # fallback: best-effort, accept whatever (>0 land)
        return self.dem[y:y + size, x:x + size]


def _augment(a, rng):
    k = rng.randint(0, 4)
    a = np.rot90(a, k)
    if rng.rand() < 0.5:
        a = np.fliplr(a)
    if rng.rand() < 0.5:
        a = np.flipud(a)
    return np.ascontiguousarray(a)


class CoarseDataset(Dataset):
    def __init__(self, mosaic: TerrainMosaic, canvas_px, coarse_px, vmax=3776.0,
                 land_lo=0.10, land_hi=0.85, length=20000, norm="sqrt"):
        self.m = mosaic
        self.canvas_px = canvas_px
        self.coarse_px = coarse_px
        self.f = canvas_px // coarse_px
        self.vmax = vmax
        self.norm = norm
        self.land_lo, self.land_hi = land_lo, land_hi
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.RandomState((idx * 2654435761) % (2**32))
        win = self.m.sample_window(self.canvas_px, self.land_lo, self.land_hi, rng=rng)
        coarse = _avgpool2d(win, self.f)
        coarse = _augment(coarse, rng)
        x = normalize(coarse, self.vmax, self.norm).astype(np.float32)
        return torch.from_numpy(x)[None]  # (1,H,W)


class SRDataset(Dataset):
    def __init__(self, mosaic: TerrainMosaic, patch_px, sr_factor, vmax=3776.0,
                 land_lo=0.30, land_hi=1.0, length=40000, norm="sqrt"):
        self.m = mosaic
        self.patch_px = patch_px
        self.sr_factor = sr_factor
        self.vmax = vmax
        self.norm = norm
        self.land_lo, self.land_hi = land_lo, land_hi
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.RandomState((idx * 40503 + 12345) % (2**32))
        hi = self.m.sample_window(self.patch_px, self.land_lo, self.land_hi, rng=rng)
        hi = _augment(hi, rng)
        lo = _avgpool2d(hi, self.sr_factor)
        # cond mimics the coarse-stage output the SR model sees at inference:
        # downsample then bicubic-upsample back to patch size, in normalized space.
        lo_t = torch.from_numpy(normalize(lo, self.vmax, self.norm).astype(np.float32))[None, None]
        cond = torch.nn.functional.interpolate(
            lo_t, size=(self.patch_px, self.patch_px), mode="bicubic",
            align_corners=False)[0]
        hi_n = torch.from_numpy(normalize(hi, self.vmax, self.norm).astype(np.float32))[None]
        return hi_n, cond  # (1,H,W), (1,H,W)
