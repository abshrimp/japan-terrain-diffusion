#!/usr/bin/env python3
"""Statistical metrics comparing generated vs real terrain distributions.

All inputs are elevation arrays in METERS (ocean ~ 0). Comparisons operate on
SETS of crops (real_set, fake_set), each a list/array of 2D float arrays of equal
shape.

Metrics:
  - elevation histogram: KS statistic + Wasserstein (Earth-Mover) distance
  - slope (deg) distribution: KS + Wasserstein
  - roughness (local std): KS + Wasserstein
  - hypsometric curve: mean curve + L1 area difference
  - radially-averaged power spectral density: slope beta (P(k) ~ k^-beta),
    fractal dimension D = 4 - beta/2, and log-log curve L2 distance
  - Sliced-Wasserstein Distance (SWD) over a Laplacian pyramid (PGGAN-style),
    pretrained-free image-distribution distance suited to single-channel terrain
"""
import numpy as np
from scipy import stats


# ----------------------------- pixel-wise dists ----------------------------- #
def slope_deg(dem, res=60.0):
    dy, dx = np.gradient(dem.astype(np.float32), res, res)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def roughness(dem, win=5):
    """Local std of elevation in win x win neighborhood (uniform filter)."""
    from scipy.ndimage import uniform_filter
    d = dem.astype(np.float32)
    mean = uniform_filter(d, win)
    sq = uniform_filter(d * d, win)
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def _pool_values(dems, fn, land_only=True, sea_thresh=0.5, max_per=200_000):
    vals = []
    for d in dems:
        v = fn(d)
        if land_only:
            v = v[d > sea_thresh]
        else:
            v = v.ravel()
        if v.size > max_per:
            idx = np.random.choice(v.size, max_per, replace=False)
            v = v[idx]
        vals.append(v)
    return np.concatenate(vals) if vals else np.array([])


def dist_compare(real_vals, fake_vals):
    if real_vals.size == 0 or fake_vals.size == 0:
        return {"ks": float("nan"), "wasserstein": float("nan")}
    ks = stats.ks_2samp(real_vals, fake_vals).statistic
    wd = stats.wasserstein_distance(real_vals, fake_vals)
    return {"ks": float(ks), "wasserstein": float(wd)}


# ----------------------------- hypsometric ----------------------------- #
def hypsometric_curve(dem, sea_thresh=0.5, n=64):
    """Normalized hypsometric curve: for relative-area a in [0,1], relative height.
    Returns h(a): fraction of land area at or above each normalized elevation."""
    land = dem[dem > sea_thresh]
    if land.size < 10:
        return np.zeros(n)
    hmin, hmax = land.min(), land.max()
    if hmax - hmin < 1e-6:
        return np.zeros(n)
    hn = (land - hmin) / (hmax - hmin)
    # for each height level, fraction of area above it
    levels = np.linspace(0, 1, n)
    curve = np.array([(hn >= lv).mean() for lv in levels])
    return curve


def hypsometric_compare(real_dems, fake_dems, n=64):
    r = np.mean([hypsometric_curve(d, n=n) for d in real_dems], axis=0)
    f = np.mean([hypsometric_curve(d, n=n) for d in fake_dems], axis=0)
    l1 = float(np.abs(r - f).mean())
    # hypsometric integral (HI) = area under curve
    return {"hypso_l1": l1, "HI_real": float(r.mean()), "HI_fake": float(f.mean()),
            "curve_real": r.tolist(), "curve_fake": f.tolist()}


# ----------------------------- radial PSD ----------------------------- #
def radial_psd(dem, res=60.0):
    """Radially-averaged 2D power spectrum (average over annuli).
    Returns (k [cycles/m], P(k)). Detrends (remove mean) + Hann window."""
    d = dem.astype(np.float32)
    d = d - d.mean()
    h, w = d.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    d = d * (wy * wx)
    F = np.fft.fftshift(np.fft.fft2(d))
    P2 = np.abs(F) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx).astype(np.int32)
    rmax = min(cy, cx)
    tbin = np.bincount(r.ravel(), P2.ravel())
    nbin = np.bincount(r.ravel())
    radial = tbin[:rmax] / np.maximum(nbin[:rmax], 1)  # average per annulus
    k = np.arange(rmax) / (max(h, w) * res)  # cycles/m
    return k[1:], radial[1:]  # drop DC


def psd_slope(k, P, fmin_frac=0.04, fmax_frac=0.45):
    """Fit P(k) ~ k^-beta on a log-log band. Returns beta, fractal D=4-beta/2."""
    kmax = k.max()
    band = (k >= fmin_frac * kmax) & (k <= fmax_frac * kmax) & (P > 0)
    if band.sum() < 5:
        return float("nan"), float("nan")
    lk, lp = np.log(k[band]), np.log(P[band])
    A = np.vstack([lk, np.ones_like(lk)]).T
    slope, _ = np.linalg.lstsq(A, lp, rcond=None)[0]
    beta = -slope
    D = 4.0 - beta / 2.0
    return float(beta), float(D)


def psd_compare(real_dems, fake_dems, res=60.0):
    def mean_curve(dems):
        ks, Ps = None, []
        for d in dems:
            k, P = radial_psd(d, res=res)
            ks = k
            Ps.append(np.log(np.maximum(P, 1e-12)))
        return ks, np.mean(Ps, axis=0)
    kr, lpr = mean_curve(real_dems)
    kf, lpf = mean_curve(fake_dems)
    m = min(len(kr), len(kf))
    curve_l2 = float(np.sqrt(np.mean((lpr[:m] - lpf[:m]) ** 2)))
    br, Dr = psd_slope(kr, np.exp(lpr))
    bf, Df = psd_slope(kf, np.exp(lpf))
    return {"beta_real": br, "beta_fake": bf, "D_real": Dr, "D_fake": Df,
            "beta_abs_err": float(abs(br - bf)) if np.isfinite(br) and np.isfinite(bf) else float("nan"),
            "psd_logcurve_l2": curve_l2,
            "k": kr[:m].tolist(), "logP_real": lpr[:m].tolist(), "logP_fake": lpf[:m].tolist()}


# ----------------------------- SWD (Laplacian pyramid) ----------------------------- #
def _gaussian_blur(x):
    # separable 5-tap binomial, x: (N,H,W)
    k = np.array([1, 4, 6, 4, 1], np.float32) / 16
    from scipy.ndimage import convolve1d
    x = convolve1d(x, k, axis=1, mode="reflect")
    x = convolve1d(x, k, axis=2, mode="reflect")
    return x


def _laplacian_pyramid(imgs, levels=4):
    """imgs: (N,H,W) float normalized ~[-1,1]. Returns list of level arrays."""
    pyr = []
    cur = imgs
    for _ in range(levels):
        blur = _gaussian_blur(cur)
        pyr.append(cur - blur)
        cur = blur[:, ::2, ::2]
    pyr.append(cur)
    return pyr


def _extract_patches(level, patch=7, n_per_img=64):
    N, H, W = level.shape
    if H < patch or W < patch:
        return np.zeros((0, patch * patch), np.float32)
    out = []
    for i in range(N):
        ys = np.random.randint(0, H - patch + 1, n_per_img)
        xs = np.random.randint(0, W - patch + 1, n_per_img)
        for y, x in zip(ys, xs):
            out.append(level[i, y:y + patch, x:x + patch].ravel())
    p = np.asarray(out, np.float32)
    # normalize each patch (PGGAN style)
    p = p - p.mean(1, keepdims=True)
    s = p.std(1, keepdims=True)
    return p / np.maximum(s, 1e-5)


def _sliced_wasserstein(A, B, n_proj=256, seed=0):
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    rng = np.random.RandomState(seed)
    dirs = rng.randn(A.shape[1], n_proj).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    pa = np.sort(A @ dirs, axis=0)
    pb = np.sort(B @ dirs, axis=0)
    n = min(pa.shape[0], pb.shape[0])
    # resample to common length
    idx = np.linspace(0, pa.shape[0] - 1, n).astype(int)
    idy = np.linspace(0, pb.shape[0] - 1, n).astype(int)
    return float(np.abs(pa[idx] - pb[idy]).mean())


def swd(real_dems, fake_dems, vmax=3776.0, levels=4):
    """Sliced-Wasserstein over a Laplacian pyramid. Lower = closer distributions.
    DEMs normalized to ~[-1,1] by /vmax for scale invariance."""
    R = np.stack([np.clip(d / vmax, -0.1, 1.5) for d in real_dems]).astype(np.float32)
    F = np.stack([np.clip(d / vmax, -0.1, 1.5) for d in fake_dems]).astype(np.float32)
    pr = _laplacian_pyramid(R, levels)
    pf = _laplacian_pyramid(F, levels)
    per_level = []
    for lr, lf in zip(pr, pf):
        a = _extract_patches(lr)
        b = _extract_patches(lf)
        per_level.append(_sliced_wasserstein(a, b))
    finite = [v for v in per_level if np.isfinite(v)]
    return {"swd_per_level": per_level,
            "swd_mean": float(np.mean(finite)) if finite else float("nan")}


# ----------------------------- top-level ----------------------------- #
def compare_all(real_dems, fake_dems, res=60.0, vmax=3776.0, sea_thresh=0.5):
    real_dems = [np.asarray(d, np.float32) for d in real_dems]
    fake_dems = [np.asarray(d, np.float32) for d in fake_dems]
    out = {}
    out["elevation"] = dist_compare(
        _pool_values(real_dems, lambda d: d, land_only=True, sea_thresh=sea_thresh),
        _pool_values(fake_dems, lambda d: d, land_only=True, sea_thresh=sea_thresh))
    out["slope"] = dist_compare(
        _pool_values(real_dems, lambda d: slope_deg(d, res), sea_thresh=sea_thresh),
        _pool_values(fake_dems, lambda d: slope_deg(d, res), sea_thresh=sea_thresh))
    out["roughness"] = dist_compare(
        _pool_values(real_dems, lambda d: roughness(d), sea_thresh=sea_thresh),
        _pool_values(fake_dems, lambda d: roughness(d), sea_thresh=sea_thresh))
    out["hypsometric"] = hypsometric_compare(real_dems, fake_dems)
    out["psd"] = psd_compare(real_dems, fake_dems, res=res)
    out["swd"] = swd(real_dems, fake_dems, vmax=vmax)
    # land area
    out["land_km2_real"] = float(np.mean([(d > sea_thresh).sum() for d in real_dems]) * res * res / 1e6)
    out["land_km2_fake"] = float(np.mean([(d > sea_thresh).sum() for d in fake_dems]) * res * res / 1e6)
    return out


def summarize(out):
    """One-line-per-metric human summary (drops big curve arrays)."""
    s = []
    s.append(f"elevation: KS={out['elevation']['ks']:.3f} W={out['elevation']['wasserstein']:.1f}m")
    s.append(f"slope:     KS={out['slope']['ks']:.3f} W={out['slope']['wasserstein']:.2f}deg")
    s.append(f"roughness: KS={out['roughness']['ks']:.3f} W={out['roughness']['wasserstein']:.2f}")
    s.append(f"hypso L1={out['hypsometric']['hypso_l1']:.3f} HI real/fake={out['hypsometric']['HI_real']:.2f}/{out['hypsometric']['HI_fake']:.2f}")
    s.append(f"PSD beta real/fake={out['psd']['beta_real']:.2f}/{out['psd']['beta_fake']:.2f} (D {out['psd']['D_real']:.2f}/{out['psd']['D_fake']:.2f}) curveL2={out['psd']['psd_logcurve_l2']:.2f}")
    s.append(f"SWD mean={out['swd']['swd_mean']:.4f} per-level={[round(x,4) for x in out['swd']['swd_per_level']]}")
    s.append(f"land km2 real/fake={out['land_km2_real']:.0f}/{out['land_km2_fake']:.0f}")
    return "\n".join(s)
