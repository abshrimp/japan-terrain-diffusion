# Data-source & Architecture Decisions (with rationale)

Project: learn Japanese DEM → generate realistic fictional-island DEMs (~5000 km²
land), final ~60 m/px, single forward pass (no tiling), statistically & visually
realistic. Decisions below are grounded in a multi-agent literature survey
(see `reports/research_synthesis.md`) plus measured environment constraints.

## 0. Environment (measured)
| Resource | Value | Consequence |
|---|---|---|
| GPU | 1× RTX 2080 Ti, **22.5 GB**, Turing | fp16 AMP only (no bf16/TF32); design must fit 22.5 GB for train AND generate |
| CPU / RAM | 72 cores / 125 GB | ample for preprocessing + in-RAM mosaic + on-the-fly crops |
| Disk | **24 GB** (`/home`) | strict discipline: compact int16 mosaic, on-the-fly crops, pruned checkpoints |
| Net | AWS Copernicus bucket reachable (no auth) | data acquisition fully automatable |

## 1. DATA SOURCE — **Copernicus DEM GLO-30** (AWS Open Data)

**Decision:** Copernicus DEM GLO-30, public bucket `copernicus-dem-30m` (no auth),
90 land-touching 1° tiles over Honshu/Hokkaido/Shikoku/Kyushu, **Nansei excluded**.

**Why GLO-30 (and not the research's FABDEM suggestion):**
- **License safety.** GLO-30 is free for any use with attribution. FABDEM v1.x is
  **CC-BY-NC-SA (non-commercial + share-alike)** — its derivatives (our generated
  DEMs) would inherit NC-SA. With use unspecified, the permissive source is safer.
- **Self-consistency of validation.** We compare *generated* vs *real* where **both**
  come from GLO-30. Any product-specific texture (incl. forest canopy in this DSM)
  is part of the learned *and* the reference distribution, so it is not a confound —
  "realism" is realism relative to how Japan looks in GLO-30, which is the stated target.
- **Automation + already acquired.** One `tileList.txt`, deterministic tile names,
  parallel HTTPS pull (90 tiles, 1.84 GB, ~5 min). User listed it as primary candidate.
- **Datum is convenient.** Heights are EGM2008 orthometric → coastal land ≈ 0 naturally;
  ocean encoded as 0. No vertical-datum conversion needed.

**Documented caveat / alternative.** GLO-30 is a DSM (includes canopy/buildings),
adding some high-frequency roughness vs true bare-earth. If a bare-earth, license-
compatible source is desired later, **FABDEM** (HuggingFace `links-ads/fabdem-v12`,
if NC-SA acceptable) or **AW3D30** (JAXA) are drop-in swaps for `src/download_dem.py`.

**Tile selection (per-island integer-degree boxes; cleanly excludes Nansei < 31 N):**
Hokkaido N41–45/E139–145 · Honshu N33–41/E130–142 · Shikoku N32–34/E132–134 ·
Kyushu N31–33/E129–131. → 90 existing tiles. Measured total land = **346,979 km²**
(matches the real 4-island area), p50=284 m, p99=1776 m, max=3760 m (Fuji).

**Projection / resolution.** Reproject the whole mosaic **once** to a Japan-centered
**Lambert Conformal Conic** (lat₁=33, lat₂=45, lat₀=38, lon₀=137) at **60 m/px** with
`Resampling.average`. Conformal → square metric pixels, correct local shape/scale →
honest slope/roughness/power-spectrum statistics. This is a *deliberate, documented*
exception to "prefer downsampling over reprojection": EPSG:4326 pixels are 14–31 %
anisotropic across 31–46 N, which would corrupt the very statistics we must match.
60 m = native 30 m ÷2 (exactly the resolution the brief endorses).
**Seam-free:** merge raw tiles in their shared 4326 grid first, then one continuous
reproject → no per-tile reprojection edges (an earlier per-tile version produced
thin boundary seams at the 1° spacing; fixed).

## 2. ARCHITECTURE — **Two-stage pixel-space cascade (EDM diffusion)**

**Decision.**
- **Stage 1 — Coarse (大局地形):** EDM (Karras 2022) diffusion UNet **with attention**,
  trained at the **native full coarse canvas** (384², 240 m/px over a 92 km canvas).
  Generates island outline, mountain ranges, major valleys coherently in one spatial pass.
- **Stage 2 — SR refiner (詳細地形):** EDM **conditional** diffusion UNet, **attention-free
  + reflect-padded → translation-equivariant**, ×4 (384→1536, 60 m/px). Conditioned on
  the bicubic-upsampled coarse field. **Trained on 256² patches** (low VRAM) but **run on
  the entire 1536² canvas in a single fully-convolutional pass** at generation.

**Why this satisfies "one forward pass / no tiling / no MultiDiffusion":**
The resolution-determining detail synthesis (Stage 2) processes the **whole canvas at
every denoising step with zero spatial tiling or seam-blending**. Diffusion iterates over
*time*, not *space*. We **empirically proved** position-independence: running the SR net on
a full canvas vs an aligned crop gives **bit-identical interior output (max diff = 0.0)** —
so there are literally no seams to hide. Stage 1 likewise processes its full canvas each step.

**Why the key VRAM trick works (fits 22.5 GB for BOTH train & generate):**
The only memory bottleneck is back-prop at full resolution. The cascade removes it:
Stage 1 trains small (384²); Stage 2 trains on 256² patches but, being purely local
(no attention, no positional embedding), runs **identically** at 1536² under `no_grad`
(fp16) at generation (~several GB). Full-canvas memory is paid only at inference.

**Rejected alternatives (with reasons):**
- **Latent diffusion (VAE).** The VAE biases the high-frequency spectrum — exactly the
  quantity we must validate (radial PSD slope). Rejected as primary.
- **Single full-res diffusion UNet at 1536².** Training OOMs on 22.5 GB; patch-training a
  single global model destroys large-scale coherence. Rejected.
- **RRDBNet/Real-ESRGAN GAN refiner** (research's Stage-2 pick). Strong, but GAN training
  on a single GPU is less stable and needs a discriminator + a perceptual loss that is
  ill-suited to single-channel terrain. Diffusion SR gives better distribution matching
  with our existing EDM code and no adversarial instability. Kept as a fallback if detail
  is insufficient after iteration.

## 3. NORMALIZATION — sqrt compression
`x = 2·√(clip(h,0,3776)/3776) − 1` (sea→−1, Fuji→+1, invertible). Japan's hypsometry
is heavily right-skewed (p50=284 m); sqrt expands the common low-elevation/coastal band
for better model capacity allocation. `linear` and `log` are A/B-selectable in config.
Metrics are computed in **meters** (after inverse) on **land pixels only**.

## 4. VALIDATION — statistical + visual
Elevation KS + Wasserstein · slope dist · roughness · hypsometric curve + HI ·
radial PSD slope β & fractal D=4−β/2 + log-curve L2 · Sliced-Wasserstein (Laplacian
pyramid, pretrained-free) · land-area km². Plus **visual self-critique**: Claude opens
generated hillshade/color-relief PNGs beside real renders and iterates.

## 5. SIZE DESIGN
Canvas 1536²@60 m = 92.2 km (8493 km²). Target island ~5000 km² (≈59 % land) hit via
**post-selection** on coarse land area (±1000 km² tolerance). Mask-conditioning of the
coarse stage is the planned upgrade if post-selection yield is low.

## 6. PHASES
- **Phase 0 PoC:** coarse 96²→SR×4→384², tiny models, ~4k steps each — validate the full
  data→train→generate→validate loop + no-tiling claim. Acceptance: runs w/o OOM; seam
  test passes; β within ±0.5 of real; finite sane Wasserstein; norm round-trips.
- **Phase 1:** coarse 384² + SR×4→1536², full models, long training, iterate to criteria.
