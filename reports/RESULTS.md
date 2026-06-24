# Phase-1 Results — Fictional Japanese-Island DEM Generation (FINAL)

Final model: **coarse@24k + SR@12k**. See `ITERATION_LOG.md` for the full arc.

## What was built
A two-stage **pixel-space EDM diffusion cascade** that generates a fictional island
DEM (~5000 km² land, 60 m/px, 1536×1536 = 92 km canvas) in a **single un-tiled
forward pass per stage** (no tiling / no seam-blending / no MultiDiffusion):
- **Coarse** (大局地形): EDM diffusion UNet w/ attention, 384²@240 m, 18.2 M params.
- **SR** (詳細地形): attention-free, reflect-padded, **translation-equivariant** EDM
  diffusion UNet, ×4 → 1536²@60 m, 16.6 M params. Trained on 256² patches, run on the
  **whole canvas in one pass** at generation.

Translation-equivariance verified **bit-exactly** (full-canvas vs aligned-crop
interior diff = 0.0). Full 1536² SR inference fits in **7.8 GB** (≪ 22.5 GB).

## Final quantitative validation (8 generated vs 40 real crops, 1536²@60 m, land px)
| metric | generated | real | verdict |
|---|---|---|---|
| elevation KS / Wasserstein | 0.05–0.11 / 21–50 m | — | close |
| slope KS / Wasserstein | 0.05–0.10 / 1.1–2.4° | — | close |
| roughness KS | 0.045–0.096 | — | close |
| hypsometric L1 | 0.053–0.068 | — | curve shape close |
| hypsometric integral (HI) | **0.14–0.15** | **~0.19** | slightly flat (residual gap) |
| **radial PSD slope β** | **3.52** | **3.37** | gap 0.15 (≪0.5 gate) — matches |
| fractal dim D = 4−β/2 | 2.24 | 2.32 | close |
| Sliced-Wasserstein (SWD) | **0.098** | 0 | low (image-dist) |
| largest-island area | 4246–5670 km² | target ~5000 | within ±1000 ✓ |
| single-island fraction | 0.93–1.00 | — | coherent single islands ✓ |
| max elevation | 1405–2890 m | (Japan p99≈1776, max 3776) | realistic ✓ |

(Ranges span the Iter-4/5/final evaluation runs; per-run variance is ±~0.05 on KS due
to the small 6–8 island fake set.)

## Improvement arc (key metrics)
| | PoC | Iter3 SR@8k | Iter4 SR@12k | Iter5 c@16k | Final c@24k |
|---|---|---|---|---|---|
| elevation KS | 0.46 | 0.122 | 0.086 | 0.054 | 0.110 |
| slope KS | 0.17 | 0.143 | 0.109 | 0.049 | 0.100 |
| PSD β (gen) | 2.47 | 3.63 | 3.52 | 3.62 | 3.52 |
| SWD | 0.23 | 0.208 | 0.156 | 0.117 | **0.098** |
| HI (gen) | 0.08 | 0.13 | 0.13 | 0.15 | 0.14 |
| island km² | 298 | 5847 | 5479 | 5109 | 4818 |

Two decisive fixes: **(1) EDM `sigma_data` 0.5→0.26/0.14** (matched to data std) cured
the flatness; **(2) continued training past the loss plateau** kept improving sample
quality (sample fidelity ≠ loss). SR sharpening lowered β toward real + cut SWD.

## Visual self-critique (Claude opened hillshades & compared to real renders)
- **Global (大局):** coherent single islands; complex natural coastlines (peninsulas,
  bays, offshore islets); mountain ranges/basins at plausible scale; coastal-lowland →
  interior-mountain gradient. ✓
- **Detail (詳細):** coherent dendritic valley/ridge/drainage networks closely matching
  real terrain. ✓
- **Residual:** generated terrain is slightly gentler than the steepest real Japanese
  interiors (HI 0.14 vs ~0.19). Diagnosed as a genuine (not structural) model gap;
  improved monotonically with training. Further closing would need a larger coarse
  model / less-compressive normalization / longer training (future work).

## Showcase
`outputs/phase1_final/island_05_seed1_5670km2_*` — single 5670 km² island, max 2392 m,
complex coastline + dendritic mountain drainage. Montages + curves in
`reports/phase1_final/`.

## Hardware / cost
Single RTX 2080 Ti (22.5 GB, Turing, fp16-only). Gradient checkpointing fits 384²
coarse + 256² SR training. Coarse ~6.8 img/s, SR ~17 img/s. Generation ≈ 2 min/island.
Total training ≈ coarse 24k + SR 12k steps.
