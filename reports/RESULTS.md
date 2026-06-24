# Phase-1 Results — Fictional Japanese-Island DEM Generation (FINAL)

Final model: **coarse@29k (mild relief sampling) + SR@12k**. Full history in
`ITERATION_LOG.md`.

## What was built
A two-stage **pixel-space EDM diffusion cascade** that generates a fictional island
DEM (~5000 km² land, 60 m/px, 1536×1536 = 92 km canvas) in a **single un-tiled
forward pass per stage** (no tiling / no seam-blending / no MultiDiffusion):
- **Coarse** (大局地形): EDM diffusion UNet w/ attention, 384²@240 m, 18.2 M params.
  Trained with **relief- & land-fraction-matched sampling** (land_lo 0.20, relief_k 1)
  so the training distribution matches the ~0.6-land generation regime.
- **SR** (詳細地形): attention-free, reflect-padded, **translation-equivariant** EDM
  diffusion UNet, ×4 → 1536²@60 m, 16.6 M params. Trained on 256² patches, run on the
  **whole canvas in one pass** at generation.

Translation-equivariance verified **bit-exactly** (full-canvas vs aligned-crop interior
diff = 0.0). Full 1536² SR inference fits in **7.8 GB** (≪ 22.5 GB).

## Final quantitative validation (14 generated vs 40 real crops, 1536²@60 m, land px)
| metric | generated | real | verdict |
|---|---|---|---|
| **elevation KS / Wasserstein** | **0.026 / 21 m** | — | near-perfect |
| **slope KS / Wasserstein** | **0.039 / 0.65°** | — | near-perfect |
| **roughness KS** | **0.029** | — | near-perfect |
| hypsometric L1 | 0.030 | — | curve shape very close |
| hypsometric integral (HI) | 0.17 | ~0.20 | close (gap 0.03) |
| radial PSD slope β | 3.58 | 3.36 | gap 0.22 (the one slightly-off metric) |
| fractal dim D = 4−β/2 | 2.21 | 2.32 | close |
| Sliced-Wasserstein (SWD) | 0.095 | 0 | low |
| largest-island area | 4157–5764 km² (coarse median 5009) | ~5000 | centered ✓ |
| single-island fraction | 0.80–1.00 (mostly 0.97–1.00) | — | coherent single islands ✓ |
| max elevation | to ~2860 m | (Japan p99≈1776, max 3776) | realistic ✓ |

## Improvement arc (key metrics)
| | PoC | Iter4 | Iter6 (prev final) | Iter7 (relief) | **Iter8 (FINAL)** |
|---|---|---|---|---|---|
| elevation KS | 0.46 | 0.086 | 0.110 | 0.162 | **0.026** |
| slope KS | 0.17 | 0.109 | 0.100 | 0.108 | **0.039** |
| roughness KS | 0.22 | 0.110 | 0.096 | 0.140 | **0.029** |
| HI (gen) | 0.08 | 0.13 | 0.14 | 0.19 | **0.17** |
| PSD β (gen) | 2.47 | 3.52 | 3.52 | 3.65 | **3.58** |
| SWD | 0.23 | 0.156 | 0.098 | 0.073 | **0.095** |
| island km² | 298 | 5479 | 4818 | 5261 | **5041** |

Decisive fixes, in order: **(1)** EDM `sigma_data` matched to data std (0.5→0.26/0.14)
cured gross flatness; **(2)** training past the loss plateau (sample quality ≠ loss);
**(3)** SR sharpening; **(4)** **relief- & land-fraction-matched coarse sampling**
(`land_lo` 0.10→0.20) — aligning the training land-fraction with the generation regime
collapsed elevation/slope/roughness KS to 0.026–0.039 and raised HI to 0.17.
(Iter-7's aggressive relief `land_lo` 0.35 + `relief_k` 2 overshot — raised β and KS —
so the milder Iter-8 was adopted.)

## Visual self-critique (Claude opened hillshades & compared to real renders)
- **Global (大局):** coherent single islands; complex natural coastlines (peninsulas,
  bays, isthmuses, offshore islets); mountain ranges/basins at plausible scale;
  coastal-lowland → interior-mountain gradient; good ocean margin. ✓
- **Detail (詳細):** crisp dendritic valley/ridge/drainage networks matching real
  terrain; some volcano-like cones with radial drainage. ✓
- **Residual:** β slightly high (3.58 vs 3.36) — a touch more large-scale relief power
  than real; not visible as blur (fine detail is sharp). Further closing would need a
  larger coarse model / spectral loss term (future work).

## Showcase
`outputs/iter8/island_00_seed8_4991km2_*` — single ~5000 km² isthmus island, two
mountainous lobes, crisp drainage. 14 islands in `outputs/iter8/`; montages + curves in
`reports/iter8/`.

## Hardware / cost
Single RTX 2080 Ti (22.5 GB, Turing, fp16-only). Gradient checkpointing fits 384²
coarse + 256² SR training. Coarse ~6 img/s, SR ~17 img/s. Generation ≈ 2 min/island.
Final training: coarse 29k + SR 12k steps (incl. fine-tuning iterations).
