# Fictional Japanese-Island DEM Generator

Learn the terrain of Japan's four main islands (Honshu, Hokkaido, Shikoku, Kyushu;
**Nansei/Ryukyu excluded**) from real DEM data and **generate realistic fictional
island elevation models** (~5000 km² land, ~60 m/px) that match real terrain both
**statistically** and **visually**.

Generation is done as a **two-stage pixel-space diffusion cascade** that outputs the
**entire canvas in a single un-tiled forward pass** (no overlap-averaging / feathering /
MultiDiffusion). See `reports/DECISIONS.md` for the full data-source & architecture
rationale, `reports/research_synthesis.md` for the literature survey, `reports/RESULTS.md`
for final metrics, and `reports/ITERATION_LOG.md` for the change-by-change history.

**Status:** Phase-0 PoC and Phase-1 complete, plus an improvement pass (Iter 7–8). The
final model (**coarse@29k with relief/land-fraction-matched sampling + SR@12k**)
generates coherent single islands centered on ~5000 km² (max elev to ~2860 m) that match
real Japanese terrain statistically — **elevation KS 0.026, slope KS 0.039, roughness KS
0.029** (near-perfect), HI 0.17 vs 0.20, radial PSD slope β 3.58 vs real 3.36, SWD 0.095
— and visually (natural coastlines + dendritic mountain drainage; volcano-like cones).
Showcase: `outputs/iter8/island_00_seed8_4991km2_shaded.png`. Metrics: `reports/RESULTS.md`.

## Pipeline

```
download_dem.py → preprocess.py → train.py (coarse) → train.py (sr) → generate.py → validate.py
   (GLO-30)        (LCC 60 m       (EDM diffusion)    (EDM cond. SR)   (cascade,     (stats + visual)
                    mosaic+mask)                                        post-select)
```

## Project layout
```
data/raw/             Copernicus GLO-30 tiles (1° GeoTIFF)         [transient]
data/processed/       japan_dem_60m.tif, japan_landmask_60m.tif, stats.json
src/                  download_dem, preprocess, dataset, networks, edm, train, generate, validate, metrics, render
configs/              poc.yaml (Phase-0), phase1.yaml (Phase-1)
checkpoints/<name>/<stage>/   latest.pt + ckpt_stepN.pt (EMA weights inside)
outputs/<name>/<stage>/       periodic training-sample montages
outputs/generated/    final generated islands (.tif + hillshade/color/shaded PNG)
reports/              DECISIONS.md, research_synthesis.md, eval reports, montages
logs/                 setup/download/preprocess logs + TensorBoard event files
scripts/              setup_env.sh, smoke_test.py
```

## Setup
```bash
bash scripts/setup_env.sh        # uv venv (py3.11) + torch cu121 + geo/ml libs
source .venv/bin/activate
```
Key versions: Python 3.11, torch 2.4.1+cu121, rasterio 1.3.10 (GDAL 3.8.4),
numpy 1.26, scipy 1.13. CUDA driver 580 / RTX 2080 Ti (22.5 GB). Pins in `requirements.txt`.

## Data acquisition
```bash
python src/download_dem.py --out data/raw --workers 16     # 90 tiles (~1.8 GB)
python src/preprocess.py   --raw data/raw --out data/processed
```
Builds a seamless Japan mosaic in Lambert Conformal Conic @60 m/px (int16 m, ocean=0),
a land mask, and `stats.json`. On-the-fly crops are sampled from this mosaic during
training (no per-crop dataset on disk).

## Train (cascade)
```bash
# Phase-0 PoC (fast end-to-end check)
python src/train.py --config configs/poc.yaml --stage coarse
python src/train.py --config configs/poc.yaml --stage sr

# Phase-1 (full; run in background, logs + periodic sample montages)
python src/train.py --config configs/phase1.yaml --stage coarse   # resume with --resume
python src/train.py --config configs/phase1.yaml --stage sr
```
fp16 AMP + EMA + gradient clipping; TensorBoard logs under `logs/<name>/<stage>`.
Checkpoints every N steps (resume with `--resume`).

## Generate
```bash
python src/generate.py --config configs/phase1.yaml \
  --coarse-ckpt checkpoints/phase1/coarse/latest.pt \
  --sr-ckpt     checkpoints/phase1/sr/latest.pt \
  --n 24 --keep 6 --target-km2 5000 --tol-km2 1000 --out outputs/generated
```
Generates N coarse candidates, post-selects those near the target land area, then runs
the SR refiner over the **full canvas in one pass**. Writes georeferenced `.tif` plus
hillshade / color-relief / shaded-relief PNGs.

## Validate (statistical + visual)
```bash
python src/validate.py --config configs/phase1.yaml \
  --gen-dir outputs/generated --n-real 32 --out reports/phase1
```
Computes elevation KS/Wasserstein, slope & roughness distributions, hypsometric curve,
radial power-spectrum slope β & fractal D, Sliced-Wasserstein distance, and land area;
renders real-vs-fake montages and PSD/hypsometric curve plots. Visual self-critique is
performed by opening the hillshade PNGs and comparing to real renders.

## License / attribution
- **Copernicus DEM GLO-30** — free to use with attribution:
  *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and
  Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all
  rights reserved."* Source: AWS Open Data bucket `copernicus-dem-30m`.
- Generated DEMs are synthetic (not real locations).
```
