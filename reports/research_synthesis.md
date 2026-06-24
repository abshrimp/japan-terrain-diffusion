# DECISIVE ARCHITECTURE RECOMMENDATION â Fictional Japanese Island DEM Generator

Bottom line up front: **Two-stage pixel-space cascade â Stage 1 = small full-canvas EDM2 diffusion (global structure), Stage 2 = fully-convolutional RRDBNet super-res refiner run over the entire canvas in one conv pass (local detail).** This is the only design that simultaneously (a) honors the no-tiling/no-MultiDiffusion constraint at the resolution-determining stage, (b) fits BOTH training and inference in 22.5 GB, and (c) sidesteps the VAE-smearing risk that hurts ridge/drainage realism. Do NOT use latent diffusion as primary (it adds a VAE that biases the high-frequency spectrum â the exact thing you must validate), and do NOT use a single full-res diffusion UNet (training at 1536 px OOMs and patch-training destroys global coherence).

---

## 1. DATA SOURCE

**DEM: FABDEM v1.2** (bare-earth, 30 m, trees+buildings removed). Rationale: Copernicus GLO-30 and AW3D30 are DSMs â they bake canopy/building height into "terrain," so the generator learns vegetation bumps as landforms. FABDEM is derived from GLO-30 with ML forest/building removal â clean bare-earth, ideal for learning geomorphology. Access via HuggingFace `links-ads/fabdem-v12` (STAC catalog, 1Â°Ã1Â° GeoTIFF tiles, ~26 MB each int16). Fallback: AW3D30 (JAXA, excellent over Japan) if you later want a DSM variant.

**Bounding boxes â 4 main islands, Nansei excluded** (use these to select 1Â° FABDEM tiles; tile `NxxEyyy` covers lat[xx,xx+1], lon[yyy,yyy+1]):

| Island | Lat (Â°N) | Lon (Â°E) | ~Area | ~1Â° tiles |
|---|---|---|---|---|
| Hokkaido | 41.3 â 45.6 | 139.3 â 146.0 | 78,000 kmÂ² | ~22 |
| Honshu | 33.4 â 41.6 | 130.8 â 142.1 | 228,000 kmÂ² | ~50 |
| Shikoku | 32.7 â 34.6 | 132.0 â 134.8 | 19,000 kmÂ² | ~9 |
| Kyushu | 30.95 â 34.0 | 129.3 â 132.1 | 37,000 kmÂ² | ~12 |

**Explicit exclusions (Nansei + oceanic outliers):** everything south of lat 30.9Â°N (Åsumi/Tanegashima/Yakushima, Amami, Okinawa, Sakishima); the Izu/Ogasawara chain (lon > 139.0, lat < 34.0 in the open Pacific); DaitÅ. Curate these out at the tile level. The Honshu box overlaps Shikoku/Kyushu around the Seto Inland Sea â harmless, since boxes only drive tile selection and you mask by land afterward.

**Ocean/water handling:** FABDEM sets sea as NoData. Set NoData â 0 (sea level). Build a **land mask** from `elevation > 0` (or the original NoData mask), persisted alongside each mosaic. Inland water (lakes) stays at its filled elevation. Keep ocean a flat 0 plateau â the model learns coastline morphology (cliffs vs. beaches) directly from Japan's varied coasts. Compute ALL metrics on land pixels only.

**Geoid/datum:** FABDEM v1.2 = horizontal WGS84/EPSG:4326, vertical **EGM2008 orthometric** (inherited from GLO-30). Heights are above-MSL, so coastal land â 0 naturally â **no vertical datum conversion needed**. Just verify coastal pixels sit near 0 and clamp ocean/NoData to 0.

**Projection / downsampling:** geographic pixels are non-square at Japan's latitudes (1â³ EâW â 25 m at 35Â°N, 21 m at 45Â°N vs. 30 m NâS). Reproject to a **metric grid per island UTM zone** (Kyushu/Shikoku/W-Honshu â EPSG:32652/53N; central/E-Honshu â 32654N; Hokkaido â 32654/55N) so pixels are square and anisotropy is correct. For reprojection use `gdalwarp -r cubic`; for any decimation use **`-r average`** (anti-aliasing is critical for honest PSD). Store the HR mosaic at the **final GSD (~55â58 m/px), int16 meters** (range 0â3776 fits int16). You lose nothing below the target GSD.

**Disk-disciplined plan (24 GB):**
1. Per island: stream-download its 1Â° tiles, build a GDAL **VRT** (zero extra disk).
2. `gdalwarp` the VRT â one metric int16 GeoTIFF at ~57 m/px â write land mask â **delete raw tiles** before moving to the next island. Peak raw footprint â 1.3 GB (Honshu), never all at once.
3. Final per-island mosaics: Honshu ~70 MB, Hokkaido ~25 MB, Kyushu ~12 MB, Shikoku ~6 MB â **~115 MB total at 57 m.** Leaves >23 GB for checkpoints/outputs.
4. Crops are generated **on-the-fly** during training from the mosaics (no pre-cut patch dataset on disk). Set `HF_HOME` to scratch and purge the download cache after each island.

---

## 2. ARCHITECTURE

**Cascaded pixel-space, single-channel. Final canvas = 1536Ã1536 px, coarse = 384Ã384 px, SR factor = 4Ã.** Pick GSD â 55â58 m/px so the physical canvas is ~85â89 km and ~5000 kmÂ² **land** occupies ~65 % of it (realistic ocean margin around the island). Both dims divisible by 32 (UNet/divisibility-safe).

| Stage | Model | Resolution | GSD | Params | Train batch | Train VRAM | Infer VRAM |
|---|---|---|---|---|---|---|---|
| 1 Coarse | EDM2 mag-preserving UNet, 1-ch | 384Â² full canvas | ~228 m/px | ~50 M | 8 (+accumâ256) | ~10â14 GB | <2 GB |
| 2 Refine | RRDBNet (Real-ESRGAN gen), 1-ch | train 128â512 patch; **infer 384â1536 full** | 228â57 m/px | ~12 M | 16 | ~6 GB | ~3â4 GB |

**Stage 1 â coarse, global structure (the part that must be globally coherent).** A small EDM2 magnitude-preserving UNet in **pixel space**, 1-channel, generating the **entire 384Ã384 canvas in one spatial pass**. Downsampling 384â192â96â48â24â12 with self-attention at 48/24/12 px so the 12-px bottleneck sees the whole island â coastline, mountain ranges, drainage basins are decided coherently. **Trained at the actual 384-px canvas size, NOT on small patches** (per the pitfall: conv/latent denoisers run above their training resolution self-tile and lose global structure). Training samples = random 88 km Ã 88 km windows from the Japanese mosaics downsampled (average) to 384 px, **filtered to land fraction 40â95 %** so the model learns island/coastal morphology rather than open ocean or pure interior. EDM2 recipe: Ï-preconditioning, log-normal noise (P_mean=â1.2, P_std=1.2), Ï_data calibrated to your normalized stats, post-hoc EMA, Heun sampler 24â32 NFE. Optionally condition on a coarse coastline/land-mask for shape control. Reference: `FutureXiang/edm2` (minimal) or `NVlabs/edm2`.

**Stage 2 â refiner (the part that must be true single-pass, no tiling).** **RRDBNet** (ESRGAN/Real-ESRGAN generator): no BatchNorm, no attention, no positional encoding â translation-equivariant, runs at any H/W. Every conv `padding_mode='reflect'`; pixel-shuffle Ã4 upsample (no transposed-conv checkerboard). **Predict the high-frequency residual over the bicubic-upsampled coarse field** (residual-over-trend stabilizes large-factor DEM SR). Trained on 128 px LR â 512 px HR patch pairs, where LR = HR averaged Ã4 â cheap, fits trivially. **At inference, feed the full 384âbicubic-1536 canvas through one `torch.no_grad()` conv pass â 1536Â² in one shot, no tiles, ~3â4 GB.** Harness: `XPixelGroup/BasicSR` (patch-train + full-image-infer with the same checkpoint, zero code change).

**Why this satisfies "ONE forward pass / no tiling / no MultiDiffusion":** The resolution-determining detail synthesis over the full 1536Â² canvas (Stage 2) is **literally one convolutional forward pass with no spatial tiling or seam blending**. Stage 1 processes the **complete spatial canvas at every step** â its iteration is over diffusion *time*, not *space*; there is zero MultiDiffusion, zero spatial windowing, zero overlap-blend anywhere in the pipeline. (If you want true near-single-eval, distill Stage 1 to a 2-step sCM consistency model as in xandergos â optional.)

**Why it fits 22.5 GB for BOTH training and generation â the key trick:** the memory bottleneck is *training a single-pass model at full res* (autograd stores activations for all blocks at 1536Â²). The cascade splits this so neither stage ever back-props at full canvas: **Stage 1 trains at small full-canvas (384Â²); Stage 2 trains on small patches (128Â²) but its purely-local architecture runs identically at 1536Â² under no_grad.** Global coherence comes from Stage 1; local detail from Stage 2; full-canvas memory is only ever paid at inference, where it's ~3â4 GB. Use gradient checkpointing + gradient accumulation to reach EDM2's needed effective batch (â256) on one GPU.

**Stage 2 statistical fidelity:** RRDBNet GANs under-produce HF power. Close the gap with a **radially-averaged PSD (FFT-magnitude) loss + slope/gradient + curvature(Laplacian) losses** (TfaSR) on top of L1, plus relativistic-GAN with minibatch-std + R1 + ADA augmentation. Apply **conditioning augmentation** (blur+noise on the LR input during SR training) so the refiner is robust to the gap between real-downsampled LR and Stage-1-generated LR (prevents compounding error).

**Latent-diffusion alternative (documented, not chosen):** train a dedicated 1-channel f=8 VAE, denoise a ~192Â² latent, single full-frame decode. Rejected as primary because the VAE measurably biases the HF spectrum and smears drainage networks (Terrain Diffusion needed a Laplacian split to fix this) â directly at odds with the "match statistics" requirement. Keep as a fallback only if RRDBNet HF detail proves insufficient.

---

## 3. NORMALIZATION

- **Sea = 0:** ocean/NoData â 0. Land mask = (h > 0).
- **Clip:** `h = clip(h, 0, 3776)` (Mt. Fuji, Japan's max; use 3800 for margin). Prevents rare peaks from dominating dynamic range / causing banding.
- **Compressive transform (sqrt):** Japan's hypsometry is heavily skewed to low elevation, so allocate capacity there:

  `x = 2*sqrt(h_clip/3776) - 1`  â sea = â1, Fuji = +1, monotonic, invertible
  inverse: `h = 3776 * ((x+1)/2)**2`

  Sqrt expands the common low-elevation/coastal band; the flat â1 ocean plateau lets the model learn coastline as the plateau boundary. (Alternatives to A/B-test via hypsometry+PSD: linear `(h/3776)*2â1`, or `log1p` if sqrt over-amplifies coastal HF. Validate, don't assume.)

- Use this on both stages; Stage 2 operates on the same normalized space, residual added in normalized units, denormalize once at the end before writing the GeoTIFF.

---

## 4. VALIDATION (metric suite â land pixels only; mask first, then detrend/window)

1. **Radial PSD (primary "is-it-real-terrain"):** subtract least-squares plane â 2D Tukey(0.25)/Hann window â `pysteps.utils.spectral.rapsd(field, return_freq=True, d=gsd_m, normalize=True)` â fit log10(P) vs log10(k) over the **intermediate band only** (drop lowest 3â5 bins and near-Nyquist). Target **Î²2D â 2.5â3.5**, fractal **D=(8âÎ²2D)/2 â 2.2â2.7**, Hurst â 0.5. Report `|Î²_gen â Î²_real|` and overlay the two RAPSD curves. **Pin the convention:** rapsd is ring-mean â use D=(8âÎ²2D)/2 (never mix with the (7âÎ²1D)/2 transect form). Cross-check with a **variogram** (`skgstat`, Î³(h)~h^(2H)).
2. **Elevation histogram:** `scipy.stats.wasserstein_distance` (meters, primary) + `ks_2samp` D-statistic (secondary). KS p-values are meaningless at ~10â¶ px â report D effect size or subsample to ~5â10k.
3. **Slope:** `np.gradient(z, dy, dx)` with real spacing â slope = arctan(â(gxÂ²+gyÂ²)); compare slope-angle distributions by Wasserstein/KS.
4. **Roughness:** std of locally-detrended residual elevation (RMSH) + std curvature (avoid TRI/std-elev â slope proxies).
5. **Hypsometry:** cumulative-area-vs-normalized-elevation curve + HI=(meanâmin)/(maxâmin); compare HI and max curve separation.
6. **Distribution distance at Nâ1:** tile both real-Japan and the generated island into 128â256 px patches â **hand-feature Frechet/MMD** on [Î², HI, mean/std slope, RMSH, std curvature, elevation quantiles] (most interpretable, robust at small N) AND/OR FD-DINOv2 (3Ã-replicate patch â frozen ViT CLS). Use KID/MMD (low bias), not Inception-FID.
7. **Seam-free proof (must pass):** generate the full canvas, then generate two halves and compare the overlap â with reflect padding and no global ops the interior must match to ~float precision, empirically confirming translation-equivariant single-pass behavior.

---

## 5. PHASE 0 PoC (fast end-to-end validation, <Â½ day)

Goal: exercise the **entire dataâtrainâgenerateâvalidate loop** and prove the no-tiling claim before scaling.

- **Data:** Shikoku only (~9 tiles, smallest). Mosaic at 57 m, land mask.
- **Stage 1:** tiny EDM UNet (~5â8 M params, base 64ch) at **96Ã96** coarse canvas, fp16/fp32, batch 16, ~1â2 h on the 2080 Ti.
- **Stage 2:** small RRDBNet (6 RRDB) Ã4, train 64â256 patches, L1+PSD only (skip GAN for PoC), ~1â2 h.
- **Generate:** 96 coarse â bicubic 384 â RRDBNet one pass â 384Â² @ ~57 m â denormalize â write GeoTIFF.
- **Run full Â§4 suite** including the seam test.
- **Acceptance gates:** (a) end-to-end runs, no OOM, big headroom; (b) seam-test interior matches < 1e-3; (c) RAPSD Î² within Â±0.5 of real Shikoku; (d) elevation/slope Wasserstein finite and sane; (e) normalization round-trips (sqrt inverse) exactly; (f) ocean mask + coastline behave. Catch normalization/geoid/divisibility/ocean bugs here, cheaply.

---

## 6. RISKS & MITIGATIONS (hardware-specific)

- **Turing has NO bf16/TF32.** The 22.5 GB card is a VRAM-modded RTX 2080 Ti (Turing). Findings casually suggest bf16 â **do not use it.** Use **fp16 AMP with loss scaling + fp32 master weights**, or train Stage 1 in fp32 (it's small at 384Â²). EDM2 magnitude-preserving layers + preconditioning give fp16 stability; RRDBNet trains fine in fp16/fp32.
- **"Code may assume 48 GB, physical 22.5 GB":** force fp16 inference, `torch.no_grad()`, never call `enable_tiling()`/VAE tiled decode (the documented seam source). Our full-canvas 1536Â² conv pass is ~3â4 GB â ~6Ã headroom; no assumptions about 48 GB anywhere.
- **24 GB disk:** stream+delete tiles, VRT instead of duplicated mosaics, store only 57 m int16 (~115 MB), on-the-fly crops, prune checkpoints to top-k + EMA, drop optimizer state for shipped models, purge HF cache.
- **Training OOM (the classic single-pass trap):** avoided structurally by the cascade (Stage 1 at 384Â², Stage 2 patch-trained). Never attempt full-res diffusion training. Grad checkpointing + accumulation for effective batch ~256.
- **Conv-model incoherence at scale:** Stage 1 trained at the true coarse canvas (global structure learned, not self-tiled); Stage 2 is intentionally local-only (global trend supplied by Stage 1). This split is what makes single-pass both coherent and memory-feasible.
- **GAN mode collapse / HF deficit (Stage 2):** PSD+slope+curvature losses, minibatch-std, R1, spectral norm, ADA augmentation. Fallback: non-adversarial diffusion-SR refiner (accepts iterative, but keep RRDBNet primary to avoid UNet-above-training-res incoherence).
- **Border ring degradation:** reflect padding + generate a small margin and crop it â never deliver the literal network edge as the island coastline.
- **Overfitting on limited effective diversity:** 8Ã dihedral augmentation, random crops across 365,000 kmÂ², elevation-histogram-balanced window sampling, modest model sizes (EDM2-XS/S scale).
- **Aliasing corrupting PSD:** always `-r average` for decimation; window before FFT; this is also why HR is stored at target GSD via averaging, not nearest.

---

### Concrete starting stack
`FutureXiang/edm2` (Stage 1) + `XPixelGroup/BasicSR`/`xinntao/Real-ESRGAN` RRDBNet adapted to 1-channel reflect-padding residual mode (Stage 2); GDAL for the FABDEM `links-ads/fabdem-v12` ingest; `pysteps`/`scipy`/`skgstat`/`POT` for Â§4. Mine `xandergos/terrain-diffusion` for the signed/sqrt elevation-encoding and EDM2 data-prep details (ignore its InfiniteDiffusion tiled sampler).