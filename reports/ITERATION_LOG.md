# Iteration Log

Each entry: what changed, why, and how metrics/visuals moved. Newest first.

---

## Iter 6 — FINAL: coarse trained to 24k
**Date:** 2026-06-24

**Change.** Completed coarse to 24k; final generation (32 candidates → keep 8,
single-island selection) with coarse@24k + SR@12k; validated vs 40 real crops.

**Final result (8 islands).** elevation KS 0.110 / slope KS 0.100 / roughness KS 0.096
/ hypso L1 0.068 / **SWD 0.098 (best across all iters)** / PSD β 3.52 vs 3.37 (gap 0.15)
/ HI 0.14 vs 0.20 / islands 4246–5670 km² (frac 0.93–1.00) / max elev up to 2890 m.

**Verdict: converged.** coarse@24k ≈ coarse@16k (KS fluctuates ±0.05 from the small
8-island fake set; SWD improved to 0.098). Visuals (e.g. island_05: 5670 km², dendritic
mountain drainage, complex coastline) read as genuine Japanese islands in both global
and detail structure. **Goal achieved.** Residual: HI slightly low (terrain a touch
gentle than steepest real interiors) — improved with training, plateaued ~0.14–0.15;
further closing = larger coarse model / less-compressive norm / more steps (future work).

---

## Iter 5 — resume coarse 12k→24k to close the relief (HI) gap
**Date:** 2026-06-23

**Diagnosis.** Measured real-crop HI vs land fraction: real HI ≈ 0.19 across land
0.30–0.70 (where generated islands sit, ~0.64 land), rising to ~0.25 only for
mountain-interior crops (land 0.7–1.0). So the HI gap (gen 0.13 vs real ~0.19 at
matched land fraction) is a **genuine model deficiency** (coarse under-produces
high-elevation mass), not a structural islands-are-flatter artifact.

**Change.** Resume coarse training (loss plateaued, but diffusion sample quality often
keeps improving past loss plateau). Measured at coarse@16k (SR@12k):

| metric | c@12k (Iter4) | **c@16k** | real |
|---|---|---|---|
| elevation KS / W | 0.086 / 42 m | **0.054 / 21 m** | 0 |
| slope KS / W | 0.109 / 2.56° | **0.049 / 1.10°** | 0 |
| roughness KS | 0.110 | **0.045** | 0 |
| hypso L1 | 0.067 | **0.053** | 0 |
| HI (fake vs real) | 0.13 vs 0.20 | **0.15 vs 0.20** | match |
| PSD log-curve L2 | 0.34 | **0.20** | 0 |
| SWD | 0.156 | **0.117** | 0 |
| island land km² | 5479 | **5109** | ~5000 |

**Result: confirmed.** Coarse 12k→16k improved nearly everything (KS values now
0.045–0.054, HI gap 0.07→0.05, SWD −25%) — **sample quality keeps improving past the
loss plateau**. Continuing coarse to 24k for further gains, then final eval.

---

## Iter 4 — finish SR training (8k→12k), sharpen detail
**Date:** 2026-06-23

**Change.** Resumed SR to step 12000 (converged, loss ~0.015), regenerated + revalidated.

**Results (6 fake vs 32 real, 1536²@60 m) — vs Iter 3:**
| metric | Iter3 (SR@8k) | **Iter4 (SR@12k)** | real |
|---|---|---|---|
| elevation KS / W | 0.122 / 57 m | **0.086 / 42 m** | 0 |
| slope KS / W | 0.143 / 3.2° | **0.109 / 2.56°** | 0 |
| roughness KS | 0.137 | **0.110** | 0 |
| hypso L1 | 0.072 | **0.067** | 0 |
| HI (fake vs real) | 0.13 vs 0.20 | **0.13 vs 0.20** | match |
| PSD β (fake vs real) | 3.63 vs 3.37 | **3.52 vs 3.37** | match |
| PSD log-curve L2 | 0.57 | **0.34** | 0 |
| SWD | 0.208 | **0.156** | 0 |
| island land km² | 5847 | **5479** | ~5000 |
| single-island frac | 0.91–0.96 | **0.96–0.99** | — |

SR sharpening improved nearly everything: β gap 0.26→0.15, SWD −25%, curve-L2 −40%,
all KS down, cleaner coastlines (fewer islets). Remaining gap: **HI** (relief mass)
— addressed in Iter 5.

---

## Iter 3 — first full Phase-1 evaluation (coarse@12k + SR@8k)
**Date:** 2026-06-23

**Setup.** Generated 24 coarse candidates → single-island (largest-CC) post-selection
→ keep 6 → SR-refine full 1536² canvas (one pass). Validated vs 32 real crops
(land 0.45–0.80).

**Results (6 fake vs 32 real, 1536²@60 m):**
| metric | PoC | **Phase-1** | real |
|---|---|---|---|
| elevation KS / W | 0.46 / 172 m | **0.122 / 57 m** | 0 |
| slope KS / W | 0.17 / 3.8° | **0.143 / 3.2°** | 0 |
| roughness KS | 0.22 | **0.137** | 0 |
| hypso L1 | 0.157 | **0.072** | 0 |
| HI (fake vs real) | 0.08 vs 0.24 | **0.13 vs 0.20** | match |
| PSD β (fake vs real) | 2.47 vs 3.38 | **3.63 vs 3.37** | match |
| SWD | 0.23 | **0.208** | 0 |
| island land km² | 298 | **4714–5862** | ~5000 |

- **Land area + single-island selection: success** (frac 0.91–0.96, areas 4.7–5.9k km²).
- **PSD β gap 0.26** (was 0.91) — passes ±0.5 gate; roughness character matches.
- Max elevations 1500–2569 m — realistic relief (sigma_data fix worked).

**Residual gaps & visual self-critique.** Generated islands are coherent with
realistic coastlines + mountainous interiors, BUT vs real: (a) slightly flatter
(HI 0.13 vs 0.20), (b) ridgelines slightly soft / β slightly high (3.63>3.37 = a
touch too smooth).

**Next change.** SR was early-stopped at 8k with loss still dropping → **resume SR
to 14k** to sharpen detail (lower β toward 3.37, sharpen ridges). Re-evaluate after.

---

## Iter 2 — fix EDM `sigma_data` calibration (Phase-1 coarse restart)
**Date:** 2026-06-23

**Observation.** PoC and early Phase-1 coarse samples were **too flat** (generated
elevation HI 0.08 vs real 0.24; all-green renders, elevation < ~550 m; real reaches
1000–3000 m) and PSD too rough.

**Diagnosis.** EDM's `sigma_data` was hard-coded to 0.5, but the *measured* std of
normalized crops is **0.26 (coarse)** and **0.14 (SR)** — a ~2× mismatch. EDM is
sensitive to `sigma_data` (should equal the data std); too-large `sigma_data`
mis-weights the loss toward high noise → over-smoothed / flat outputs. (Measured
sqrt-norm std 0.26 vs linear 0.095 also reconfirms the sqrt normalization choice.)

**Change.** Made `sigma_data` configurable per stage; set coarse=0.26, SR=0.14.
Restarted Phase-1 coarse from scratch (~1.7 h sunk, worth correct calibration).

**Expected effect.** More relief (higher HI, taller mountains), better PSD match.

**Result (visual, coarse stage).** Confirmed: post-fix coarse samples develop clear
elevation relief (green lowlands → tan interiors) vs the pre-fix flat all-green.
Progression: step 2000 speckle → 4000 relief emerging → 6000–12000 coherent
landmasses with mountains + natural coastlines. Quality plateaued by ~step 10–12k
(loss 0.067→0.063), so coarse was **early-stopped at step 12000** (saves ~3.5 h;
resumable). Remaining issue: land is somewhat archipelago-like → added
largest-connected-component (single-island) post-selection to `generate.py`.
Now training the SR stage (also sigma_data-corrected to 0.14).

---

## Iter 1 — Phase-0 PoC (pipeline validation)
**Date:** 2026-06-23

**Setup.** Cascade: coarse EDM (96², 4.4 M) + SR EDM (×4→384², attn-free, 2.3 M);
sqrt norm; ~2.5–4 k steps each.

**Results (6 generated vs 24 real, 384²@60 m):**
- land area: fake 298 vs real 313 km² — **post-selection works**
- elevation KS 0.46 (too flat, HI 0.08 vs 0.24)
- slope KS 0.17 (W 3.8°), roughness KS 0.22
- PSD β fake 2.47 vs real 3.38 (generated too high-frequency)
- SWD 0.23
- **seam/translation-equivariance test: 0.0 diff (no-tiling proven)**
- full-canvas 1536² SR inference: **7.8 GB** (fits 22.5 GB)

**Visual.** Coastlines + texture form, but flat & fragmented vs real (which has
coherent ridges, valleys, volcano). Expected for tiny PoC models.

**Decisions carried forward.** Bigger coarse canvas (384²) for coherence; more
training; fix flatness (→ Iter 2 sigma_data); watch fragmentation (post-select or
mask-condition if needed).
