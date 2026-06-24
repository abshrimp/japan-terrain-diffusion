# Model checkpoints

The trained model weights (`*.pt`) are **excluded from git** (see `.gitignore`) because
they are large binaries. This directory and this note are tracked so the layout is clear.

Final (shipped) models — present locally, EMA weights only (inference-ready):

| file | stage | step | size |
|---|---|---|---|
| `phase1/coarse/coarse_step029000.pt` | coarse (大局) | 29000 | ~73 MB |
| `phase1/sr/sr_step012000.pt`         | SR (詳細)      | 12000 | ~67 MB |

The coarse model uses relief/land-fraction-matched sampling (Iter 8); SR is the 12k
checkpoint. See `reports/ITERATION_LOG.md` and `reports/RESULTS.md`.

## How to obtain
- **Train them** (see top-level `README.md`):
  ```
  python src/train.py --config configs/phase1.yaml --stage coarse
  python src/train.py --config configs/phase1.yaml --stage sr
  ```
  (Training writes full checkpoints to `checkpoints/phase1/<stage>/latest.pt`.)
- Or distribute these EMA-only files via a **GitHub Release asset** or **Git LFS**.

## How to use
```
python src/generate.py --config configs/phase1.yaml \
  --coarse-ckpt checkpoints/phase1/coarse/coarse_step029000.pt \
  --sr-ckpt     checkpoints/phase1/sr/sr_step012000.pt \
  --n 24 --keep 6 --target-km2 5000
```
Note: these are EMA-only checkpoints (optimizer state stripped) — usable for generation,
not for `--resume`. Retrain from `configs/phase1.yaml` if you need to continue training.
