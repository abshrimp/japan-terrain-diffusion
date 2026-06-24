#!/usr/bin/env python3
"""Fast correctness checks for model / EDM / metrics / seam (no-tiling) test.
Run BEFORE the PoC to catch shape/logic bugs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
import torch

from networks import UNet
from edm import EDM, EMA
import metrics as M
from dataset import normalize, denormalize

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
ok = True

# 1. coarse model: loss + sample
net_c = UNet(in_ch=1, cond_ch=0, base=32, ch_mult=(1, 2, 2), num_res=1, use_attn=True)
edm_c = EDM(net_c).to(dev)
print("coarse params (M):", round(sum(p.numel() for p in net_c.parameters()) / 1e6, 2))
y = torch.randn(4, 1, 96, 96, device=dev)
l = edm_c.loss(y)
l.backward()
print("coarse loss:", float(l), "ok" if torch.isfinite(l) else "NAN")
with torch.no_grad():
    s = edm_c.sample((2, 1, 96, 96), dev, steps=8, seed=0)
print("coarse sample shape:", tuple(s.shape), "range", round(float(s.min()), 2), round(float(s.max()), 2))

# 2. SR model: conditional loss + sample
net_s = UNet(in_ch=1, cond_ch=1, base=32, ch_mult=(1, 2, 2), num_res=1, use_attn=False)
edm_s = EDM(net_s).to(dev)
print("sr params (M):", round(sum(p.numel() for p in net_s.parameters()) / 1e6, 2))
hi = torch.randn(4, 1, 128, 128, device=dev)
cond = torch.randn(4, 1, 128, 128, device=dev)
ls = edm_s.loss(hi, cond=cond)
ls.backward()
print("sr loss:", float(ls), "ok" if torch.isfinite(ls) else "NAN")
with torch.no_grad():
    ss = edm_s.sample((2, 1, 128, 128), dev, cond=cond[:2], steps=8, seed=0)
print("sr sample shape:", tuple(ss.shape))

# 3. SEAM / translation-equivariance test (the no-tiling proof):
#    attn-free + reflect-pad net must give the SAME interior output whether run on
#    a full canvas or on an aligned crop containing that interior's receptive field.
net_s.eval()
with torch.no_grad():
    big = torch.randn(1, 1, 192, 192, device=dev)
    sig = torch.full((1,), 1.0, device=dev)
    out_big = edm_s.precond(big, sig, cond=torch.zeros_like(big))
    crop = big[:, :, 48:144, 48:144]  # center 96, offset 48 (div by 4 -> aligned)
    out_crop = edm_s.precond(crop, sig, cond=torch.zeros_like(crop))
    # compare global region [64:128] : in big = [64:128]; in crop = [16:80]
    a = out_big[:, :, 64:128, 64:128]
    b = out_crop[:, :, 16:80, 16:80]
    diff = (a - b).abs().max().item()
print(f"seam/equivariance max interior diff: {diff:.2e} "
      f"({'PASS <1e-3' if diff < 1e-3 else 'CHECK'})")
if diff >= 1e-3:
    ok = False

# 4. normalization round-trip
h = np.array([0, 1, 10, 100, 1000, 3776], np.float32)
for nm in ("sqrt", "linear", "log"):
    rt = denormalize(normalize(h, 3776, nm), 3776, nm)
    err = np.abs(rt - h).max()
    print(f"norm[{nm}] round-trip max err: {err:.3e}", "PASS" if err < 1e-2 else "FAIL")
    if err >= 1e-2:
        ok = False

# 5. metrics on synthetic terrain sets
def synth(seed, rough=1.0):
    rng = np.random.RandomState(seed)
    f = np.fft.fftfreq(128)
    kx, ky = np.meshgrid(f, f)
    k = np.hypot(kx, ky); k[0, 0] = 1
    amp = k ** (-1.8 * rough)
    ph = np.exp(2j * np.pi * rng.rand(128, 128))
    field = np.fft.ifft2(np.fft.fftshift(amp) * ph).real
    field = (field - field.min()) / (field.ptp() + 1e-9) * 1500
    field[field < 200] = 0  # sea
    return field.astype(np.float32)

real = [synth(i, 1.0) for i in range(6)]
fake = [synth(100 + i, 1.05) for i in range(6)]
res = M.compare_all(real, fake, res=60, vmax=3776)
print("--- metrics smoke ---")
print(M.summarize(res))

print("\nSMOKE", "PASS" if ok else "ISSUES FOUND")
