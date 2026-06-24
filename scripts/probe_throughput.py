#!/usr/bin/env python3
"""Measure real train throughput + VRAM for Phase-1 configs, and the critical
full-canvas SR inference VRAM at 1536^2 (must fit 22.5 GB)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch, yaml
from networks import UNet
from edm import EDM

dev = "cuda"
cfg = yaml.safe_load(open("configs/phase1.yaml"))
g = cfg["geometry"]


def bench_train(stage, size, batch, iters=15):
    m = cfg[stage]["model"]
    net = UNet(in_ch=1, cond_ch=m.get("cond_ch", 0), base=m["base"],
               ch_mult=tuple(m["ch_mult"]), num_res=m["num_res"],
               use_attn=m.get("attn", True)).to(dev)
    edm = EDM(net).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    y = torch.randn(batch, 1, size, size, device=dev)
    cond = torch.randn(batch, 1, size, size, device=dev) if m.get("cond_ch") else None
    # warmup
    for _ in range(3):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = edm.loss(y, cond=cond)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = edm.loss(y, cond=cond)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize(); dt = time.time() - t0
    ips = iters * batch / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[train {stage}] {size}^2 batch{batch} {params:.0f}M params: "
          f"{ips:.1f} img/s, {dt/iters*1000:.0f} ms/step, peak {peak:.1f} GB")
    del net, edm, opt, y, cond; torch.cuda.empty_cache()
    return ips


def bench_sr_infer(canvas, steps=8):
    m = cfg["sr"]["model"]
    net = UNet(in_ch=1, cond_ch=1, base=m["base"], ch_mult=tuple(m["ch_mult"]),
               num_res=m["num_res"], use_attn=False).to(dev).eval()
    edm = EDM(net).to(dev)
    torch.cuda.reset_peak_memory_stats()
    cond = torch.randn(1, 1, canvas, canvas, device=dev)
    t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        x = edm.sample((1, 1, canvas, canvas), dev, cond=cond, steps=steps, seed=0)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[infer SR] full canvas {canvas}^2, {steps} steps: "
          f"{time.time()-t0:.1f}s, peak {peak:.1f} GB  "
          f"{'FITS 22.5GB' if peak < 21 else 'TIGHT/OOM-RISK'}")
    del net, edm, cond, x; torch.cuda.empty_cache()


def bench_coarse_infer(size, steps=48, n=8):
    m = cfg["coarse"]["model"]
    net = UNet(in_ch=1, cond_ch=0, base=m["base"], ch_mult=tuple(m["ch_mult"]),
               num_res=m["num_res"], use_attn=True).to(dev).eval()
    edm = EDM(net).to(dev)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        x = edm.sample((n, 1, size, size), dev, steps=steps, seed=0)
    torch.cuda.synchronize()
    print(f"[infer coarse] {n}x {size}^2, {steps} steps: {time.time()-t0:.1f}s, "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    del net, edm, x; torch.cuda.empty_cache()


print("=== Phase-1 throughput / VRAM probe ===")
bench_train("coarse", g["coarse_px"], cfg["coarse"]["train"]["batch"])
bench_train("sr", g["sr_patch_px"], cfg["sr"]["train"]["batch"])
bench_coarse_infer(g["coarse_px"])
bench_sr_infer(g["canvas_px"])  # the critical 1536^2 full-canvas test
