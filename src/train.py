#!/usr/bin/env python3
"""Train a cascade stage (coarse or SR) with the EDM recipe.

  python src/train.py --config configs/poc.yaml --stage coarse
  python src/train.py --config configs/poc.yaml --stage sr

Single forward-pass / no-tiling friendly: the UNet is translation-equivariant, so
the SR stage trained on patches runs on the full canvas at generation time.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import TerrainMosaic, CoarseDataset, SRDataset, denormalize  # noqa
from networks import UNet  # noqa
from edm import EDM, EMA  # noqa
from render import grid_montage, save_png  # noqa


def cycle(dl):
    while True:
        for b in dl:
            yield b


def build(cfg, stage, device):
    m = cfg[stage]["model"]
    net = UNet(in_ch=1, cond_ch=m.get("cond_ch", 0), out_ch=1, base=m["base"],
               ch_mult=tuple(m["ch_mult"]), num_res=m["num_res"],
               use_attn=m.get("attn", True), dropout=m.get("dropout", 0.0),
               grad_ckpt=m.get("grad_ckpt", False))
    edm = EDM(net, sigma_data=m.get("sigma_data", 0.5)).to(device)
    return edm


def make_loader(cfg, stage, mosaic):
    g = cfg["geometry"]
    tr = cfg[stage]["train"]
    norm = cfg["data"].get("norm", "sqrt")
    if stage == "coarse":
        ds = CoarseDataset(mosaic, g["canvas_px"], g["coarse_px"],
                           vmax=cfg["data"]["vmax"], norm=norm,
                           land_lo=tr["land_lo"], land_hi=tr["land_hi"])
    else:
        ds = SRDataset(mosaic, g["sr_patch_px"], g["sr_factor"],
                       vmax=cfg["data"]["vmax"], norm=norm,
                       land_lo=tr["land_lo"], land_hi=tr["land_hi"])
    dl = DataLoader(ds, batch_size=tr["batch"], shuffle=True, num_workers=8,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4,
                    drop_last=True)
    return dl


@torch.no_grad()
def preview(edm, ema, cfg, stage, device, step, out_dir, sample_batch=None):
    # swap in EMA weights
    backup = copy.deepcopy(edm.net.state_dict())
    edm.net.load_state_dict(ema.shadow)
    edm.eval()
    vmax = cfg["data"]["vmax"]
    res = cfg["data"]["res_m"]
    nm = cfg["data"].get("norm", "sqrt")
    steps = cfg[stage]["sample"]["steps"]
    g = cfg["geometry"]
    try:
        if stage == "coarse":
            n = 8
            x = edm.sample((n, 1, g["coarse_px"], g["coarse_px"]), device,
                           steps=steps, seed=1234)
            dems = [denormalize(x[i, 0].float().cpu().numpy(), vmax, nm) for i in range(n)]
            mont = grid_montage(dems, res=res * (g["canvas_px"] // g["coarse_px"]),
                                vmax=vmax, cols=4, mode="shaded")
            save_png(mont, Path(out_dir) / f"sample_step{step:06d}.png")
        else:
            hi, cond = sample_batch
            hi, cond = hi[:6].to(device), cond[:6].to(device)
            x = edm.sample(hi.shape, device, cond=cond, steps=steps, seed=1234)
            rows = []
            for i in range(hi.shape[0]):
                c = denormalize(cond[i, 0].float().cpu().numpy(), vmax, nm)
                gen = denormalize(x[i, 0].float().cpu().numpy(), vmax, nm)
                real = denormalize(hi[i, 0].float().cpu().numpy(), vmax, nm)
                rows += [c, gen, real]  # cond | generated | real
            mont = grid_montage(rows, res=res, vmax=vmax, cols=3, mode="shaded")
            save_png(mont, Path(out_dir) / f"sample_step{step:06d}.png")
    finally:
        edm.net.load_state_dict(backup)
        edm.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["coarse", "sr"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    name = cfg["name"]
    out_dir = Path("outputs") / name / args.stage
    ck_dir = Path("checkpoints") / name / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path("logs") / name / args.stage)

    print(f"[load] mosaic ...")
    mosaic = TerrainMosaic(cfg["data"]["mosaic"], cfg["data"]["mask"],
                           cfg["data"]["sea_thresh"])
    dl = make_loader(cfg, args.stage, mosaic)
    it = cycle(dl)

    edm = build(cfg, args.stage, device)
    nparams = sum(p.numel() for p in edm.net.parameters()) / 1e6
    print(f"[model] {args.stage}: {nparams:.1f}M params")
    tr = cfg[args.stage]["train"]
    opt = torch.optim.AdamW(edm.net.parameters(), lr=tr["lr"], betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=tr["amp"])
    ema = EMA(edm.net, decay=tr["ema"])

    step0 = 0
    latest = ck_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=device)
        edm.net.load_state_dict(ck["net"])
        ema.shadow = ck["ema"]
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step0 = ck["step"]
        print(f"[resume] from step {step0}")

    max_steps = args.max_steps or tr["steps"]
    # a fixed SR preview batch
    sr_prev = next(it) if args.stage == "sr" else None

    t0 = time.time()
    running = 0.0
    for step in range(step0, max_steps):
        batch = next(it)
        if args.stage == "coarse":
            y = batch.to(device, non_blocking=True)
            cond = None
        else:
            hi, cond = batch
            y = hi.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=tr["amp"], dtype=torch.float16):
            loss = edm.loss(y, cond=cond)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(edm.net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(edm.net)
        running += loss.item()

        if (step + 1) % tr["log_every"] == 0:
            avg = running / tr["log_every"]
            running = 0.0
            ips = tr["log_every"] * tr["batch"] / (time.time() - t0)
            t0 = time.time()
            print(f"[{args.stage}] step {step+1}/{max_steps} loss {avg:.4f} "
                  f"{ips:.0f} img/s", flush=True)
            writer.add_scalar("loss", avg, step + 1)
            writer.add_scalar("img_per_s", ips, step + 1)

        if (step + 1) % tr["sample_every"] == 0:
            torch.cuda.empty_cache()  # release reserved cache before sampling
            preview(edm, ema, cfg, args.stage, device, step + 1, out_dir,
                    sample_batch=sr_prev)
            torch.cuda.empty_cache()

        if (step + 1) % tr["ckpt_every"] == 0 or step + 1 == max_steps:
            ck = {"net": edm.net.state_dict(), "ema": ema.shadow,
                  "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                  "step": step + 1, "cfg": cfg, "stage": args.stage}
            torch.save(ck, latest)
            torch.save(ck, ck_dir / f"ckpt_step{step+1:06d}.pt")
            # disk discipline: keep only the most recent KEEP step-checkpoints
            KEEP = 2
            old = sorted(ck_dir.glob("ckpt_step*.pt"))[:-KEEP]
            for p in old:
                p.unlink()
            print(f"[ckpt] saved step {step+1} (pruned {len(old)} old)", flush=True)

    print("[done] training complete")


if __name__ == "__main__":
    main()
