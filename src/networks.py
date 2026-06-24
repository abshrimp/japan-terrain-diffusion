#!/usr/bin/env python3
"""A compact, VRAM-friendly UNet for EDM diffusion over single-channel terrain.

Design notes:
  - Translation-equivariant: NO spatial positional embeddings, so a model trained
    on small patches can run on a FULL large canvas in ONE forward pass (the key
    trick enabling no-tiling generation while keeping training VRAM low).
  - Accepts optional conditioning channels concatenated to the (preconditioned)
    noisy input: 0 -> unconditional, 1 -> land-mask cond (coarse) or low-res cond (SR).
  - GroupNorm + SiLU residual blocks; self-attention only at low resolutions.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def zero_module(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


def gnorm(c, max_groups=32):
    """GroupNorm with the largest power-of-2 group count (<=max_groups) dividing c."""
    g = max_groups
    while c % g != 0:
        g //= 2
    return nn.GroupNorm(g, c)


class FourierEmb(nn.Module):
    """Random Fourier features for log-sigma (EDM-style)."""
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        x = x[:, None] * self.freqs[None] * 2 * math.pi
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.0, pad="reflect"):
        super().__init__()
        self.norm1 = gnorm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode=pad)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = gnorm(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(cout, cout, 3, padding=1, padding_mode=pad))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = gnorm(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = zero_module(nn.Conv2d(c, c, 1))
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))  # B,heads,HW,dim
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1, padding_mode="reflect")

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, cond_ch=0, out_ch=1, base=64,
                 ch_mult=(1, 2, 3, 4), num_res=2, use_attn=True,
                 dropout=0.0, grad_ckpt=False, img_channels=1):
        super().__init__()
        self.cond_ch = cond_ch
        self.use_attn = use_attn
        self.grad_ckpt = grad_ckpt
        emb_dim = base * 4
        self.map = nn.Sequential(FourierEmb(base), nn.Linear(base, emb_dim),
                                 nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.in_conv = nn.Conv2d(in_ch + cond_ch, base, 3, padding=1,
                                 padding_mode="reflect")

        chs = [base]
        cur = base
        self.downs = nn.ModuleList()
        res_levels = len(ch_mult)
        for i, m in enumerate(ch_mult):
            cout = base * m
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(cur, cout, emb_dim, dropout)])
                cur = cout
                blk.append(_LevelAttn(cur, i, res_levels, use_attn))
                self.downs.append(blk)
                chs.append(cur)
            if i != res_levels - 1:
                self.downs.append(nn.ModuleList([Down(cur)]))
                chs.append(cur)

        self.mid = nn.ModuleList([
            ResBlock(cur, cur, emb_dim, dropout),
            Attn(cur) if use_attn else nn.Identity(),
            ResBlock(cur, cur, emb_dim, dropout)])

        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mult))):
            cout = base * m
            for _ in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(cur + chs.pop(), cout, emb_dim, dropout)])
                cur = cout
                blk.append(_LevelAttn(cur, i, res_levels, use_attn))
                self.ups.append(blk)
            if i != 0:
                self.ups.append(nn.ModuleList([Up(cur)]))

        self.out_norm = gnorm(cur)
        self.out_conv = zero_module(nn.Conv2d(cur, out_ch, 3, padding=1,
                                              padding_mode="reflect"))

    def _rb(self, block, h, emb):
        """Run a ResBlock, with gradient checkpointing during training if enabled."""
        if self.grad_ckpt and self.training and h.requires_grad:
            return torch.utils.checkpoint.checkpoint(block, h, emb, use_reentrant=False)
        return block(h, emb)

    def forward(self, x, sigma_emb_in, cond=None):
        emb = self.map(sigma_emb_in)
        if self.cond_ch and cond is not None:
            x = torch.cat([x, cond], dim=1)
        h = self.in_conv(x)
        hs = [h]
        for blk in self.downs:
            if isinstance(blk[0], Down):
                h = blk[0](h)
            else:
                h = self._rb(blk[0], h, emb)
                h = blk[1](h)
            hs.append(h)
        h = self._rb(self.mid[0], h, emb)
        h = self.mid[1](h)
        h = self._rb(self.mid[2], h, emb)
        for blk in self.ups:
            if isinstance(blk[0], Up):
                h = blk[0](h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = self._rb(blk[0], h, emb)
                h = blk[1](h)
        return self.out_conv(F.silu(self.out_norm(h)))


class _LevelAttn(nn.Module):
    """Self-attention on the two deepest levels only (small maps -> cheap).
    Disabled entirely when use_attn=False (SR stage -> translation-equivariant)."""
    def __init__(self, c, level, n_levels, use_attn):
        super().__init__()
        # attention only at the DEEPEST level (smallest map) -> avoids the O(N^2)
        # memory blowup at 96^2 (the mid-block already attends at the bottleneck).
        self.enabled = bool(use_attn) and (level == n_levels - 1)
        self.attn = Attn(c) if self.enabled else None

    def forward(self, x):
        return self.attn(x) if self.enabled else x
