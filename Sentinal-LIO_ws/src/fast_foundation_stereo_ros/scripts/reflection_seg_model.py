#!/usr/bin/env python3
"""Lightweight RGB/DA3-guided reflection segmentation models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class DSConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyReflectionSeg(nn.Module):
    """Small encoder-decoder for binary reflection-risk masks.

    Input channels:
      3 RGB only
      4 RGB + DA3 normalized depth prior
    Output:
      one logit channel at input resolution
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6

        self.stem = conv_bn_act(in_channels, c1)
        self.enc1 = DSConv(c1, c1)
        self.down2 = DSConv(c1, c2, stride=2)
        self.enc2 = DSConv(c2, c2)
        self.down3 = DSConv(c2, c3, stride=2)
        self.enc3 = DSConv(c3, c3)
        self.down4 = DSConv(c3, c4, stride=2)
        self.bottleneck = nn.Sequential(DSConv(c4, c4), DSConv(c4, c4))

        self.up3 = nn.Sequential(conv_bn_act(c4 + c3, c3), DSConv(c3, c3))
        self.up2 = nn.Sequential(conv_bn_act(c3 + c2, c2), DSConv(c2, c2))
        self.up1 = nn.Sequential(conv_bn_act(c2 + c1, c1), DSConv(c1, c1))
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(self.down2(x1))
        x3 = self.enc3(self.down3(x2))
        x4 = self.bottleneck(self.down4(x3))

        y = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up3(torch.cat([y, x3], dim=1))
        y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, x2], dim=1))
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, x1], dim=1))
        y = self.head(y)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return y


class MobileNetV3ReflectionSeg(nn.Module):
    """MobileNetV3-small encoder with a lightweight FPN-style decoder."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except Exception as exc:
            raise RuntimeError("torchvision is required for MobileNetV3ReflectionSeg") from exc

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained and in_channels == 3 else None
        backbone = mobilenet_v3_small(weights=weights).features
        if in_channels != 3:
            old = backbone[0][0]
            new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                if in_channels > 3:
                    extra = old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
                    new.weight[:, 3:] = extra
            backbone[0][0] = new

        self.backbone = backbone
        self.proj0 = nn.Conv2d(16, 32, 1)
        self.proj1 = nn.Conv2d(24, 48, 1)
        self.proj2 = nn.Conv2d(48, 64, 1)
        self.proj3 = nn.Conv2d(576, 96, 1)
        self.fuse2 = conv_bn_act(64 + 96, 64)
        self.fuse1 = conv_bn_act(48 + 64, 48)
        self.fuse0 = conv_bn_act(32 + 48, 32)
        self.head = nn.Sequential(DSConv(32, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = []
        y = x
        for idx, block in enumerate(self.backbone):
            y = block(y)
            if idx in (0, 3, 8, len(self.backbone) - 1):
                feats.append(y)
        f0, f1, f2, f3 = feats[0], feats[1], feats[2], feats[3]
        p3 = self.proj3(f3)
        p2 = self.proj2(f2)
        p1 = self.proj1(f1)
        p0 = self.proj0(f0)

        y = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse2(torch.cat([p2, y], dim=1))
        y = F.interpolate(y, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse1(torch.cat([p1, y], dim=1))
        y = F.interpolate(y, size=p0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse0(torch.cat([p0, y], dim=1))
        y = self.head(y)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


def create_reflection_model(
    arch: str = "tiny",
    in_channels: int = 3,
    base_channels: int = 32,
    pretrained: bool = False,
) -> nn.Module:
    arch = arch.lower()
    if arch in ("tiny", "tiny_reflection", "tinyseg"):
        return TinyReflectionSeg(in_channels=in_channels, base_channels=base_channels)
    if arch in ("mobilenetv3", "mobilenet_v3", "mbv3"):
        return MobileNetV3ReflectionSeg(in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unsupported reflection segmentation arch: {arch}")


def load_reflection_checkpoint(path: str, device: str = "cpu") -> Dict[str, object]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt
    if isinstance(ckpt, OrderedDict):
        return {"model": ckpt}
    if isinstance(ckpt, dict):
        return {"model": ckpt}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def build_model_from_checkpoint(path: str, device: str = "cuda") -> nn.Module:
    ckpt = load_reflection_checkpoint(path, device="cpu")
    arch = str(ckpt.get("arch", "tiny"))
    in_channels = int(ckpt.get("in_channels", 3))
    base_channels = int(ckpt.get("base_channels", 32))
    model = create_reflection_model(arch, in_channels, base_channels, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    den = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()
