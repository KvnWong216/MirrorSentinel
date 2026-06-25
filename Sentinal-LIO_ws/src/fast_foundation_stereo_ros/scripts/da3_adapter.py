#!/usr/bin/env python3
"""Runtime adapter for monocular DA3-style depth priors.

The official Depth Anything V3 code/weights are intentionally kept outside this
repository.  This adapter gives the ROS nodes a stable interface while allowing
the actual DA3 implementation to be plugged in as a Python module, TorchScript,
or ONNX model.  The heuristic backend is for plumbing tests only.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import torch


def _to_rgb_float_tensor(rgb: np.ndarray, image_size: Optional[Tuple[int, int]], device: str) -> torch.Tensor:
    if image_size is not None:
        width, height = image_size
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def normalize_depth_prior(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth) & (depth > 0)
    if not finite.any():
        return np.zeros(depth.shape, dtype=np.float32)
    vals = depth[finite]
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if hi <= lo:
        return np.zeros(depth.shape, dtype=np.float32)
    norm = (depth - lo) / (hi - lo)
    norm[~finite] = 0.0
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def heuristic_depth(rgb: np.ndarray) -> np.ndarray:
    """A deterministic pseudo-depth fallback for ROS wiring tests.

    This is not DA3 and must not be reported as DA3 in experiments.
    """

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 5.0)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    inv_depth = 0.55 * (1.0 - blur) + 0.45 * cv2.GaussianBlur(edges, (0, 0), 3.0)
    inv_depth = np.clip(inv_depth, 0.0, 1.0)
    return (1.0 + 8.0 * (1.0 - inv_depth)).astype(np.float32)


@dataclass
class DA3Config:
    backend: str = "none"
    model: str = ""
    checkpoint: str = ""
    device: str = "cuda"
    input_width: int = 518
    input_height: int = 518
    metric_scale: float = 1.0
    metric_shift: float = 0.0
    min_depth: float = 0.05
    max_depth: float = 80.0


class DA3DepthRunner:
    def __init__(self, config: DA3Config):
        self.config = config
        self.backend = config.backend.lower()
        self.device = config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu"
        self.model: Any = None
        self.session: Any = None
        self.input_name: Optional[str] = None

        if self.backend in ("none", ""):
            return
        if self.backend == "heuristic":
            return
        if self.backend == "torchscript":
            if not config.checkpoint:
                raise ValueError("da3 torchscript backend requires checkpoint")
            self.model = torch.jit.load(config.checkpoint, map_location=self.device).eval()
            return
        if self.backend == "onnx":
            if not config.checkpoint:
                raise ValueError("da3 onnx backend requires checkpoint")
            import onnxruntime as ort  # type: ignore

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.startswith("cuda") else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(config.checkpoint, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            return
        if self.backend == "module":
            self.model = self._load_module_model(config.model, config.checkpoint)
            return
        raise ValueError(f"Unsupported DA3 backend: {config.backend}")

    def _load_module_model(self, spec: str, checkpoint: str) -> Any:
        if ":" not in spec:
            raise ValueError("DA3 module spec must be 'package.module:callable_or_object'")
        module_name, attr_name = spec.split(":", 1)
        attr = getattr(importlib.import_module(module_name), attr_name)

        if hasattr(attr, "from_pretrained"):
            model = attr.from_pretrained(checkpoint) if checkpoint else attr.from_pretrained()
        elif isinstance(attr, type):
            try:
                model = attr(checkpoint=checkpoint) if checkpoint else attr()
            except TypeError:
                model = attr(checkpoint) if checkpoint else attr()
        else:
            model = attr

        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model = model.eval()
        return model

    def enabled(self) -> bool:
        return self.backend not in ("none", "")

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.backend in ("none", ""):
            raise RuntimeError("DA3 backend is disabled")
        if self.backend == "heuristic":
            depth = heuristic_depth(rgb)
        elif self.backend == "torchscript":
            depth = self._infer_torchscript(rgb)
        elif self.backend == "onnx":
            depth = self._infer_onnx(rgb)
        elif self.backend == "module":
            depth = self._infer_module(rgb)
        else:
            raise RuntimeError(f"Unsupported DA3 backend: {self.backend}")

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = np.squeeze(depth)
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = depth * float(self.config.metric_scale) + float(self.config.metric_shift)
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        depth[(depth < self.config.min_depth) | (depth > self.config.max_depth)] = 0.0
        return depth

    def _infer_torchscript(self, rgb: np.ndarray) -> np.ndarray:
        size = (self.config.input_width, self.config.input_height)
        tensor = _to_rgb_float_tensor(rgb, size, self.device)
        with torch.inference_mode():
            out = self.model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().float().cpu().numpy().squeeze()

    def _infer_onnx(self, rgb: np.ndarray) -> np.ndarray:
        size = (self.config.input_width, self.config.input_height)
        tensor = _to_rgb_float_tensor(rgb, size, "cpu").cpu().numpy()
        out = self.session.run(None, {self.input_name: tensor})[0]
        return np.asarray(out).squeeze()

    def _infer_module(self, rgb: np.ndarray) -> np.ndarray:
        model = self.model
        if hasattr(model, "infer_image"):
            return model.infer_image(rgb)
        if hasattr(model, "infer"):
            return model.infer(rgb)
        if hasattr(model, "predict"):
            return model.predict(rgb)

        size = (self.config.input_width, self.config.input_height)
        tensor = _to_rgb_float_tensor(rgb, size, self.device)
        with torch.inference_mode():
            out = model(tensor)
        if isinstance(out, dict):
            out = out.get("depth", next(iter(out.values())))
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            out = out.detach().float().cpu().numpy()
        return np.asarray(out).squeeze()


def make_da3_config_from_ros(node, prefix: str = "da3_") -> DA3Config:
    return DA3Config(
        backend=node.get_parameter(prefix + "backend").get_parameter_value().string_value,
        model=node.get_parameter(prefix + "model").get_parameter_value().string_value,
        checkpoint=node.get_parameter(prefix + "checkpoint").get_parameter_value().string_value,
        device=node.get_parameter(prefix + "device").get_parameter_value().string_value,
        input_width=node.get_parameter(prefix + "input_width").get_parameter_value().integer_value,
        input_height=node.get_parameter(prefix + "input_height").get_parameter_value().integer_value,
        metric_scale=node.get_parameter(prefix + "metric_scale").get_parameter_value().double_value,
        metric_shift=node.get_parameter(prefix + "metric_shift").get_parameter_value().double_value,
        min_depth=node.get_parameter(prefix + "min_depth").get_parameter_value().double_value,
        max_depth=node.get_parameter(prefix + "max_depth").get_parameter_value().double_value,
    )
