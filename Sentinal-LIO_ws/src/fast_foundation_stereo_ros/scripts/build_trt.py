#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import tensorrt as trt


def compile_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    try:
        builder = trt.Builder(logger)
    except Exception as e:
        print(f"[!] Builder init failed: {e}")
        print("[*] Hint: Try adding your TensorRT runtime directory to LD_LIBRARY_PATH")
        return False

    if not onnx_path.exists():
        print(f"[!] ONNX file not found: {onnx_path}")
        return False

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    print(f"\n[*] Parsing {onnx_path}...")
    with onnx_path.open('rb') as f:
        if not parser.parse(f.read()):
            print("[!] Parse error!")
            return False

    print("[*] Building TensorRT engine...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("[!] Build failed!")
        return False

    with engine_path.open('wb') as f:
        f.write(plan)
    print(f"[+] Success: {engine_path}")
    return True


def default_model_root() -> Path:
    env_root = os.environ.get("SENTINEL_LIO_MODEL_ROOT")
    if env_root:
        return Path(env_root).expanduser()

    for base in Path(__file__).resolve().parents:
        candidate = base / "models" / "Fast-FoundationStereo"
        if candidate.exists():
            return candidate

    return Path(__file__).resolve().parents[3] / "models" / "Fast-FoundationStereo"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Fast-FoundationStereo TensorRT engines from ONNX files.")
    parser.add_argument(
        "--model-root",
        default=str(default_model_root()),
        help="Fast-FoundationStereo model root. Defaults to SENTINEL_LIO_MODEL_ROOT or this workspace's models/Fast-FoundationStereo.")
    parser.add_argument(
        "--onnx-dir",
        default="",
        help="Directory containing feature_runner.onnx and post_runner.onnx. Defaults to <model-root>/output.")
    parser.add_argument(
        "--engine-dir",
        default="",
        help="Output directory for .engine files. Defaults to --onnx-dir.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_root = Path(args.model_root).expanduser().resolve()
    onnx_dir = Path(args.onnx_dir).expanduser().resolve() if args.onnx_dir else model_root / "output"
    engine_dir = Path(args.engine_dir).expanduser().resolve() if args.engine_dir else onnx_dir

    ok_feature = compile_engine(
        onnx_dir / "feature_runner.onnx",
        engine_dir / "feature_runner.engine")
    ok_post = compile_engine(
        onnx_dir / "post_runner.onnx",
        engine_dir / "post_runner.engine")

    raise SystemExit(0 if ok_feature and ok_post else 1)
