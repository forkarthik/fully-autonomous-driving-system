#!/usr/bin/env python3
"""
TensorRT Engine Exporter & Benchmark Utility for NVIDIA Jetson Orin Nano (JetPack 6)
===================================================================================
Exports PyTorch YOLO models (YOLO11n, YOLOv8n) to an optimized TensorRT engine (.engine)
with FP16 precision, dynamic or fixed batch size, and bench-tests inference latency.

Usage:
    python3 export_tensorrt.py --model yolo11n.pt --imgsz 480 640 --half
"""

import argparse
import sys
import os
import time
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO to TensorRT on NVIDIA Jetson Orin Nano")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to input PyTorch weights (e.g., yolo11n.pt or yolov8n.pt)")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[480, 640], help="Image size [height, width] or [size]")
    parser.add_argument("--half", action="store_true", default=True, help="Enable FP16 precision (Ampere Tensor Core acceleration)")
    parser.add_argument("--device", default="0", help="CUDA device index (default 0)")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace memory limit in GB")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Benchmark exported engine after generation")
    return parser.parse_args()

def check_jetson_environment():
    print("=" * 60)
    print(" Jetson Orin Nano System Environment Check")
    print("=" * 60)
    
    # Check L4T / JetPack
    nv_tegra_release = "/etc/nv_tegra_release"
    if os.path.exists(nv_tegra_release):
        with open(nv_tegra_release, "r") as f:
            print(f"[JetPack OS] {f.readline().strip()}")
    else:
        print("[Host OS] Not running directly on Jetson Tegra board or running inside non-L4T environment.")
        
    try:
        import torch
        print(f"[PyTorch] {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[GPU] {torch.cuda.get_device_name(0)}")
            print(f"[Compute Capability] {torch.cuda.get_device_capability(0)}")
    except Exception as e:
        print(f"[PyTorch Warning] {e}")

    try:
        import tensorrt as trt
        print(f"[TensorRT] Version: {trt.__version__}")
    except ImportError:
        print("[TensorRT] Not installed in current Python environment. Ultralytics export will attempt compilation.")
    print("=" * 60)

def export_engine(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' package is not installed. Please run: pip3 install ultralytics")
        sys.exit(1)

    print(f"\n[1/3] Loading PyTorch model weights: {args.model} ...")
    model = YOLO(args.model)

    # Format image size
    imgsz = args.imgsz
    if len(imgsz) == 1:
        imgsz = (imgsz[0], imgsz[0])
    elif len(imgsz) == 2:
        imgsz = (imgsz[0], imgsz[1])

    print(f"[2/3] Exporting to TensorRT format...")
    print(f"      - Image size: {imgsz}")
    print(f"      - Half (FP16): {args.half}")
    print(f"      - Device: {args.device}")
    print(f"      - Workspace: {args.workspace} GB")
    
    try:
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=args.half,
            device=args.device,
            workspace=args.workspace,
            verbose=True
        )
        print(f"\n[SUCCESS] Engine successfully exported to: {engine_path}")
    except Exception as e:
        print(f"\n[ERROR] TensorRT export failed: {e}")
        print("\nTroubleshooting tips for Jetson Orin Nano:")
        print("1. Ensure TensorRT python bindings are installed: sudo apt-get install python3-libnvinfer")
        print("2. Ensure CUDA and cuDNN paths are in LD_LIBRARY_PATH:")
        print("   export PATH=/usr/local/cuda/bin:$PATH")
        print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        print("3. Check that Jetson power mode is MAXN: sudo nvpmodel -m 0 && sudo jetson_clocks")
        sys.exit(1)

    if args.benchmark and engine_path and os.path.exists(engine_path):
        benchmark_engine(engine_path, imgsz)

def benchmark_engine(engine_path, imgsz):
    print(f"\n[3/3] Benchmarking exported engine: {engine_path} ...")
    from ultralytics import YOLO
    trt_model = YOLO(engine_path, task="detect")
    
    # Create dummy frame matching input size
    dummy_input = np.random.randint(0, 255, (imgsz[0], imgsz[1], 3), dtype=np.uint8)
    
    # Warmup
    print("      Warming up TensorRT GPU execution context (10 runs)...")
    for _ in range(10):
        _ = trt_model(dummy_input, verbose=False)
        
    # Benchmark runs
    num_runs = 50
    print(f"      Measuring inference speed over {num_runs} iterations...")
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = trt_model(dummy_input, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = 1000.0 / avg_latency
    
    print("-" * 60)
    print(f" TensorRT Inference Performance on Input {imgsz[1]}x{imgsz[0]}:")
    print(f"   Average Latency: {avg_latency:.2f} ms")
    print(f"   95th Percentile: {p95_latency:.2f} ms")
    print(f"   Throughput:      {fps:.1f} FPS")
    print("-" * 60)
    if fps >= 30.0:
        print(" [OPTIMAL] Engine exceeds 30 FPS target for real-time edge driving!")
    else:
        print(" [NOTICE] To boost FPS, verify FP16 (--half) is active and Jetson clocks are locked.")

if __name__ == "__main__":
    args = parse_args()
    check_jetson_environment()
    export_engine(args)
