#!/usr/bin/env python3
"""
Unified Benchmark Script for KV Cache Compression Methods

This script benchmarks multiple KV cache compression methods on the PG-19 dataset.

Usage Examples:
    # Test L2 compression (KnormPress)
    python scripts/benchmark.py --method l2_compress --keep_ratios 1.0,0.8,0.5

    # Test fixed-size L2 compression
    python scripts/benchmark.py --method fix_size_l2 --fix_kv_sizes 256,512 --strategies keep_low

    # Test StreamingLLM
    python scripts/benchmark.py --method streaming_llm --start_size 4 --recent_sizes 252,508,1020

    # Test H2O-L2
    python scripts/benchmark.py --method h2o_l2 --heavy_hitter_sizes 32,64,128

    # Test SnapKV-Lite
    python scripts/benchmark.py --method snapkv_lite --observation_windows 16,32,64

    # Test Pyramid KV
    python scripts/benchmark.py --method pyramid_kv --base_sizes 256,512

    # Test Adaptive L2
    python scripts/benchmark.py --method adaptive_l2 --target_sizes 256,512

    # Compare all methods
    python scripts/benchmark.py --compare_all

    # Compare new methods only
    python scripts/benchmark.py --compare_new

Supported Methods:
    - l2_compress: KnormPress ratio-based compression
    - fix_size_l2: Fixed-size KV cache with L2-based eviction
    - streaming_llm: StreamingLLM with attention sinks
    - h2o_l2: H2O-inspired with L2 norm approximation
    - snapkv_lite: SnapKV-inspired observation window voting
    - pyramid_kv: Layer-wise adaptive compression
    - adaptive_l2: Dynamic sequence-length aware compression
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from kvcompress.methods import (
    get_compress_fn, list_methods,
    l2_compress, fix_size_l2_compress, streaming_llm_compress, recent_only_compress,
    h2o_l2_compress, snapkv_lite_compress, pyramid_kv_compress, adaptive_l2_compress
)
from kvcompress.benchmark import benchmark, run_benchmark_suite, print_benchmark_summary
from kvcompress.evaluate import evaluate_with_compression


# Dataset paths
LOCAL_PG19_PATH = os.path.join(project_root, "data", "pg19.parquet")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str = "EleutherAI/pythia-2.8b"):
    """
    Load model and tokenizer using Auto classes.
    
    AutoModelForCausalLM automatically detects the model architecture from config.json
    and loads the appropriate model class (e.g., GPTNeoXForCausalLM for Pythia,
    LlamaForCausalLM for Llama, etc.)
    
    Args:
        model_id: HuggingFace model ID or local path
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    print(f"Loading model: {model_id}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Use AutoModelForCausalLM for automatic model class detection
    # This provides better flexibility for different model architectures
    import os
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"  # Enable transformers logging
    
    print("Loading model (this may take a while for large models)...")
    
    # Try offline first to avoid network timeouts for cached models
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            local_files_only=True,  # Try cache first
        )
        print("Model loaded from cache")
    except (OSError, ValueError) as e:
        # Model not in cache, download it
        print(f"Model not in cache, downloading from HuggingFace Hub...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            local_files_only=False,  # Allow download
        )
        print("Model downloaded and loaded")
    
    print("Moving model to device...")
    model.to(device)
    model.eval()
    print("Model ready!")
    
    # Same logic for tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=False
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def warmup_model(model, tokenizer, device, num_warmup: int = 3):
    """
    Perform warmup runs to eliminate cold-start effects.
    
    This helps avoid abnormally high TTFT in the first benchmark run by:
    1. Warming up GPU/MPS
    2. Triggering PyTorch JIT compilation
    3. Pre-allocating memory
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Device to use
        num_warmup: Number of warmup iterations
    """
    print(f"\nPerforming {num_warmup} warmup iterations...")
    
    # Simple warmup text
    warmup_text = "The quick brown fox jumps over the lazy dog. " * 10
    input_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        for i in range(num_warmup):
            # Forward pass (prefill)
            outputs = model(input_ids, use_cache=True)
            
            # Generate a few tokens
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            
            for _ in range(10):  # Generate 10 tokens
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            
            print(f"  Warmup {i+1}/{num_warmup} completed")
    
    # Clear any cached memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print("  Warmup finished!\n")


def load_pg19_samples(num_samples: int = 3):
    """Load samples from PG-19 dataset."""
    print("\nLoading PG-19 dataset...")
    
    # Try local file first
    if os.path.exists(LOCAL_PG19_PATH):
        print(f"  Found local file: {LOCAL_PG19_PATH}")
        try:
            dataset = load_dataset(
                "parquet",
                data_files={'test': LOCAL_PG19_PATH},
                split="test"
            )
            print(f"  Loaded {len(dataset)} samples from local file")
        except Exception as e:
            print(f"  Failed to load local file: {e}")
            dataset = None
    else:
        dataset = None
    
    # Fallback to HuggingFace
    if dataset is None:
        try:
            print("  Loading from HuggingFace...")
            dataset = load_dataset("pg19", split="test")
        except Exception as e:
            print(f"  Failed to load from HuggingFace: {e}")
            return []
    
    # Extract text samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        text = sample.get("text", "")
        if len(text) > 10000:
            samples.append(text)
            print(f"  Sample {i+1}: {len(text)} characters")
    
    print(f"  Total: {len(samples)} samples loaded")
    return samples


def build_methods_config(args) -> list:
    """Build methods configuration based on command line arguments."""
    methods = []
    
    # Always add baseline if not disabled
    if not args.no_baseline:
        methods.append({
            "name": "baseline",
            "compress_fn": None,
            "kwargs": {}
        })
    
    if args.method == "l2_compress":
        # L2 compression with different keep ratios (no recent_only control for dynamic size)
        keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
        for kr in keep_ratios:
            if kr >= 1.0 and not args.no_baseline:
                continue  # Skip duplicate baseline
            methods.append({
                "name": f"l2_kr={kr:.1f}",
                "compress_fn": l2_compress,
                "kwargs": {
                    "keep_ratio": kr,
                    "prune_after": args.prune_after,
                }
            })
    
    elif args.method == "fix_size_l2":
        # Fixed-size L2 compression
        fix_kv_sizes = [int(x) for x in args.fix_kv_sizes.split(",")]
        strategies = [x.strip() for x in args.strategies.split(",")]
        keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
        
        # Add recent_only control group for each fix_kv_size if not disabled
        if not args.no_recent_only:
            for fix_size in fix_kv_sizes:
                methods.append({
                    "name": f"recent_only_{fix_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": fix_size,
                    }
                })
        
        for fix_size in fix_kv_sizes:
            for strategy in strategies:
                for kr in keep_ratios:
                    methods.append({
                        "name": f"fix{fix_size}_{strategy}_kr={kr:.1f}",
                        "compress_fn": fix_size_l2_compress,
                        "kwargs": {
                            "fix_kv_size": fix_size,
                            "strategy": strategy,
                            "keep_ratio": kr,
                        }
                    })
    
    elif args.method == "streaming_llm":
        # StreamingLLM with different cache sizes
        recent_sizes = [int(x) for x in args.recent_sizes.split(",")]
        
        # Add recent_only control group for each total cache size if not disabled
        if not args.no_recent_only:
            for recent_size in recent_sizes:
                total_size = args.start_size + recent_size
                methods.append({
                    "name": f"recent_only_{total_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": total_size,
                    }
                })
        
        for recent_size in recent_sizes:
            total_size = args.start_size + recent_size
            methods.append({
                "name": f"streaming_{total_size}",
                "compress_fn": streaming_llm_compress,
                "kwargs": {
                    "start_size": args.start_size,
                    "recent_size": recent_size,
                }
            })
    
    elif args.method == "h2o_l2":
        # H2O-L2 with different heavy hitter sizes
        heavy_hitter_sizes = [int(x) for x in args.heavy_hitter_sizes.split(",")]
        
        # Add recent_only control group if not disabled
        if not args.no_recent_only:
            for hh_size in heavy_hitter_sizes:
                total_size = args.start_size + hh_size + args.h2o_recent_size
                methods.append({
                    "name": f"recent_only_{total_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": total_size,
                    }
                })
        
        for hh_size in heavy_hitter_sizes:
            total_size = args.start_size + hh_size + args.h2o_recent_size
            methods.append({
                "name": f"h2o_l2_{total_size}",
                "compress_fn": h2o_l2_compress,
                "kwargs": {
                    "start_size": args.start_size,
                    "heavy_hitter_size": hh_size,
                    "recent_size": args.h2o_recent_size,
                }
            })
    
    elif args.method == "snapkv_lite":
        # SnapKV-Lite with different observation windows
        observation_windows = [int(x) for x in args.observation_windows.split(",")]
        keep_sizes = [int(x) for x in args.snapkv_keep_sizes.split(",")]
        
        for keep_size in keep_sizes:
            # Add recent_only control group if not disabled
            if not args.no_recent_only:
                methods.append({
                    "name": f"recent_only_{keep_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": keep_size,
                    }
                })
            
            for obs_win in observation_windows:
                methods.append({
                    "name": f"snapkv_{keep_size}_obs{obs_win}",
                    "compress_fn": snapkv_lite_compress,
                    "kwargs": {
                        "observation_window": obs_win,
                        "keep_size": keep_size,
                    }
                })
    
    elif args.method == "pyramid_kv":
        # Pyramid KV with different base sizes
        base_sizes = [int(x) for x in args.base_sizes.split(",")]
        
        for base_size in base_sizes:
            # Add recent_only control group if not disabled
            if not args.no_recent_only:
                methods.append({
                    "name": f"recent_only_{base_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": base_size,
                    }
                })
            
            methods.append({
                "name": f"pyramid_{base_size}",
                "compress_fn": pyramid_kv_compress,
                "kwargs": {
                    "base_size": base_size,
                    "layer_decay": args.layer_decay,
                    "min_size": args.min_size,
                    "profile": args.pyramid_profile,
                }
            })
    
    elif args.method == "adaptive_l2":
        # Adaptive L2 with different target sizes
        target_sizes = [int(x) for x in args.target_sizes.split(",")]
        
        for target_size in target_sizes:
            # Add recent_only control group if not disabled
            if not args.no_recent_only:
                methods.append({
                    "name": f"recent_only_{target_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": target_size,
                    }
                })
            
            methods.append({
                "name": f"adaptive_{target_size}",
                "compress_fn": adaptive_l2_compress,
                "kwargs": {
                    "target_size": target_size,
                    "soft_limit": args.soft_limit,
                    "hard_limit": args.hard_limit,
                }
            })
    
    elif args.compare_all:
        # Comprehensive comparison: all methods at 512 and 1024 cache sizes with control groups
        methods.extend([
            # ========== 512 Cache Size ==========
            # Control group: Recent-Only 512
            {
                "name": "recent_only_512",
                "compress_fn": recent_only_compress,
                "kwargs": {"window_size": 512}
            },
            # StreamingLLM 512
            {
                "name": "streaming_512",
                "compress_fn": streaming_llm_compress,
                "kwargs": {"start_size": 4, "recent_size": 508}
            },
            # H2O-L2 512
            {
                "name": "h2o_l2_512",
                "compress_fn": h2o_l2_compress,
                "kwargs": {"start_size": 4, "heavy_hitter_size": 64, "recent_size": 444}
            },
            # SnapKV-Lite 512
            {
                "name": "snapkv_512",
                "compress_fn": snapkv_lite_compress,
                "kwargs": {"observation_window": 32, "keep_size": 512}
            },
            # Pyramid KV 512
            {
                "name": "pyramid_512",
                "compress_fn": pyramid_kv_compress,
                "kwargs": {"base_size": 512, "layer_decay": 0.9, "min_size": 64}
            },
            # Adaptive L2 512
            {
                "name": "adaptive_512",
                "compress_fn": adaptive_l2_compress,
                "kwargs": {"target_size": 512, "soft_limit": 256, "hard_limit": 1024}
            },
            # Fix-Size L2 512
            {
                "name": "fix_l2_512",
                "compress_fn": fix_size_l2_compress,
                "kwargs": {"fix_kv_size": 512, "strategy": "keep_low", "keep_ratio": 0.5}
            },
            
            # ========== 1024 Cache Size ==========
            # Control group: Recent-Only 1024
            {
                "name": "recent_only_1024",
                "compress_fn": recent_only_compress,
                "kwargs": {"window_size": 1024}
            },
            # StreamingLLM 1024
            {
                "name": "streaming_1024",
                "compress_fn": streaming_llm_compress,
                "kwargs": {"start_size": 4, "recent_size": 1020}
            },
            # H2O-L2 1024
            {
                "name": "h2o_l2_1024",
                "compress_fn": h2o_l2_compress,
                "kwargs": {"start_size": 4, "heavy_hitter_size": 128, "recent_size": 892}
            },
            # SnapKV-Lite 1024
            {
                "name": "snapkv_1024",
                "compress_fn": snapkv_lite_compress,
                "kwargs": {"observation_window": 64, "keep_size": 1024}
            },
            # Pyramid KV 1024
            {
                "name": "pyramid_1024",
                "compress_fn": pyramid_kv_compress,
                "kwargs": {"base_size": 1024, "layer_decay": 0.9, "min_size": 128}
            },
            # Adaptive L2 1024
            {
                "name": "adaptive_1024",
                "compress_fn": adaptive_l2_compress,
                "kwargs": {"target_size": 1024, "soft_limit": 512, "hard_limit": 2048}
            },
            # Fix-Size L2 1024
            {
                "name": "fix_l2_1024",
                "compress_fn": fix_size_l2_compress,
                "kwargs": {"fix_kv_size": 1024, "strategy": "keep_low", "keep_ratio": 0.5}
            },
        ])
    
    return methods


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark for KV Cache Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # L2 compression (KnormPress)
  python scripts/benchmark.py --method l2_compress --keep_ratios 1.0,0.8,0.5

  # Fixed-size L2
  python scripts/benchmark.py --method fix_size_l2 --fix_kv_sizes 256,512 --strategies keep_low

  # StreamingLLM
  python scripts/benchmark.py --method streaming_llm --start_size 4 --recent_sizes 252,508

  # H2O-L2 (NEW)
  python scripts/benchmark.py --method h2o_l2 --heavy_hitter_sizes 32,64,128

  # SnapKV-Lite (NEW)
  python scripts/benchmark.py --method snapkv_lite --observation_windows 16,32,64

  # Pyramid KV (NEW)
  python scripts/benchmark.py --method pyramid_kv --base_sizes 256,512

  # Adaptive L2 (NEW)
  python scripts/benchmark.py --method adaptive_l2 --target_sizes 256,512

  # Compare all methods
  python scripts/benchmark.py --compare_all

  # Compare new methods only
  python scripts/benchmark.py --compare_new
        """
    )
    
    # General arguments
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-2.8b",
                       help="Model ID from HuggingFace")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of PG-19 samples to test")
    parser.add_argument("--max_tokens", type=int, default=2000,
                       help="Maximum tokens for PPL evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                       help="Number of tokens to generate for TTFT/TPOT")
    parser.add_argument("--skip_layers", type=str, default="0,1",
                       help="Comma-separated layer indices to skip")
    parser.add_argument("--no_baseline", action="store_true",
                       help="Skip baseline (no compression) benchmark")
    parser.add_argument("--no_recent_only", action="store_true",
                       help="Skip recent_only (sliding window) control group for fixed-size methods")
    parser.add_argument("--num_warmup", type=int, default=3,
                       help="Number of warmup iterations before benchmark (default: 3)")
    
    # Method selection
    parser.add_argument("--method", type=str, 
                       choices=["l2_compress", "fix_size_l2", "streaming_llm", 
                               "h2o_l2", "snapkv_lite", "pyramid_kv", "adaptive_l2"],
                       help="Compression method to benchmark")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all original methods with default configurations")
    parser.add_argument("--compare_new", action="store_true",
                       help="Compare new methods (H2O-L2, SnapKV-Lite, Pyramid, Adaptive)")
    
    # L2 compress arguments
    parser.add_argument("--keep_ratios", type=str, default="0.8,0.5,0.3",
                       help="Comma-separated keep_ratio values (for l2_compress)")
    parser.add_argument("--prune_after", type=int, default=100,
                       help="Only compress after this many tokens (for l2_compress)")
    
    # Fix-size L2 arguments
    parser.add_argument("--fix_kv_sizes", type=str, default="256,512",
                       help="Comma-separated fix_kv_size values (for fix_size_l2)")
    parser.add_argument("--strategies", type=str, default="keep_low",
                       help="Comma-separated strategies: keep_low,keep_high,random")
    
    # StreamingLLM arguments
    parser.add_argument("--start_size", type=int, default=4,
                       help="Number of initial tokens (attention sinks)")
    parser.add_argument("--recent_sizes", type=str, default="252,508,1020",
                       help="Comma-separated recent_size values for StreamingLLM")
    
    # H2O-L2 arguments
    parser.add_argument("--heavy_hitter_sizes", type=str, default="32,64,128",
                       help="Comma-separated heavy_hitter_size values for H2O-L2")
    parser.add_argument("--h2o_recent_size", type=int, default=444,
                       help="Recent window size for H2O-L2 (default: 444)")
    
    # SnapKV-Lite arguments
    parser.add_argument("--observation_windows", type=str, default="16,32,64",
                       help="Comma-separated observation_window values for SnapKV-Lite")
    parser.add_argument("--snapkv_keep_sizes", type=str, default="512",
                       help="Comma-separated keep_size values for SnapKV-Lite")
    
    # Pyramid KV arguments
    parser.add_argument("--base_sizes", type=str, default="256,512",
                       help="Comma-separated base_size values for Pyramid KV")
    parser.add_argument("--layer_decay", type=float, default=0.9,
                       help="Layer decay factor for Pyramid KV (default: 0.9)")
    parser.add_argument("--min_size", type=int, default=64,
                       help="Minimum cache size for any layer in Pyramid KV (default: 64)")
    parser.add_argument("--pyramid_profile", type=str, default="exponential",
                       choices=["exponential", "linear", "constant"],
                       help="Pyramid profile: exponential, linear, or constant")
    
    # Adaptive L2 arguments
    parser.add_argument("--target_sizes", type=str, default="256,512",
                       help="Comma-separated target_size values for Adaptive L2")
    parser.add_argument("--soft_limit", type=int, default=256,
                       help="Soft limit (no compression below this) for Adaptive L2")
    parser.add_argument("--hard_limit", type=int, default=1024,
                       help="Hard limit (max compression above this) for Adaptive L2")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.method and not args.compare_all and not args.compare_new:
        parser.error("Must specify --method, --compare_all, or --compare_new")
    
    # Parse skip_layers
    skip_layers = [int(x) for x in args.skip_layers.split(",")]
    
    print("="*70)
    print("KV Cache Compression Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Method: {args.method or ('compare_all' if args.compare_all else 'compare_new')}")
    print(f"  Skip layers: {skip_layers}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Max eval tokens: {args.max_tokens}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Warmup iterations: {args.num_warmup}")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_id)
    
    # Perform warmup to avoid cold-start effects
    if args.num_warmup > 0:
        warmup_model(model, tokenizer, device, num_warmup=args.num_warmup)
    
    # Load PG-19 samples
    samples = load_pg19_samples(args.num_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return
    
    # Build methods configuration
    methods_config = build_methods_config(args)
    
    print(f"\nMethods to test:")
    for m in methods_config:
        print(f"  - {m['name']}: {m['kwargs']}")
    
    # Run benchmark on each sample
    all_results = []
    
    for i, text in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{len(samples)} ({len(text)} characters)")
        print("="*70)
        
        results = run_benchmark_suite(
            model=model,
            tokenizer=tokenizer,
            text=text,
            methods_config=methods_config,
            max_new_tokens=args.max_new_tokens,
            eval_tokens=args.max_tokens,
            skip_layers=skip_layers,
            device=device,
        )
        
        all_results.extend(results)
    
    # Aggregate results by method
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (averaged across samples)")
    print("="*80)
    
    # Group by method name
    grouped = {}
    for r in all_results:
        method = r.get('method', 'unknown')
        if method not in grouped:
            grouped[method] = []
        grouped[method].append(r)
    
    # Print header
    print(f"\n{'Method':<25} {'TTFT(s)':>10} {'TPOT(s)':>10} "
          f"{'Thruput':>10} {'PPL':>10} {'Acc':>10} {'Cache':>8}")
    print("-"*90)
    
    # Find baseline for comparison
    baseline_ppl = None
    baseline_acc = None
    baseline_throughput = None
    baseline_tpot = None
    
    if 'baseline' in grouped:
        baseline_results = grouped['baseline']
        baseline_ppl = np.mean([r['perplexity'] for r in baseline_results])
        baseline_acc = np.mean([r['accuracy'] for r in baseline_results])
        baseline_throughput = np.mean([r['throughput'] for r in baseline_results])
        baseline_tpot = np.mean([r['tpot'] for r in baseline_results])
    
    # Print results
    for method, results in grouped.items():
        avg_ttft = np.mean([r['ttft'] for r in results])
        avg_tpot = np.mean([r['tpot'] for r in results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        avg_ppl = np.mean([r['perplexity'] for r in results])
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_cache = np.mean([r['final_cache_size'] for r in results])
        
        print(f"{method:<25} {avg_ttft:>10.4f} {avg_tpot:>10.4f} "
              f"{avg_throughput:>10.2f} {avg_ppl:>10.2f} {avg_acc:>10.2%} {avg_cache:>8.0f}")
    
    print("="*90)
    
    # Print comparison with baseline
    if baseline_ppl is not None and len(grouped) > 1:
        print("\nComparison with baseline (Throughput ↑ better, TPOT ↓ better, PPL ↓ better):")
        for method, results in grouped.items():
            if method == 'baseline':
                continue
            
            avg_throughput = np.mean([r['throughput'] for r in results])
            avg_tpot = np.mean([r['tpot'] for r in results])
            avg_ppl = np.mean([r['perplexity'] for r in results])
            avg_acc = np.mean([r['accuracy'] for r in results])
            
            # Throughput improvement (higher is better)
            throughput_imp = (avg_throughput / baseline_throughput - 1) * 100 if baseline_throughput > 0 else 0
            # TPOT improvement (lower is better)
            tpot_imp = (1 - avg_tpot / baseline_tpot) * 100 if baseline_tpot > 0 else 0
            ppl_change = (avg_ppl / baseline_ppl - 1) * 100 if baseline_ppl > 0 else 0
            acc_change = (avg_acc / baseline_acc - 1) * 100 if baseline_acc > 0 else 0
            
            print(f"  {method}: Throughput {throughput_imp:+.1f}%, TPOT {tpot_imp:+.1f}%, PPL {ppl_change:+.1f}%, Acc {acc_change:+.1f}%")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
