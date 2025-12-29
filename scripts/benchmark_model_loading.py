#!/usr/bin/env python3
"""
Benchmark script to measure model loading performance.

This script measures:
1. Initial load time (cold start)
2. Repeated load time (to test caching)
3. Memory usage across multiple invocations
4. Batch mode performance (multiple predictions in one process)

Run this before and after implementing caching optimization to measure improvement.

Usage:
    python scripts/benchmark_model_loading.py
"""

import time
import sys
from pathlib import Path
import tracemalloc
import statistics

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def measure_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function call."""
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, current / 1024 / 1024, peak / 1024 / 1024  # Convert to MB


def benchmark_direct_loading():
    """Benchmark the CURRENT approach (direct joblib.load in cli.py)."""
    import joblib
    
    models_dir = PROJECT_ROOT / "models"
    vectorizer_path = models_dir / "vectorizer.joblib"
    model_path = models_dir / "enhanced_logreg.joblib"
    
    if not vectorizer_path.exists() or not model_path.exists():
        print("❌ Model files not found. Please ensure models are trained.")
        return None
    
    print("\n" + "="*60)
    print("🔍 BASELINE: Direct joblib.load() - CURRENT APPROACH")
    print("="*60)
    
    # Single invocation timing
    times = []
    for i in range(5):
        start = time.perf_counter()
        _ = joblib.load(vectorizer_path)
        _ = joblib.load(model_path)
        end = time.perf_counter()
        load_time = (end - start) * 1000  # Convert to milliseconds
        times.append(load_time)
        print(f"  Load #{i+1}: {load_time:.2f} ms")
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"\n📊 Statistics:")
    print(f"  Average load time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  Min: {min(times):.2f} ms")
    print(f"  Max: {max(times):.2f} ms")
    
    # Memory measurement
    _, current_mem, peak_mem = measure_memory_usage(
        lambda: (joblib.load(vectorizer_path), joblib.load(model_path))
    )
    print(f"\n💾 Memory Usage:")
    print(f"  Current: {current_mem:.2f} MB")
    print(f"  Peak: {peak_mem:.2f} MB")
    
    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min(times),
        "max_time": max(times),
        "current_mem": current_mem,
        "peak_mem": peak_mem,
    }


def benchmark_cached_loading():
    """Benchmark the OPTIMIZED approach (using cached load_vectorizer_and_model)."""
    # Reset the cache first to ensure clean test
    import src.triage.model as model_module
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    from src.triage.model import load_vectorizer_and_model
    
    print("\n" + "="*60)
    print("⚡ OPTIMIZED: Cached load_vectorizer_and_model() - NEW APPROACH")
    print("="*60)
    
    # Test that cache works - first call should load, subsequent calls should be instant
    times = []
    
    # First call (cache miss - should load from disk)
    start = time.perf_counter()
    vectorizer1, clf1 = load_vectorizer_and_model()
    end = time.perf_counter()
    first_load = (end - start) * 1000
    print(f"  Initial load (cache miss): {first_load:.2f} ms")
    
    # Subsequent calls (cache hits - should be nearly instant)
    for i in range(10):
        start = time.perf_counter()
        _, _ = load_vectorizer_and_model()
        end = time.perf_counter()
        cached_time = (end - start) * 1000
        times.append(cached_time)
        if i < 5:  # Print first 5 to show pattern
            print(f"  Cached call #{i+1}: {cached_time:.4f} ms")
    
    avg_cached = statistics.mean(times)
    
    print(f"\n📊 Statistics:")
    print(f"  First load: {first_load:.2f} ms")
    print(f"  Average cached access: {avg_cached:.4f} ms")
    print(f"  Speedup: {first_load/avg_cached:.0f}x faster after caching")
    
    # Verify cache returns same objects
    vectorizer3, clf3 = load_vectorizer_and_model()
    same_objects = (vectorizer1 is vectorizer3) and (clf1 is clf3)
    print(f"\n✅ Cache verification: Objects are {'identical' if same_objects else 'DIFFERENT'} (should be identical)")
    
    # Memory measurement for cached calls
    _, current_mem, peak_mem = measure_memory_usage(load_vectorizer_and_model)
    print(f"\n💾 Memory Usage (cached call):")
    print(f"  Current: {current_mem:.2f} MB")
    print(f"  Peak: {peak_mem:.2f} MB")
    
    return {
        "first_load": first_load,
        "avg_cached": avg_cached,
        "speedup": first_load/avg_cached if avg_cached > 0 else 0,
        "cache_works": same_objects,
        "current_mem": current_mem,
        "peak_mem": peak_mem,
    }


def benchmark_batch_predictions():
    """Simulate batch mode: multiple predictions in a single process."""
    from src.triage.model import predict_event_type
    
    # Reset cache to ensure clean state
    import src.triage.model as model_module
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    print("\n" + "="*60)
    print("🔄 BATCH MODE: Multiple predictions in one process")
    print("="*60)
    
    test_incidents = [
        "Suspicious email with attachment received by user",
        "Multiple failed login attempts detected from external IP",
        "Unusual outbound network traffic to unknown domain",
        "User downloaded file from suspicious website",
        "Privileged account accessed from unusual location",
    ]
    
    # Measure total time for batch predictions using the high-level API
    start = time.perf_counter()
    predictions = []
    for incident in test_incidents:
        try:
            label, proba = predict_event_type(incident)
            predictions.append((label, proba))
        except Exception as e:
            print(f"  Warning: Prediction failed: {e}")
            predictions.append(("error", {}))
    end = time.perf_counter()
    
    total_time = (end - start) * 1000
    per_prediction = total_time / len(test_incidents) if predictions else 0
    
    print(f"  Total time for {len(test_incidents)} predictions: {total_time:.2f} ms")
    print(f"  Average per prediction: {per_prediction:.2f} ms")
    
    if predictions and predictions[0][0] != "error":
        print(f"\n  Sample predictions:")
        for i, (incident, (label, proba)) in enumerate(zip(test_incidents[:3], predictions[:3]), 1):
            if proba:
                max_prob = max(proba.values())
                print(f"    {i}. '{incident[:45]}...' → {label} ({max_prob:.2%})")
    
    return {
        "total_time": total_time,
        "per_prediction": per_prediction,
        "num_predictions": len(test_incidents),
    }


def test_memory_leak():
    """Verify no memory leaks with repeated invocations."""
    from src.triage.model import load_vectorizer_and_model
    
    print("\n" + "="*60)
    print("🔬 MEMORY LEAK TEST: 50 repeated invocations")
    print("="*60)
    
    tracemalloc.start()
    
    memory_snapshots = []
    for i in range(50):
        load_vectorizer_and_model()
        if i % 10 == 0:
            current, _ = tracemalloc.get_traced_memory()
            memory_snapshots.append(current / 1024 / 1024)
            print(f"  After {i+1} calls: {current / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    
    # Check if memory is stable (not growing)
    memory_growth = memory_snapshots[-1] - memory_snapshots[0]
    is_stable = abs(memory_growth) < 5  # Less than 5MB growth is acceptable
    
    print(f"\n  Memory change: {memory_growth:+.2f} MB")
    print(f"  {'✅ PASS' if is_stable else '❌ FAIL'}: Memory is {'stable' if is_stable else 'GROWING (potential leak)'}")
    
    return {
        "memory_growth": memory_growth,
        "is_stable": is_stable,
    }


def main():
    """Run all benchmarks and generate report."""
    print("\n" + "="*60)
    print("🚀 MODEL LOADING PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Project: AlertSage1")
    print(f"Issue: #17 - Optimize Model Loading Performance")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run benchmarks
        baseline = benchmark_direct_loading()
        if baseline is None:
            print("\n❌ Cannot proceed without model files.")
            return
        
        optimized = benchmark_cached_loading()
        batch = benchmark_batch_predictions()
        leak_test = test_memory_leak()
        
        # Summary
        print("\n" + "="*60)
        print("📈 SUMMARY")
        print("="*60)
        
        if baseline and optimized:
            improvement = ((baseline["avg_time"] - optimized["avg_cached"]) / baseline["avg_time"]) * 100
            print(f"\n⚡ Performance Improvement:")
            print(f"  Baseline (uncached): {baseline['avg_time']:.2f} ms")
            print(f"  Optimized (cached): {optimized['avg_cached']:.4f} ms")
            print(f"  Improvement: {improvement:.1f}% faster for cached calls")
            print(f"  Speedup factor: {optimized['speedup']:.0f}x")
        
        print(f"\n🔄 Batch Mode Benefits:")
        print(f"  {batch['num_predictions']} predictions: {batch['total_time']:.2f} ms")
        print(f"  Average per prediction: {batch['per_prediction']:.2f} ms")
        print(f"  (Models loaded once, reused {batch['num_predictions']} times)")
        
        print(f"\n💾 Memory Status:")
        print(f"  {'✅ PASS' if leak_test['is_stable'] else '❌ FAIL'}: No memory leaks detected")
        print(f"  Memory growth after 50 calls: {leak_test['memory_growth']:+.2f} MB")
        
        print(f"\n✅ Cache Functionality:")
        print(f"  Cache working: {'✅ YES' if optimized['cache_works'] else '❌ NO'}")
        print(f"  Same objects returned: {optimized['cache_works']}")
        
        print("\n" + "="*60)
        print("✅ Benchmark Complete!")
        print("="*60)
        print("\n💡 Next Steps:")
        print("  1. Update cli.py to use cached loader")
        print("  2. Run this benchmark again to verify improvements")
        print("  3. Run test suite to ensure all tests pass")
        print("  4. Document results in docs/performance.md")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
