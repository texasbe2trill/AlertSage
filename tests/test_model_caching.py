"""
Tests for model caching functionality (Issue #17).

This test suite verifies:
1. Model caching prevents reloading from disk
2. Cache returns identical objects on repeated calls
3. No memory leaks with multiple invocations
4. Cache works correctly in CLI context
"""

import sys
import importlib
import pytest


def test_cache_prevents_reloading(tmp_path, monkeypatch):
    """Test that models are only loaded from disk once."""
    # Import model module fresh
    if "src.triage.model" in sys.modules:
        importlib.reload(sys.modules["src.triage.model"])
    
    from src.triage.model import load_vectorizer_and_model
    import src.triage.model as model_module
    
    # Reset cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Track number of joblib.load calls
    original_joblib_load = None
    load_count = {"count": 0}
    
    import joblib
    original_joblib_load = joblib.load
    
    def counting_load(*args, **kwargs):
        load_count["count"] += 1
        return original_joblib_load(*args, **kwargs)
    
    monkeypatch.setattr("joblib.load", counting_load)
    
    # First call should load from disk (2 files: vectorizer + model)
    vec1, model1 = load_vectorizer_and_model()
    first_load_count = load_count["count"]
    
    # Subsequent calls should NOT load from disk
    vec2, model2 = load_vectorizer_and_model()
    vec3, model3 = load_vectorizer_and_model()
    
    # Verify only initial load happened
    assert load_count["count"] == first_load_count, \
        f"Models reloaded from disk! Expected {first_load_count} loads, got {load_count['count']}"
    
    # Verify same objects returned
    assert vec1 is vec2 is vec3, "Vectorizer objects are not identical"
    assert model1 is model2 is model3, "Model objects are not identical"


def test_cache_returns_identical_objects():
    """Test that cache returns same object references, not copies."""
    from src.triage.model import load_vectorizer_and_model
    import src.triage.model as model_module
    
    # Reset cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Load multiple times
    vec1, model1 = load_vectorizer_and_model()
    vec2, model2 = load_vectorizer_and_model()
    vec3, model3 = load_vectorizer_and_model()
    
    # Check identity (same object in memory, not just equal)
    assert vec1 is vec2, "Vectorizers are not the same object"
    assert vec2 is vec3, "Vectorizers are not the same object"
    assert model1 is model2, "Models are not the same object"
    assert model2 is model3, "Models are not the same object"
    
    # Verify they're not just equal but actually identical
    assert id(vec1) == id(vec2) == id(vec3), "Vectorizer IDs differ"
    assert id(model1) == id(model2) == id(model3), "Model IDs differ"


def test_cache_consistency_across_predictions():
    """Test that cache works correctly for multiple load operations."""
    from src.triage.model import load_vectorizer_and_model
    import src.triage.model as model_module
    
    # Reset cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Load multiple times and verify cache persists
    results = []
    for i in range(5):
        vec, model = load_vectorizer_and_model()
        results.append((id(vec), id(model)))
        
        # Verify cache is populated after first call
        assert model_module._VECTORIZER is not None, "Cache not populated"
        assert model_module._MODEL is not None, "Cache not populated"
    
    # All loads should return same object IDs (cache working)
    vec_ids = [r[0] for r in results]
    model_ids = [r[1] for r in results]
    
    assert len(set(vec_ids)) == 1, "Vectorizer objects differ - cache not working"
    assert len(set(model_ids)) == 1, "Model objects differ - cache not working"
    
    # Verify objects are not None and have expected attributes
    vec, model = load_vectorizer_and_model()
    assert hasattr(vec, 'transform'), "Vectorizer missing transform method"
    assert hasattr(model, 'predict'), "Model missing predict method"


def test_cli_load_artifacts_uses_cache():
    """Test that CLI's load_artifacts() function uses the cache."""
    from src.triage.cli import load_artifacts
    import src.triage.model as model_module
    
    # Reset cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Load artifacts twice
    vec1, clf1, emb1, classes1 = load_artifacts()
    vec2, clf2, emb2, classes2 = load_artifacts()
    
    # Verify same model objects returned (cache working)
    assert vec1 is vec2, "CLI not using cached vectorizer"
    assert clf1 is clf2, "CLI not using cached model"
    
    # Verify results are valid
    assert vec1 is not None
    assert clf1 is not None
    assert len(classes1) > 0
    assert len(classes2) > 0


def test_cache_memory_stability():
    """Test that repeated calls don't cause memory issues."""
    from src.triage.model import load_vectorizer_and_model
    import src.triage.model as model_module
    
    # Reset cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Load many times
    references = []
    for _ in range(100):
        vec, model = load_vectorizer_and_model()
        references.append((id(vec), id(model)))
    
    # All calls should return same object IDs
    unique_vec_ids = set(ref[0] for ref in references)
    unique_model_ids = set(ref[1] for ref in references)
    
    assert len(unique_vec_ids) == 1, f"Multiple vectorizer objects created: {len(unique_vec_ids)}"
    assert len(unique_model_ids) == 1, f"Multiple model objects created: {len(unique_model_ids)}"


def test_cache_invalidation_on_reset():
    """Test that cache can be manually invalidated."""
    from src.triage.model import load_vectorizer_and_model
    import src.triage.model as model_module
    
    # Reset cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Load first time
    vec1, model1 = load_vectorizer_and_model()
    id1 = (id(vec1), id(model1))
    
    # Manually invalidate cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Load again - should create new objects
    vec2, model2 = load_vectorizer_and_model()
    id2 = (id(vec2), id(model2))
    
    # IDs should differ (new objects loaded)
    # Note: This tests the invalidation mechanism, not that it happens automatically
    assert id1 != id2, "Cache invalidation failed - same objects returned"


def test_concurrent_predictions_use_cache():
    """Test that cache provides performance benefit for repeated loads."""
    from src.triage.model import load_vectorizer_and_model
    import src.triage.model as model_module
    import time
    
    # Reset cache
    model_module._VECTORIZER = None
    model_module._MODEL = None
    
    # Time first load (cold - from disk)
    start = time.perf_counter()
    vec1, model1 = load_vectorizer_and_model()
    first_time = time.perf_counter() - start
    
    # Time subsequent loads (hot - from cache)
    cached_times = []
    for i in range(10):
        start = time.perf_counter()
        vec, model = load_vectorizer_and_model()
        elapsed = time.perf_counter() - start
        cached_times.append(elapsed)
    
    avg_cached_time = sum(cached_times) / len(cached_times)
    
    # Cached loads should be MUCH faster than first load
    # We expect at least 1000x improvement based on benchmark results
    speedup = first_time / avg_cached_time if avg_cached_time > 0 else 0
    
    print(f"\n  First load: {first_time*1000:.2f} ms")
    print(f"  Avg cached load: {avg_cached_time*1000:.4f} ms")
    print(f"  Speedup: {speedup:.0f}x")
    
    # Very conservative check - cache should be at least 100x faster
    assert speedup > 100, \
        f"Cache not providing significant speedup: {speedup:.1f}x (expected >100x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
