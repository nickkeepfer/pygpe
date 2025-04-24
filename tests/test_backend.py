import os
import pytest
import numpy as np
import importlib

import pygpe
from pygpe.shared.backend import get_array_module, use_gpu, to_numpy


def test_default_backend():
    """Test that the backend module returns the correct module by default."""
    # By default, we should get numpy if cupy is not available
    xp = get_array_module()
    
    # Check if cupy is available
    cupy_available = importlib.util.find_spec("cupy") is not None
    
    if cupy_available:
        try:
            import cupy
            assert xp.__name__ == "cupy"
        except ImportError:
            assert xp.__name__ == "numpy"
    else:
        assert xp.__name__ == "numpy"


def test_force_cpu():
    """Test that setting use_gpu(False) forces CPU usage."""
    # Force CPU usage
    pygpe.use_gpu(False)
    
    # Get the array module
    xp = get_array_module()
    
    # Check that we're using numpy
    assert xp.__name__ == "numpy"
    
    # Test creating an array
    arr = xp.ones((5, 5))
    assert isinstance(arr, np.ndarray)


def test_to_numpy_conversion():
    """Test that to_numpy correctly converts arrays."""
    # Force CPU usage (to be safe)
    pygpe.use_gpu(False)
    
    # Create a numpy array
    arr = np.ones((5, 5))
    
    # Convert it (should be a no-op since it's already numpy)
    converted = to_numpy(arr)
    
    # Check that it's still a numpy array
    assert isinstance(converted, np.ndarray)
    
    # Check that the data is the same
    np.testing.assert_array_equal(arr, converted)


def test_gpu_detection():
    """Test that use_gpu reports the correct status."""
    # Force CPU usage
    use_gpu(False)
    assert not use_gpu()
    
    # Try to use GPU if available
    try:
        use_gpu(True)
        import cupy
        assert use_gpu()
    except ImportError:
        # GPU not available, should stay as CPU
        assert not use_gpu()


def test_env_variable_override():
    """Test that the PYGPE_BACKEND environment variable works."""
    # Save the original value to restore later
    original = os.environ.get("PYGPE_BACKEND", None)
    
    try:
        # Force CPU via environment variable
        os.environ["PYGPE_BACKEND"] = "numpy"
        
        # Reload the module to apply the environment variable
        import importlib
        import pygpe.shared.backend
        importlib.reload(pygpe.shared.backend)
        
        # Check that we're using numpy
        xp = pygpe.shared.backend.get_array_module()
        assert xp.__name__ == "numpy"
        
    finally:
        # Restore the original value
        if original is None:
            os.environ.pop("PYGPE_BACKEND", None)
        else:
            os.environ["PYGPE_BACKEND"] = original
            
        # Reload the module to restore the original state
        importlib.reload(pygpe.shared.backend) 