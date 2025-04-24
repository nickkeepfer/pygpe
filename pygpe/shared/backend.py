"""
Backend selection module for PyGPE.

This module provides a centralized way to control whether to use GPU (cupy) or CPU (numpy)
for all computations in the PyGPE package.
"""

import os
import importlib.util
import numpy as np
from typing import Any, Union, Optional

# Check if cupy is available
_CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None

# Environment variable to control backend choice
# Set to 'numpy' to force CPU usage even if cupy is available
_PYGPE_BACKEND = os.environ.get("PYGPE_BACKEND", "auto").lower()

# Initialize global variables
_using_gpu = False
_xp = None  # Will hold the array module (numpy or cupy)
_cupy = None  # Will hold the cupy module if available

def use_gpu(use_gpu=None):
    """
    Get or set whether to use GPU (cupy) for computations.
    
    Args:
        use_gpu (bool, optional): If provided, sets whether to use GPU.
            If None, returns the current setting without changing it.
    
    Returns:
        bool: True if using GPU (cupy), False if using CPU (numpy)
    
    Raises:
        ImportError: If attempting to use GPU but cupy is not installed
    """
    global _using_gpu, _xp, _cupy
    
    if use_gpu is not None:
        if use_gpu and not _CUPY_AVAILABLE:
            raise ImportError(
                "Cannot use GPU: cupy is not installed. "
                "Please install cupy with 'pip install cupy-cuda11x' "
                "(replace with appropriate CUDA version)"
            )
        
        _using_gpu = use_gpu
        
        # Import the appropriate module
        if _using_gpu:
            if _cupy is None:
                import cupy
                _cupy = cupy
            _xp = _cupy
        else:
            _xp = np
    
    return _using_gpu

def get_array_module():
    """
    Returns the current array module (either numpy or cupy).
    
    Returns:
        module: The array module (numpy or cupy)
    """
    global _xp
    if _xp is None:
        # Initialize based on environment variable and availability
        if _PYGPE_BACKEND == "numpy":
            use_gpu(False)
        elif _PYGPE_BACKEND == "cupy":
            if not _CUPY_AVAILABLE:
                raise ImportError(
                    "PYGPE_BACKEND set to 'cupy' but cupy is not installed. "
                    "Please install cupy or set PYGPE_BACKEND to 'numpy' or 'auto'."
                )
            use_gpu(True)
        else:  # 'auto' - use cupy if available
            use_gpu(_CUPY_AVAILABLE)
    
    return _xp

def to_numpy(array: Any) -> np.ndarray:
    """
    Convert a cupy array to numpy if needed.
    
    Args:
        array: A numpy or cupy array
        
    Returns:
        numpy.ndarray: The array converted to numpy if it was a cupy array
    """
    if _using_gpu and _cupy is not None and isinstance(array, _cupy.ndarray):
        return _cupy.asnumpy(array)
    return array

def ensure_array_type(array: Any) -> Any:
    """
    Ensure an array is of the currently selected backend type.
    
    This function is crucial for maintaining consistent array types across
    the codebase. It converts arrays to the correct type based on the
    current backend selection.
    
    Args:
        array: Any array-like object (numpy array, cupy array, list, etc.)
        
    Returns:
        array: The array converted to the current backend type
    """
    # If array is None or not array-like, return as is
    if array is None:
        return None
    
    xp = get_array_module()
    
    # Handle numpy arrays
    if isinstance(array, np.ndarray):
        if _using_gpu and _cupy is not None:
            return _cupy.array(array)
        return array
    
    # Handle cupy arrays
    if _CUPY_AVAILABLE and _cupy is not None and isinstance(array, _cupy.ndarray):
        if not _using_gpu:
            return _cupy.asnumpy(array)
        return array
    
    # Handle other array-like objects (lists, tuples, etc.)
    return xp.array(array)

def asarray(array: Any) -> Any:
    """
    Convert input to an array of the current backend type.
    
    This function is a drop-in replacement for numpy.asarray or cupy.asarray,
    but automatically uses the correct backend.
    
    Args:
        array: Any array-like object
        
    Returns:
        array: The input converted to the current backend's array type
    """
    return get_array_module().asarray(array)

# Initialize the backend on module import
_ = get_array_module() 