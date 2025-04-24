"""
This example demonstrates how to control whether PyGPE uses GPU (cupy) or CPU (numpy).
"""

import numpy as np
import pygpe
from pygpe.scalar.wavefunction import ScalarWavefunction
from pygpe.shared.grid import Grid
from pygpe.shared.backend import ensure_array_type, asarray

# Check if GPU is initially being used
print(f"Initial GPU status: {pygpe.use_gpu()}")

# Example 1: Force CPU usage
pygpe.use_gpu(False)
print(f"After setting to CPU: {pygpe.use_gpu()}")

# Create a grid and wavefunction using CPU
grid_cpu = Grid(64, 1.0)
wfn_cpu = ScalarWavefunction(grid_cpu)

# Add some data and check the type
wfn_cpu.set_wavefunction(np.ones((64,), dtype=complex))
print(f"Wavefunction array type: {type(wfn_cpu.component)}")
print(f"Is NumPy array: {isinstance(wfn_cpu.component, np.ndarray)}")

# Example 2: Try to use GPU if available
try:
    pygpe.use_gpu(True)
    print(f"After setting to GPU: {pygpe.use_gpu()}")
    
    # Create a grid and wavefunction using GPU
    grid_gpu = Grid(64, 1.0)
    wfn_gpu = ScalarWavefunction(grid_gpu)
    
    # Get the array module (will be cupy if GPU is enabled)
    xp = pygpe.get_array_module()
    
    # Add some data and check the type
    wfn_gpu.set_wavefunction(xp.ones((64,), dtype=complex))
    print(f"Wavefunction array type: {type(wfn_gpu.component)}")
    
    # Convert GPU array to NumPy for visualization or saving
    cpu_array = pygpe.to_numpy(wfn_gpu.component)
    print(f"Converted array type: {type(cpu_array)}")
    print(f"Is NumPy array: {isinstance(cpu_array, np.ndarray)}")
    
except ImportError as e:
    print(f"GPU not available: {e}")

# Example 3: Use environment variable
# Before running this script, you can set:
# export PYGPE_BACKEND=numpy  # Force CPU
# export PYGPE_BACKEND=cupy   # Force GPU (if available)
# export PYGPE_BACKEND=auto   # Use GPU if available (default)
print(f"\nCurrent backend module: {pygpe.get_array_module().__name__}")

# Example 4: Handling mixed array types
print("\nHandling mixed array types:")
# Create a NumPy array
numpy_array = np.ones((5, 5))
print(f"Original array type: {type(numpy_array)}")

# Ensure it's the correct type for the current backend
backend_array = ensure_array_type(numpy_array)
print(f"After ensure_array_type: {type(backend_array)}")

# Using asarray for convenience
another_array = asarray([1, 2, 3, 4, 5])
print(f"asarray result type: {type(another_array)}") 