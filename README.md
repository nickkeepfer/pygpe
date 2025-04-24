<p align="center"><img src="docs/pygpe.png" alt="logo" ></p>

<h4 align="center">A fast and easy to use Gross-Pitaevskii equation solver.</h4>

## Description

PyGPE is a CUDA-accelerated Python library for solving the Gross-Pitaevskii equations for use in simulating
Bose-Einstein condensate systems.

- Documentation: https://wheelermt.github.io/pygpe-docs/

### Supported features

- Scalar, two-component, spin-1, and spin-2 BEC systems.
- 1D, 2D, and 3D grid lattices.
- GPU support.
- HDF5 data saving system.
- Method for generating vortices within the system.

### Requirements

- Python (3.10 and above),
- [h5py](https://github.com/h5py/h5py) (^3.6.0),
- [numpy](https://numpy.org/) (^2.0.0),
- Matplotlib (^3.8.2)

If using a GPU:
  - CUDA Toolkit (>=11.2)
  - [CuPy](https://github.com/cupy/cupy) (>=10.2.0).

## Installation

The simplest way to begin using PyGPE is through pip:

    pip install pygpe

By default, PyGPE will use the CPU to perform calculations.
However, if a CUDA-capable GPU is detected, PyGPE will automatically utilise it for drastic
speed-ups in computation time.

### Controlling GPU/CPU Usage

PyGPE provides several ways to control whether computations run on CPU (using NumPy) or GPU (using CuPy):

1. **Programmatically**:
   ```python
   import pygpe
   
   # Force CPU usage
   pygpe.use_gpu(False)
   
   # Use GPU if available
   pygpe.use_gpu(True)
   
   # Check current setting
   is_using_gpu = pygpe.use_gpu()
   
   # Get the current array module (numpy or cupy)
   xp = pygpe.get_array_module()
   
   # Convert GPU arrays to NumPy for visualization
   numpy_array = pygpe.to_numpy(gpu_array)
   ```

2. **Using environment variables**:
   ```bash
   # Force CPU usage
   export PYGPE_BACKEND=numpy
   
   # Force GPU usage (if available)
   export PYGPE_BACKEND=cupy
   
   # Automatic detection (default)
   export PYGPE_BACKEND=auto
   ```

3. **Handling mixed array types**:
   
   When mixing NumPy and CuPy arrays, use these helper functions to ensure consistent array types:
   ```python
   from pygpe.shared.backend import ensure_array_type, asarray
   
   # Convert any array to the current backend type
   backend_array = ensure_array_type(some_array)
   
   # Like np.asarray but uses the current backend
   backend_array = asarray([1, 2, 3, 4])
   ```

See the [backend_example.py](examples/backend_example.py) for a complete example.

## Examples

See [examples](examples) folder for various examples on the usage of the library.
Below is an animation of superfluid turbulence in a scalar BEC simulated using PyGPE on a $512^2$ lattice
for $N_t=200000$ time steps taking **~5 minutes** to complete on an RTX 2060.

<p align="center"><img src="docs/animation.gif" alt="logo" > </p>
