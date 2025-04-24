from pygpe.shared.backend import get_array_module, to_numpy, ensure_array_type

# Get the array module (numpy or cupy)
xp = get_array_module()
import numpy as np


def handle_array(arr):
    """
    Converts a CuPy array to the equivalent NumPy array.
    If the array passed in is already a NumPy array, the array will
    just be returned.

    :param arr: The array to be converted
    :type arr: cupy.ndarray or numpy.ndarray
    """
    # Use the to_numpy function from the backend module
    return to_numpy(arr)
