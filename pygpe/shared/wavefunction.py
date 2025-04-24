from abc import ABC, abstractmethod

from pygpe.shared.grid import Grid
from pygpe.shared.backend import get_array_module, ensure_array_type

# Get the array module (numpy or cupy)
xp = get_array_module()


class _Wavefunction(ABC):
    """Defines the abstract Wavefunction base class.
    Each system's wavefunction inherits from this class and provides overrides
    for the abstract methods.
    """

    def __init__(self, grid: Grid) -> None:
        """The default constructor for the abstract `Wavefunction` class, to be
        inherited by subclasses of `Wavefunction`.

        :param grid: Grid object of the system.
        :type grid: Grid
        """
        self.grid = grid

    @abstractmethod
    def set_wavefunction(self, wfn: xp.ndarray) -> None:
        """Sets the components of the wavefunction to the specified
        array(s).
        """
        pass

    @abstractmethod
    def add_noise(self, *args, mean: float, std_dev: float) -> None:
        """Adds noise to the specified component(s), drawn from a normal
        distribution.

        :param mean: Mean of the normal distribution.
        :type mean: float
        :param std_dev: Standard deviation of the normal distribution.
        :type std_dev: float
        """
        pass

    def _generate_complex_normal_dist(self, mean: float, std_dev: float) -> xp.ndarray:
        """Returns a `xp.ndarray` of complex values containing results from
        a normal distribution.
        """
        # Get the current array module to ensure consistent array types
        xp = get_array_module()
        
        return xp.random.normal(
            mean, std_dev, size=self.grid.shape
        ) + 1j * xp.random.normal(mean, std_dev, size=self.grid.shape)

    @abstractmethod
    def apply_phase(self, phase: xp.ndarray, **kwargs) -> None:
        """Applies a phase to the specified component(s).

        :param phase: Array of the condensate phase.
        :type phase: xp.ndarray
        """
        pass

    @abstractmethod
    def fft(self) -> None:
        """Computes the forward Fourier transform on all wavefunction
        components.
        """
        pass

    @abstractmethod
    def ifft(self) -> None:
        """Computes the backward Fourier transform on all k-space wavefunction
        components.
        """
        pass

    @abstractmethod
    def density(self) -> xp.ndarray:
        """Computes the total density of the condensate.

        :return: Total density of the condensate.
        :rtype: xp.ndarray
        """
        pass
        
    def _ensure_array_type(self, array):
        """Ensure array is of the correct type for the current backend.
        
        This is a helper method to ensure consistent array types across the codebase.
        """
        return ensure_array_type(array)
