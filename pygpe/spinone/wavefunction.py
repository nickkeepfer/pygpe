from pygpe.shared.grid import Grid
from pygpe.shared.wavefunction import _Wavefunction
from pygpe.shared.backend import get_array_module, ensure_array_type

# Get the array module (numpy or cupy)
xp = get_array_module()

class SpinOneWavefunction(_Wavefunction):
    """Represents the spin-1 BEC wavefunction.
    This class contains the wavefunction arrays, in addition to various useful
    functions for manipulating and using the wavefunction.

    :param grid: The numerical grid.
    :type grid: :class:`Grid`

    :ivar plus_component: The +1 spin component of the real-space wavefunction
        array.
    :ivar zero_component: The 0 spin component of the real-space wavefunction
        array.
    :ivar minus_component: The -1 spin component of the real-space wavefunction
        array.
    :ivar fourier_plus_component: The +1 spin component of the Fourier-space
        wavefunction array.
    :ivar fourier_zero_component: The 0 spin component of the Fourier-space
        wavefunction array.
    :ivar fourier_minus_component: The -1 spin component of the Fourier-space
        wavefunction array.
    :ivar atom_num_plus: The atom number in the +1 spin component.
    :ivar atom_num_zero: The atom number in the 0 spin component.
    :ivar atom_num_minus: The atom number in the -1 spin component.
    :ivar grid: Reference to the grid object of the simulation.
    """

    def __init__(self, grid: Grid) -> None:
        """Constructs the wavefunction object.

        :param grid: The numerical grid.
        :type grid: :class:`Grid`
        """
        super().__init__(grid)

        self.plus_component = xp.zeros(grid.shape, dtype="complex128")
        self.zero_component = xp.zeros(grid.shape, dtype="complex128")
        self.minus_component = xp.zeros(grid.shape, dtype="complex128")
        self.fourier_plus_component = xp.zeros(grid.shape, dtype="complex128")
        self.fourier_zero_component = xp.zeros(grid.shape, dtype="complex128")
        self.fourier_minus_component = xp.zeros(grid.shape, dtype="complex128")

        self.atom_num_plus = 0
        self.atom_num_zero = 0
        self.atom_num_minus = 0

    def set_ground_state(self, ground_state: str, params: dict) -> None:
        """Sets the components of the wavefunction according to
        the ground state we wish to be in.

        :param ground_state: "polar", "ferromagnetic", or "antiferromagnetic".
            The ground state of the wavefunction.
        :param params: Dictionary containing condensate parameters.
        """
        ground_states = {
            "polar": _polar_initial_state,
            "ferromagnetic": _ferromagnetic_initial_state,
            "antiferromagnetic": _antiferromagnetic_initial_state,
            "BA": _broken_axisymmetry_initial_state,
        }

        ground_states[ground_state](self, params)

        self._update_atom_numbers()

    def set_wavefunction(
        self,
        plus_component: xp.ndarray = None,
        zero_component: xp.ndarray = None,
        minus_component: xp.ndarray = None,
    ) -> None:
        """Sets the wavefunction components to the specified arrays.

        :param plus_component: Plus component of the wavefunction.
        :param zero_component: Zero component of the wavefunction.
        :param minus_component: Minus component of the wavefunction.
        """
        if plus_component is not None:
            self.plus_component = plus_component
        if zero_component is not None:
            self.zero_component = zero_component
        if minus_component is not None:
            self.minus_component = minus_component

        self._update_atom_numbers()

    def add_noise(
        self, components: str | list[str], mean: float, std_dev: float
    ) -> None:
        """Adds noise to the specified wavefunction components
        using a normal distribution.

        :param components: "all", "outer", "plus", "zero", "minus", or list of
            strings specifying the required components to add noise to.
        :param mean: The mean of the normal distribution.
        :param std_dev: The standard deviation of the normal distribution.
        """
        match components:
            case [*_]:
                for component in components:
                    self._add_noise_to_components(component, mean, std_dev)
            case "outer":
                for component in ["plus", "minus"]:
                    self._add_noise_to_components(component, mean, std_dev)
            case "all":
                for component in ["plus", "zero", "minus"]:
                    self._add_noise_to_components(component, mean, std_dev)
            case str(component):
                self._add_noise_to_components(component, mean, std_dev)
            case _:
                raise ValueError(f"{components} is not a supported configuration")

        self._update_atom_numbers()

    def _add_noise_to_components(
        self, component: str, mean: float, std_dev: float
    ) -> None:
        """Adds noise from drawn from a normal distribution to the specified
        component.
        """
        match component.lower():
            case "plus":
                self.plus_component += super()._generate_complex_normal_dist(
                    mean, std_dev
                )
            case "zero":
                self.zero_component += super()._generate_complex_normal_dist(
                    mean, std_dev
                )
            case "minus":
                self.minus_component += super()._generate_complex_normal_dist(
                    mean, std_dev
                )
            case _:
                raise ValueError(f"{component} is not a supported configuration")

    def apply_phase(
        self, phase: xp.ndarray, components: str | list[str] = "all"
    ) -> None:
        """Applies a phase to specified components.

        :param phase: The phase to be applied.
        :param components: "all", "plus", "zero", "minus" or a list of strings
            specifying the required components.
        """
        match components:
            case [*_]:
                for component in components:
                    self._apply_phase_to_component(phase, component)
            case "all":
                for component in ["plus", "zero", "minus"]:
                    self._apply_phase_to_component(phase, component)
            case str(component):
                self._apply_phase_to_component(phase, component)
            case _:
                raise ValueError(f"Components type {components} is unsupported")

    def _apply_phase_to_component(self, phase: xp.ndarray, component: str) -> None:
        """Applies the specified phase to the specified component."""
        match component.lower():
            case "plus":
                self.plus_component *= xp.exp(1j * phase)
            case "zero":
                self.zero_component *= xp.exp(1j * phase)
            case "minus":
                self.minus_component *= xp.exp(1j * phase)
            case _:
                raise ValueError(f"Component type {component} is unsupported")

    def _update_atom_numbers(self) -> None:
        self.atom_num_plus = self.grid.grid_spacing_product * xp.sum(
            xp.abs(self.plus_component) ** 2
        )
        self.atom_num_zero = self.grid.grid_spacing_product * xp.sum(
            xp.abs(self.zero_component) ** 2
        )
        self.atom_num_minus = self.grid.grid_spacing_product * xp.sum(
            xp.abs(self.minus_component) ** 2
        )

    def fft(self) -> None:
        """Fourier transforms real-space components and updates Fourier-space
        components.
        """
        self.fourier_plus_component = xp.fft.fftn(self.plus_component)
        self.fourier_zero_component = xp.fft.fftn(self.zero_component)
        self.fourier_minus_component = xp.fft.fftn(self.minus_component)

    def ifft(self) -> None:
        """Inverse Fourier transforms Fourier-space components and updates
        real-space components.
        """
        self.plus_component = xp.fft.ifftn(self.fourier_plus_component)
        self.zero_component = xp.fft.ifftn(self.fourier_zero_component)
        self.minus_component = xp.fft.ifftn(self.fourier_minus_component)

    def density(self) -> xp.ndarray:
        """Returns an array of the total condensate density.

        :return: Total condensate density.
        """
        return (
            xp.abs(self.plus_component) ** 2
            + xp.abs(self.zero_component) ** 2
            + xp.abs(self.minus_component) ** 2
        )


def _polar_initial_state(wfn: SpinOneWavefunction, params: dict) -> None:
    """Sets wavefunction components to (easy-axis) polar state."""
    wfn.plus_component = xp.zeros(wfn.grid.shape, dtype="complex128")
    wfn.zero_component = xp.sqrt(params["n0"]) * xp.ones(
        wfn.grid.shape, dtype="complex128"
    )
    wfn.minus_component = xp.zeros(wfn.grid.shape, dtype="complex128")


def _ferromagnetic_initial_state(wfn: SpinOneWavefunction, params: dict) -> None:
    """Sets wavefunction components to ferromagnetic state."""
    wfn.plus_component = xp.sqrt(params["n0"]) * xp.ones(
        wfn.grid.shape, dtype="complex128"
    )
    wfn.zero_component = xp.zeros(wfn.grid.shape, dtype="complex128")
    wfn.minus_component = xp.zeros(wfn.grid.shape, dtype="complex128")


def _antiferromagnetic_initial_state(wfn: SpinOneWavefunction, params: dict) -> None:
    """Sets wavefunction components to antiferromagnetic state."""
    p = params["p"]  # Linear Zeeman
    c2 = params["c2"]  # Spin-dependent interaction strength
    n = params["n0"]

    wfn.plus_component = (
        xp.sqrt(n)
        * xp.sqrt((1 + p / c2) / 2)
        * xp.ones(wfn.grid.shape, dtype="complex128")
    )
    wfn.zero_component = xp.zeros(wfn.grid.shape, dtype="complex128")
    wfn.minus_component = (
        xp.sqrt(n)
        * xp.sqrt((1 - p / c2) / 2)
        * xp.ones(wfn.grid.shape, dtype="complex128")
    )


def _broken_axisymmetry_initial_state(wfn: SpinOneWavefunction, params: dict) -> None:
    """Sets wavefunction components to antiferromagnetic state."""
    p = params["p"]  # Linear Zeeman
    q = params["q"]  # Quadratic Zeeman
    c2 = params["c2"]  # Spin-dependent interaction strength
    n = params["n0"]

    wfn.plus_component = (
        xp.sqrt(n)
        * (q + p)
        / (2 * q)
        * xp.sqrt((-(p**2) + q**2 + 2 * c2 * n * q) / (2 * c2 * n * q))
        * xp.ones(wfn.grid.shape, dtype="complex128")
    )
    wfn.zero_component = (
        xp.sqrt(n)
        * xp.sqrt(
            (q**2 - p**2) * (-(p**2) - q**2 + 2 * c2 * n * q) / (4 * c2 * n * q**3)
        )
        * xp.ones(wfn.grid.shape, dtype="complex128")
    )
    wfn.minus_component = (
        xp.sqrt(n)
        * (q - p)
        / (2 * q)
        * xp.sqrt((-(p**2) + q**2 + 2 * c2 * n * q) / (2 * c2 * n * q))
        * xp.ones(wfn.grid.shape, dtype="complex128")
    )
