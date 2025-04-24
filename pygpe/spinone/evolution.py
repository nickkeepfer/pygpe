from pygpe.shared.backend import get_array_module
from pygpe.spinone.wavefunction import SpinOneWavefunction

# Get the array module (numpy or cupy)
xp = get_array_module()


def step_wavefunction(wfn: SpinOneWavefunction, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :type wfn: :class:`Wavefunction`
    :param params: The parameters of the system.
    :type params: dict
    """
    _kinetic_zeeman_step(wfn, params)
    wfn.ifft()
    _interaction_step(wfn, params)
    wfn.fft()
    _kinetic_zeeman_step(wfn, params)
    if isinstance(params["dt"], complex):
        _renormalise_wavefunction(wfn)


def _kinetic_zeeman_step(wfn: SpinOneWavefunction, pm: dict) -> None:
    """Computes the kinetic-zeeman subsystem for half a time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameter. dictionary.
    """
    wfn.fourier_plus_component *= xp.exp(
        -0.25 * 1j * pm["dt"] * (wfn.grid.wave_number + 2 * pm["q"])
    )
    wfn.fourier_zero_component *= xp.exp(-0.25 * 1j * pm["dt"] * wfn.grid.wave_number)
    wfn.fourier_minus_component *= xp.exp(
        -0.25 * 1j * pm["dt"] * (wfn.grid.wave_number + 2 * pm["q"])
    )


def _interaction_step(wfn: SpinOneWavefunction, pm: dict) -> None:
    """Computes the interaction subsystem for a full time step.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    spin_perp, spin_z = _calculate_spins(wfn)
    spin_mag = xp.sqrt(abs(spin_perp) ** 2 + spin_z**2)
    dens = _calculate_density(wfn)

    # Trig terms needed in solution
    cos_term = xp.cos(pm["c2"] * spin_mag * pm["dt"])
    sin_term = xp.nan_to_num(1j * xp.sin(pm["c2"] * spin_mag * pm["dt"]) / spin_mag)

    plus_comp_temp = cos_term * wfn.plus_component - sin_term * (
        spin_z * wfn.plus_component
        + xp.conj(spin_perp) / xp.sqrt(2) * wfn.zero_component
    )
    zero_comp_temp = cos_term * wfn.zero_component - sin_term / xp.sqrt(2) * (
        spin_perp * wfn.plus_component + xp.conj(spin_perp) * wfn.minus_component
    )
    minus_comp_temp = cos_term * wfn.minus_component - sin_term * (
        spin_perp / xp.sqrt(2) * wfn.zero_component - spin_z * wfn.minus_component
    )

    wfn.plus_component = plus_comp_temp * xp.exp(
        -1j * pm["dt"] * (pm["trap"] - pm["p"] + pm["c0"] * dens)
    )
    wfn.zero_component = zero_comp_temp * xp.exp(
        -1j * pm["dt"] * (pm["trap"] + pm["c0"] * dens)
    )
    wfn.minus_component = minus_comp_temp * xp.exp(
        -1j * pm["dt"] * (pm["trap"] + pm["p"] + pm["c0"] * dens)
    )


def _calculate_spins(
    wfn: SpinOneWavefunction,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Calculates the perpendicular and longitudinal spins.

    :param wfn: The wavefunction of the system.
    :return: The perpendicular & longitudinal spin, respectively.
    """
    spin_perp = xp.sqrt(2.0) * (
        xp.conj(wfn.plus_component) * wfn.zero_component
        + xp.conj(wfn.zero_component) * wfn.minus_component
    )
    spin_z = xp.abs(wfn.plus_component) ** 2 - xp.abs(wfn.minus_component) ** 2

    return spin_perp, spin_z


def _calculate_density(wfn: SpinOneWavefunction) -> xp.ndarray:
    """Calculates the total condensate density.

    :param wfn: The wavefunction of the system.
    :return: The total atomic density.
    """
    return (
        xp.abs(wfn.plus_component) ** 2
        + xp.abs(wfn.zero_component) ** 2
        + xp.abs(wfn.minus_component) ** 2
    )


def _renormalise_wavefunction(wfn: SpinOneWavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    wfn.ifft()
    correct_atom_num = wfn.atom_num_plus + wfn.atom_num_zero + wfn.atom_num_minus
    current_atom_num = _calculate_atom_num(wfn)
    wfn.plus_component *= xp.sqrt(correct_atom_num / current_atom_num)
    wfn.zero_component *= xp.sqrt(correct_atom_num / current_atom_num)
    wfn.minus_component *= xp.sqrt(correct_atom_num / current_atom_num)
    wfn.fft()


def _calculate_atom_num(wfn: SpinOneWavefunction) -> float:
    """Calculates the total atom number of the system.

    :param wfn: The wavefunction of the system.
    :return: The total atom number.
    """
    atom_num_plus = wfn.grid.grid_spacing_product * xp.sum(
        xp.abs(wfn.plus_component) ** 2
    )
    atom_num_zero = wfn.grid.grid_spacing_product * xp.sum(
        xp.abs(wfn.zero_component) ** 2
    )
    atom_num_minus = wfn.grid.grid_spacing_product * xp.sum(
        xp.abs(wfn.minus_component) ** 2
    )

    return atom_num_plus + atom_num_zero + atom_num_minus
