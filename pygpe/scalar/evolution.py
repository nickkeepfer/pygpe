from pygpe.shared.backend import get_array_module, ensure_array_type
from pygpe.scalar.wavefunction import ScalarWavefunction

# Get the array module (numpy or cupy)
xp = get_array_module()


def step_wavefunction(wfn: ScalarWavefunction, params: dict) -> None:
    """Propagates the wavefunction forward one time step.

    :param wfn: The wavefunction of the system.
    :type wfn: :class:`Wavefunction`
    :param params: The parameters of the system.
    :type params: dict
    """
    _kinetic_step(wfn, params)
    wfn.ifft()
    _potential_step(wfn, params)
    wfn.fft()
    _kinetic_step(wfn, params)
    if isinstance(params["dt"], complex) or (params.get("gamma", 0) != 0):
        _renormalise_wavefunction(wfn)


def _kinetic_step(wfn: ScalarWavefunction, pm: dict) -> None:
    """Computes the kinetic energy subsystem for half a time step, including dissipation.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    # Get the current array module to ensure consistent array types
    xp = get_array_module()
    
    gamma = pm.get("gamma", 0)  # Dissipation coefficient, default to 0 if unspecified
    wfn.fourier_component *= xp.exp(
        -0.25 * (1 - 1j * gamma) * 1j * pm["dt"] * wfn.grid.wave_number
    )


def _potential_step(wfn: ScalarWavefunction, pm: dict) -> None:
    """Computes the potential subsystem for a full time step, including dissipation.

    :param wfn: The wavefunction of the system.
    :param pm: The parameters' dictionary.
    """
    # Get the current array module to ensure consistent array types
    xp = get_array_module()
    
    gamma = pm.get("gamma", 0)  # Dissipation coefficient, default to 0 if unspecified
    wfn.component *= xp.exp(
        -1j
        * pm["dt"]
        * (1 - 1j * gamma)
        * (pm["trap"] + pm["g"] * xp.abs(wfn.component) ** 2)
    )


def _renormalise_wavefunction(wfn: ScalarWavefunction) -> None:
    """Re-normalises the wavefunction to the correct atom number.

    :param wfn: The wavefunction of the system.
    """
    # Get the current array module to ensure consistent array types
    xp = get_array_module()
    
    wfn.ifft()
    correct_atom_num = wfn.atom_num
    current_atom_num = _calculate_atom_num(wfn)
    wfn.component *= xp.sqrt(correct_atom_num / current_atom_num)
    wfn.fft()


def _calculate_atom_num(wfn: ScalarWavefunction) -> float:
    """Calculates the current atom number of the wavefunction.

    :param wfn: The wavefunction of the system.
    :return: The atom number.
    """
    # Get the current array module to ensure consistent array types
    xp = get_array_module()
    
    return wfn.grid.grid_spacing_product * xp.sum(xp.abs(wfn.component) ** 2)
