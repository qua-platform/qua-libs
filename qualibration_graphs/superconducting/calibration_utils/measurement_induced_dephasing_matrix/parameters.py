"""Parameter definitions for the measurement-induced dephasing matrix experiment."""

from typing import List, Optional

import numpy as np

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Sweep and fit parameters specific to the measurement-induced dephasing matrix."""

    num_shots: int = 1000
    """Number of averages per (qubit, driven resonator, xi, phase) point. Default is 1000."""
    num_xi_points: int = 15
    """Number of non-zero readout amplitude scaling points. A xi = 0 reference point is always
    prepended, so each sweep contains num_xi_points + 1 values. Default is 15."""
    num_phase_points: int = 12
    """Number of points used to sweep the phase of the final pi/2 pulse over one full turn.
    Default is 12."""
    xi_max_off_diagonal: float = 1.0
    """Maximum readout amplitude scaling for crosstalk (i != j) pairs, normalised to the calibrated
    single-shot readout amplitude. Swept linearly from 0. Default is 1.0."""
    xi_min_diagonal: float = 1e-2
    """Minimum non-zero readout amplitude scaling for self-dephasing (i == j) pairs. The diagonal is
    swept logarithmically because Gamma_ii is orders of magnitude larger. Default is 1e-2."""
    xi_max_diagonal: float = 0.3
    """Maximum readout amplitude scaling for self-dephasing (i == j) pairs. The decay is only well
    conditioned when the exponent Gamma_ii * tau_p * xi_max**2 reaches order unity; for a typical
    Gamma_ii of tens of MHz and a microsecond readout pulse that means xi_max of a few tenths, not a
    few hundredths. Default is 0.3."""
    idle_time_in_ns: Optional[int] = None
    """Fixed half-echo idle time tau, identical for every qubit and every driven resonator. If None
    it is derived as max_j(readout_length_j + depletion_time_j), rounded up to a multiple of 4 ns,
    so that the probe pulse and the subsequent resonator ring-down fit exactly inside the first half
    of the echo. Default is None."""
    min_contrast_snr: float = 2.0
    """Contrast points whose fitted amplitude is below min_contrast_snr times its own uncertainty are
    considered fully dephased (noise floor) and excluded from the Gamma fit. Default is 2.0."""
    max_crosstalk_dephasing_in_hz: float = 1e3
    """A qubit is marked as successful if all of its off-diagonal dephasing rates stay below this
    value. Default is 1e3 Hz."""
    plot_phase_oscillations: bool = False
    """Whether to produce the (large) diagnostic figure showing every phase oscillation together with
    its sinusoidal fit. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for the measurement-induced dephasing matrix node."""

    use_state_discrimination: bool = True
    """Discriminated state readout is the default here: the contrast is read off a P(e) oscillation."""


def build_phases(parameters: NodeSpecificParameters) -> np.ndarray:
    """Return the phases of the final pi/2 pulse, in units of 2*pi (i.e. turns).

    One full turn is covered with ``num_phase_points`` equally spaced points, excluding the
    end point so that no phase is sampled twice.
    """
    return np.linspace(0, 1, parameters.num_phase_points, endpoint=False)


def build_xi_values(qubit_names: List[str], parameters: NodeSpecificParameters) -> np.ndarray:
    """Build the per-pair readout amplitude scaling arrays.

    The diagonal (self-dephasing) and the off-diagonal (crosstalk) elements differ by several orders
    of magnitude in dephasing rate, so they cannot share a single amplitude axis. Both arrays have
    the same length, which keeps the acquired dataset rectangular; the actual values are stored as a
    two-dimensional ``xi`` coordinate instead of a dimension.

    Parameters
    ----------
    qubit_names : list of str
        Names of the measured qubits, in acquisition order. The driven resonators are the resonators
        of those same qubits, so the returned array is square in its first two axes.
    parameters : NodeSpecificParameters
        Node parameters defining the amplitude ranges.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_qubits, num_qubits, num_xi_points + 1)`` indexed as
        ``[measured qubit, driven resonator, xi index]``. Index 0 along the last axis is always 0.
    """
    num = parameters.num_xi_points
    off_diagonal = np.concatenate(
        [[0.0], np.linspace(parameters.xi_max_off_diagonal / num, parameters.xi_max_off_diagonal, num)]
    )
    diagonal = np.concatenate(
        [[0.0], np.geomspace(parameters.xi_min_diagonal, parameters.xi_max_diagonal, num)]
    )
    num_qubits = len(qubit_names)
    xi_values = np.empty((num_qubits, num_qubits, num + 1))
    for i in range(num_qubits):
        for j in range(num_qubits):
            xi_values[i, j] = diagonal if i == j else off_diagonal
    return xi_values
