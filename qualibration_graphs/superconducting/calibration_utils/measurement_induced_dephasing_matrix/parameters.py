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
    readout_len_in_ns: Optional[int] = None
    """Duration tau_p of the probe pulse played on the driven resonator, in ns. If None, each driven
    resonator probes with the native length of its own calibrated 'readout' operation. Otherwise the
    same pulse is stretched to this duration through the QUA 'duration' argument, leaving its
    amplitude untouched, so the resonator holds the same photon number for longer. The uncertainty on
    the fitted dephasing rate falls as 1/tau_p, which makes a longer probe the cheapest way to
    resolve small crosstalk: it costs no extra shots. The gain is not unbounded, because the probe
    has to fit inside the half-echo and the echo contrast decays as exp(-2*idle_time/T2echo); the
    optimum sits near tau_p = T2echo/2, and the idle time check below caps it. Must be a multiple of
    4 ns and at least 16 ns. Default is None."""
    idle_time_in_ns: Optional[int] = None
    """Fixed half-echo idle time tau, identical for every qubit and every driven resonator. If None
    it is derived as max_j(probe_length_j + depletion_time_j), rounded up to a multiple of 4 ns,
    so that the probe pulse and the subsequent resonator ring-down fit exactly inside the first half
    of the echo. Default is None."""
    probe_in_both_halves: bool = False
    """Whether to play the probe pulse in both halves of the echo instead of only the first.

    With the probe in one half only, the mean photon number in the driven resonator pulls the qubit
    frequency and the resulting AC-Stark phase is not refocused, so the oscillation both loses
    contrast (photon shot noise) and shifts in phase (mean photon number). With the probe in both
    halves the x180 pulse inverts the sign of the accumulated Stark phase, so the coherent shift
    cancels while the shot-noise dephasing of the two halves adds up: the oscillation then only
    loses contrast. That makes the contrast decay well conditioned and doubles the probe time the
    decay is sensitive to, at the cost of losing the Stark-shift channel. The Stark-shift matrix is
    therefore only computed and reported when this is False.

    The cancellation and the doubling both assume the photon-noise correlation time 1/kappa is short
    compared with the probe duration, which holds for any probe longer than a few resonator ring-down
    times. Default is False."""
    min_contrast_snr: float = 3.0
    """Contrast points whose fitted amplitude is below min_contrast_snr times its own uncertainty are
    considered fully dephased (noise floor) and excluded from the Gamma and Stark fits.

    Debiasing the contrast removes the upward bias of the noise floor but not its other effect: the
    fits work on ln(c), whose error is only symmetric while c is several times its own uncertainty.
    Points below that turn a clean exponential decay into a tail that bends away from the fitted
    line, which inflates the reduced chi-squared and can get a perfectly good element rejected. Three
    sigma is where ln(c) is Gaussian enough for the chi-squared to mean what it says. Default is
    3.0."""
    max_reduced_chi2: float = 5.0
    """Largest reduced chi-squared a matrix element's straight-line fit may have before the element is
    rejected. A large value means ln(c) is not straight against xi**2 (so the photon number is no
    longer proportional to xi**2, e.g. the resonator has been driven out of its linear range) or that
    single points are outliers. A rejected element keeps its fitted value for inspection but is
    excluded from the pass criterion. Default is 5.0."""
    max_phase_residual_in_rad: float = float(np.pi / 2)
    """Largest distance, in radians, that an AC-Stark phase point may sit from the trend used to
    unwrap it before the Stark fit of that element is rejected.

    The phase advances as xi**2 while xi is swept linearly, so consecutive points can legitimately be
    several radians apart at the top of the sweep and a test on the step size alone would reject the
    strongly shifted pairs the measurement is after. The unwrapping instead extrapolates the trend
    established at low amplitude and places each point on the nearest branch of 2*pi, which is
    unambiguous as long as that prediction is good to well under half a turn. Default is pi/2.
    Ignored when probe_in_both_halves is True."""
    max_crosstalk_dephasing_in_hz: float = 1e3
    """A qubit is marked as successful if every one of its off-diagonal dephasing rates is either too
    small to resolve or resolved below this value. An element is considered resolved when its
    magnitude exceeds twice its own standard error; an unresolved element is an upper bound, not a
    measured crosstalk, and is not judged against this threshold. Default is 1e3 Hz."""
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


def probe_length_in_ns(readout_length_in_ns: int, parameters: NodeSpecificParameters) -> int:
    """Return tau_p, the duration of the probe pulse played on one driven resonator, in ns.

    The QUA program and the analysis must agree on this value: the program uses it to place the
    pulse inside the half-echo, and the fit divides the decay slope by it to get the dephasing rate.
    Deriving both from this one function keeps them from drifting apart.

    Parameters
    ----------
    readout_length_in_ns : int
        Native length of the driven resonator's calibrated 'readout' operation.
    parameters : NodeSpecificParameters
        Node parameters, whose ``readout_len_in_ns`` overrides the native length when set.
    """
    if parameters.readout_len_in_ns is None:
        return readout_length_in_ns
    return parameters.readout_len_in_ns


def num_probe_pulses(parameters: NodeSpecificParameters) -> int:
    """Return how many probe pulses are played per echo, one per half that carries a probe."""
    return 2 if parameters.probe_in_both_halves else 1


def total_probe_length_in_ns(readout_length_in_ns: int, parameters: NodeSpecificParameters) -> int:
    """Return the total time the driven resonator is probed during one echo, in ns.

    This is the duration the fitted decay slope is divided by, because the dephasing accumulated by
    the measured qubit is proportional to the total time photons are present, not to the length of a
    single pulse. With the probe in both halves of the echo it is twice the single-pulse duration.
    """
    return num_probe_pulses(parameters) * probe_length_in_ns(readout_length_in_ns, parameters)


def validate_readout_len(parameters: NodeSpecificParameters) -> None:
    """Raise if the requested probe duration cannot be played.

    QUA takes the duration in clock cycles and needs at least four of them, so the value has to be a
    multiple of 4 ns and no shorter than 16 ns.
    """
    requested = parameters.readout_len_in_ns
    if requested is None:
        return
    if requested < 16 or requested % 4 != 0:
        raise ValueError(
            f"readout_len_in_ns ({requested} ns) must be a multiple of 4 ns and at least 16 ns, "
            f"because the QUA 'duration' argument counts 4 ns clock cycles."
        )


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
    diagonal = np.concatenate([[0.0], np.geomspace(parameters.xi_min_diagonal, parameters.xi_max_diagonal, num)])
    num_qubits = len(qubit_names)
    xi_values = np.empty((num_qubits, num_qubits, num + 1))
    for i in range(num_qubits):
        for j in range(num_qubits):
            xi_values[i, j] = diagonal if i == j else off_diagonal
    return xi_values
