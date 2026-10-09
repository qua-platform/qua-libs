from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per (state, zeta point). Default is 1000."""
    zeta_num_points: int = 15
    """Number of zeta points scanned per state. Every point adds one temporary DRACHMA pulse per state and
    qubit to the config, so keep it modest. Default is 10."""
    zeta_step_hz: float = 10e3
    """Spacing (Hz) between consecutive zeta points. Together with zeta_num_points it sets the scan range,
    which is centred on the current zeta (zeta on the grid, the extra point of an even count above it) and
    shifted up if its lower end would go below 0. Default is 10 kHz."""
    max_zeta_difference_hz: float | None = 50e3
    """Hard limit (Hz) on |zeta_ground_hz - zeta_excited_hz| of the pair written to state. The two zetas are
    chosen jointly: the scan point pair inside the limit with the lowest noise-weighted excess residual power
    above each state's own minimum (a state that barely reacts to zeta follows the other one within the limit).
    Must be >= 0. None disables the limit (each state takes its own minimum). Default is 50 kHz."""
    max_waveform_peak: float = 0.99
    """The DRACHMA waveform is normalised to a fixed area, so a different zeta can make it peakier and exceed the
    full-scale limit of 1. If any scan point of a qubit peaks above this value, the amplitude of all of that
    qubit's scan pulses is scaled down by one common factor (logged) and the zetas are divided by its square so
    the self-Kerr term is preserved. Must be in (0, 1]. Default is 0.99."""
    drachma_operation: str = "readout_drachma"
    """Name under which the per-qubit DRACHMA pulse (a DrachmaReadoutPulse that must already exist)
    is stored on qubit.resonator.operations."""
    probe_length: int = 100
    """Length (ns) of the dedicated zero-amplitude probe pulse (PROBE_OPERATION, built per qubit in
    create_qua_program and removed before saving). Must be a positive multiple of 4. Default 100 ns."""
    depletion_time: int = 5000
    """Time (ns) waited after each probe pulse, before the next (state, point) condition starts with its
    qubit reset, so the resonator is empty for sure. Overrides resonator.depletion_time for this node only.
    Must be a positive multiple of 4. Default 5 us."""
    alpha: float = 0.001
    """Significance level of the test-vs-no-operation z-test: p > alpha means the probe field is
    indistinguishable from the no-operation reference, i.e. the resonator is depleted. Default is 0.05."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
