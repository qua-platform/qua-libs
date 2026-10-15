from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per (state, kappa point). Default is 1000."""
    kappa_num_points: int = 15
    """Number of kappa points scanned per state. Every point adds one temporary DRACHMA pulse per
    qubit (both states are measured with it) to the config, so keep it modest. Default is 15."""
    kappa_step_hz: float = 50e3
    """Spacing (Hz) between consecutive kappa points. Together with kappa_num_points it sets the scan range,
    which is centred on the current kappa (kappa on the grid, the extra point of an even count above it) and
    shifted up if the lower end would not stay positive. Must be > 0. Default is 50 kHz."""
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
    alpha: float = 0.01
    """Significance level of the test-vs-no-operation z-test: p > alpha means the probe field is
    indistinguishable from the no-operation reference, i.e. the resonator is depleted. Default is 0.01."""
    flatness_alpha: float = 0.01
    """Significance level of the per-state flatness test (chi-square of "the residual power is constant over the
    scan"): p > flatness_alpha means the state does not respond to kappa and keeps its current value.
    Default is 0.01."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
