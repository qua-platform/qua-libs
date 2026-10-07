from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per condition. Default is 1000."""
    test_operation: str = "readout_drachma"
    """Name of the readout operation (on qubit.resonator.operations) whose residual photons are
    measured -- e.g. "readout_square", a CLEAR pulse or "readout_drachma". It is compared against
    a "no_operation" reference. Qubits lacking this operation are skipped. The residual-photon
    probe itself always uses a dedicated zero-amplitude pulse of length probe_length -- see the
    node's PROBE_OPERATION and create_qua_program."""
    probe_length: int = 3000
    """Length (ns) of the zero-amplitude residual-photon probe pulse (PROBE_OPERATION, built per
    qubit in create_qua_program). Same for every qubit. Default 3 us."""
    depletion_time: int = 5000
    """Time (ns) waited after each probe pulse, before the next (condition, state) starts with its qubit
    reset, so the resonator is empty for sure. Overrides resonator.depletion_time for this node's wait only
    (the value written to qubit.resonator.depletion_time in update_state is the measured one). Must be a
    positive multiple of 4. Default 5 us."""
    segment_length_ns: int = 100
    """Length (ns) of each sliced-demodulation segment of the probe. Must be a multiple of 4 and
    divide probe_length exactly (no partial trailing segment); the number of segments is
    probe_length // segment_length_ns, identical for every qubit."""
    depletion_debounce_segments: int = 1
    """Number of consecutive segments a depletion test's pass condition (p_value > alpha) must
    hold before that segment is accepted as the depletion time -- avoids a single noisy segment
    triggering a false-early depletion time. Shared by compute_stat_depletion_time and
    compute_ge_depletion_time (K). Must be >=1 (0 would make the pass-condition check on an
    empty window, which numpy treats as vacuously true -- both functions clamp this internally
    too, but keep the default sane for runs that skip custom_param, e.g. external/orchestrator
    runs)."""
    alpha: float = 0.01
    """Significance threshold for the statistical (chi-squared) depletion-time test: a segment
    counts as "depleted" once its p-value (test_operation vs no_operation) exceeds alpha."""
    min_depletion_time_ns: int = 16
    """Lower bound (ns) written to qubit.resonator.depletion_time in update_state. Measured
    ground-vs-excited depletion times below this are clamped up (QUA wait uses 4 ns clock cycles,
    so 16 ns is the shortest practical depletion wait)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
