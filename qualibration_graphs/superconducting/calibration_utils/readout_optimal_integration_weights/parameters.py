from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of shots per state (g and e each get this many). Default is 1000.

    Every shot streams a full 1 ns-resolution ADC trace per qubit and state (`save_all`); the
    per-sample mean and variance are computed on the PC. The node is limited by stream-processing
    bandwidth: samples streamed per batch = num_shots * T * 2 (I, Q) * 2 (g, e) * qubits in the
    batch, with T the readout length + 2 * smearing in ns (e.g. 1000 shots, T = 2000, one qubit:
    8e6 samples, ~64 MB once fetched as float64). Too many shots (or simultaneous qubits) makes
    the OPX drop samples, which the cloud backend reports as "Data loss detected in data for
    job ..." and turns into an EMPTY result dict -- not a partial one. The per-batch fetch and
    reduction times are logged to size this against the PC."""
    use_variance_weighting: bool = False
    """Whether to divide the matched-filter weight by the pooled per-sample variance (spec
    §3's optional noise-weighting switch). The variance is computed on the PC from the
    single-shot traces. Default is False (matched filter on the mean envelope difference only)."""
    max_qubits_per_fem: int = 1
    """Maximum number of qubits measured simultaneously per MW-FEM input when `multiplexed` is
    True. 1 (default) measures one qubit per feedline at a time: multiplexed neighbour tones
    are state-independent and cancel exactly in W = env_e - env_g, but they still inflate the
    per-sample variance that the overflow-normalization bound relies on. Raise for full
    multiplexing. Ignored when `multiplexed` is False (each qubit already gets its own batch)."""
    min_snr: float = 2.0
    """Minimum predicted matched-filter SNR for a qubit to be marked successful. Default is 2.0."""
    debug_plots: bool = False
    """Whether to produce the extra debugging plots (variance, before/after software
    demodulation in time and frequency, normalization headroom) and compute the hardware
    demod-phase diagnostic, in addition to the five plots that are always produced (predicted
    SNR, weight spectrum, demodulated envelopes, normalized weight, IQ trajectory).
    Default is False."""
    correct_iq_imbalance: bool = True
    """Whether to estimate an IQ-imbalance ratio `b` (mirror-image model y = x + b*conj(x)) from
    the ground-state trace and fold its correction into the matched-filter weight (see
    weights.fold_iq_imbalance_into_weight). Default is True. The estimate is guarded by
    `max_iq_imbalance`: a rejected estimate falls back to no correction (b=0), never aborts the
    qubit -- see OptimalWeightsFit.iq_imbalance_applied/iq_imbalance_reason."""
    max_iq_imbalance: float = 0.5
    """Reject-and-fall-back threshold on the estimated |b|. Physical mixer/ADC imbalance is a
    few percent; an estimate approaching 1 indicates a failed fit (e.g. a DC-dominated spectrum
    or too short a trace), not a real imbalance. Default is 0.5. Ignored when
    `correct_iq_imbalance` is False."""
    smooth_bandwidth_hz: float | None = None
    """Cutoff of an optional zero-phase Hann-windowed low-pass filter (weights.lowpass_hann)
    applied to `env_g`/`env_e` after IQ-imbalance correction and before the matched-filter weight
    is built. Default is None (off). A resonator's own dynamics live at sub-MHz to a few-MHz
    bandwidth, so a value like `10e6` (10 MHz) removes averaging jitter and any residual carrier
    leakage (see `plot_weight_spectrum`) without touching real signal, and applies identically to
    every plot built from `env_g`/`env_e`. Does NOT touch the raw traces used for the OPX
    overflow bound (`weights.normalization_factor`/`hb`) or the `_demod_phase_offset` diagnostic
    -- both need to see exactly what the hardware sees, unfiltered."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
