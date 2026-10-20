from typing import Literal

from pydantic import model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of shots per state (g and e each get this many). Default is 1000.

    With `measure_variance=True`, every shot streams a full 1 ns-resolution ADC trace per qubit
    and state (`save_all`); the per-sample mean and variance are computed on the PC. With
    `measure_variance=False` only the averaged trace is streamed and the limit below does not
    apply. The node is limited by stream-processing
    bandwidth: samples streamed per batch = num_shots * T * 2 (I, Q) * 2 (g, e) * qubits in the
    batch, with T the readout length + 2 * smearing in ns (e.g. 1000 shots, T = 2000, one qubit:
    8e6 samples, ~64 MB once fetched as float64). Too many shots (or simultaneous qubits) makes
    the OPX drop samples, which the cloud backend reports as "Data loss detected in data for
    job ..." and turns into an EMPTY result dict -- not a partial one. The per-batch fetch and
    reduction times are logged to size this against the PC."""
    measure_variance: bool = False
    """Whether to stream every single-shot ADC trace (`save_all`) so the per-sample variance can
    be computed on the PC. Default is False.

    True: SNR values are absolute (V), `min_snr` is enforced, the noise-aware overflow bound uses
    the measured variance, and the variance debug plot is available. Costs `num_shots` times more
    streamed data (see `num_shots` for the data-loss limit).

    False: only the OPX-side AVERAGED trace is streamed. The variance is a flat constant
    estimated from the averaged trace (`weights.estimate_single_shot_sigma`, used only for the
    overflow bound), SNR values are in arbitrary units and `min_snr` is not enforced (the SNR
    gain and the overflow margins still decide success). The ratios between weight shapes are
    exact for white noise."""
    use_variance_weighting: bool = False
    """Whether to divide the matched-filter weight by the pooled per-sample variance (spec §3's
    optional noise-weighting switch). Default is False (matched filter on the mean envelope
    difference only). REQUIRES `measure_variance=True`: without a measured variance there is
    nothing to weight by, so `use_variance_weighting=True` with `measure_variance=False` raises
    (see `validate_variance_options`) instead of silently falling back."""
    max_qubits_per_fem: int = 1
    """Maximum number of qubits measured simultaneously per MW-FEM input when `multiplexed` is
    True. 1 (default) measures one qubit per feedline at a time: multiplexed neighbour tones
    are state-independent and cancel exactly in W = env_e - env_g, but they still inflate the
    per-sample variance that the overflow-normalization bound relies on. Raise for full
    multiplexing. Ignored when `multiplexed` is False (each qubit already gets its own batch)."""
    min_snr: float = 2.0
    """Minimum predicted matched-filter SNR for a qubit to be marked successful. Default is 2.0."""
    deployed_weight: Literal["signed", "magnitude"] = "signed"
    """Which real weight `update_state` writes, and which weight `min_snr`, the SNR gain and the
    overflow margins are judged on. Default is "signed".

    "signed": w(t) = Re[W(t) * exp(-i*phi)], the optimal real weight for a stock ReadoutPulse
    (real weight and one global angle, weights.signed_real_weight). Contains negative values.

    "magnitude": |W(t)|, which discards the sign of the projection where the g/e difference
    points away from the global axis. Fallback if hardware rejects negative weights."""
    min_detection_sigma: float = 3.0
    """How far above the noise-only expectation the deployed SNR^2 must sit for a qubit to be
    marked successful, in standard deviations of that noise-only distribution
    (weights.noise_only_snr2, a seeded simulation of the same weight construction on pure noise).
    A weight built from the averaged g/e difference scores above zero even when there is no
    contrast at all, so a gain over the constant weight alone cannot tell signal from noise.
    Default is 3.0."""
    max_snr_gain: float = 10.0
    """Upper bound on the predicted SNR gain over constant weights. A larger (or non-finite) gain
    means the constant-weight SNR is near zero (env_e - env_g cancels over time) and the ratio is
    meaningless, so the qubit is marked failed. Default is 10.0; genuine gains seen are < 3."""
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
    smooth_bandwidth_hz: float | None = 10e6
    """Cutoff of a zero-phase Hann-windowed low-pass filter (weights.lowpass_hann) applied to
    `env_g`/`env_e` after IQ-imbalance correction and before the matched-filter weight is built.
    Default is 10 MHz; None turns it off. A resonator's own dynamics live at sub-MHz to a few-MHz
    bandwidth, so 10 MHz removes averaging jitter and any residual carrier leakage (see
    `plot_weight_spectrum`) without touching real signal, and applies identically to every plot
    built from `env_g`/`env_e`. It matters for the SNR too: a weight built from the averaged g/e
    difference also fits that difference's noise, and the noise above the resonator bandwidth is
    what inflates the predicted gain (see `min_detection_sigma`). Does NOT touch the raw traces used for the OPX
    overflow bound (`weights.normalization_factor`/`hb`) or the `_demod_phase_offset` diagnostic
    -- both need to see exactly what the hardware sees, unfiltered."""


def validate_variance_options(parameters: NodeSpecificParameters) -> None:
    """Fail loudly on an invalid `measure_variance`/`use_variance_weighting` combination.

    Weighting by the variance needs a measured variance. Also called from the node's actions
    (not only the pydantic validator below) because `custom_param` and the GUI can set fields
    after construction. An explicit raise rather than `assert`, so it survives `python -O`."""
    if parameters.use_variance_weighting and not parameters.measure_variance:
        raise ValueError(
            "use_variance_weighting=True requires measure_variance=True: the matched-filter "
            "weight can only be divided by a variance that was measured. Set "
            "measure_variance=True, or set use_variance_weighting=False."
        )


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    @model_validator(mode="after")
    def _check_variance_options(self):
        validate_variance_options(self)
        return self
