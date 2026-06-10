"""Parameter definitions for CZ / iSWAP flux bootstrap (node 30)."""

# pylint: disable=too-few-public-methods

import logging
from typing import Callable, ClassVar, Literal, Optional
import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for the CZ / iSWAP flux bootstrap node: 2D coupler and moving-qubit flux sweeps.

    Attributes
    ----------
    num_shots : int, default = 100
        Number of acquisition shots (repetitions) per flux point and pulse condition.
        Larger values improve SNR but increase total experiment time.

    coupler_flux_min : float, default = -0.05
        Minimum relative coupler flux bias (in normalized flux units) with respect to the
        current coupler operating set point. Negative values move below the set point.

    coupler_flux_max : float, default = 0.05
        Maximum relative coupler flux bias (in normalized flux units) with respect to the
        current coupler operating set point. Positive values move above the set point.

    coupler_flux_step : float, default = 0.0005
        Increment used when sweeping the coupler flux between coupler_flux_min and
        coupler_flux_max. Controls resolution of the coupler flux scan.

    qubit_flux_span : float, default = 0.05
        Total relative span (peak-to-peak) of the flux detuning applied across (or between)
        the two qubits, referenced to the known/calculated nominal detuning.

    qubit_flux_step : float, default = 0.0005
        Increment used when sweeping qubit flux detuning across the specified span.
        Determines granularity of the qubit detuning scan.

    operation : str, default = "cz_unipolar"
        Pair macro used for the flux sweep and state update (e.g. ``cz_unipolar`` or
        ``iswap_unipolar``). Set this to the macro you are calibrating.

    cz_or_iswap : Literal["cz", "iswap"], default = "cz"
        Specifies which entangling interaction is being characterized: controlled-Z ("cz")
        or iSWAP ("iswap"). Controls preparation, readout, and analysis only.

    use_saved_detuning : bool, default = False
        If True, reuse previously extracted qubit detuning value rather than recalculating.
        Speeds up subsequent runs but risks drift if the system has changed.

    analysis_debug : bool, default = False
        If True, ``plot_data`` adds the 1D contrast-cut debug figure (smoothed trace, flat /
        oscillation masks, decouple and gate coupler markers). Diagnostic only; does not
        affect fitting.

    analysis_fit_preset : Literal["default", "noisy", "coarse"], default = "default"
        Contrast-cut fit profile. ``default`` for normal coupler sweep resolution and SNR;
        ``noisy`` for good resolution but poor SNR (try more shots first); ``coarse`` for a
        wide exploratory coupler scan to locate idle and interaction dynamics.

    Notes
    -----
    Choosing smaller step sizes greatly increases the number of measurement points:
        num_coupler_points ≈ (coupler_flux_max - coupler_flux_min) / coupler_flux_step + 1
        num_qubit_points   ≈ qubit_flux_span / qubit_flux_step + 1
    Total shots ≈ num_shots * num_coupler_points * num_qubit_points (for joint sweeps).

    The selected macro operation determines the underlying pulse shape and duration.
    """

    num_shots: int = 100
    """Number of shots to perform. Default is 100."""
    coupler_flux_min: float = -0.1
    """Minimum coupler flux bias relative to the coupler set point. Default is -0.1."""
    coupler_flux_max: float = 0.1
    """Maximum coupler flux bias relative to the coupler set point. Default is 0.1."""
    coupler_flux_step: float = 0.001
    """Step size for the coupler flux sweep. Default is 0.001."""
    qubit_flux_span: float = 0.1
    """Total qubit flux detuning span, relative to the known/calculated detuning between the qubits. Default is 0.1."""
    qubit_flux_step: float = 0.001
    """Step size for the qubit flux detuning sweep. Default is 0.001."""
    operation: str = "cz_unipolar"
    """Pair macro for sweep and state update (e.g. cz_unipolar or iswap_unipolar). Default is cz_unipolar."""
    cz_or_iswap: Literal["cz", "iswap"] = "cz"
    """Entangling interaction to characterise: 'cz' or 'iswap'. Default is 'cz'."""
    use_saved_detuning: bool = False
    """Whether to reuse the previously extracted qubit detuning instead of recalculating. Default is False."""
    analysis_debug: bool = False
    """If True, add the 1D contrast-cut debug figure in plot_data. Default is False."""
    analysis_fit_preset: Literal["default", "noisy", "coarse"] = "default"
    """Contrast-cut fit preset: default, noisy, or coarse. Default is default."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for CZ / iSWAP flux bootstrap."""

    targets_name: ClassVar[str] = "qubit_pairs"


def get_moving_qubit(qp, gate_type: Literal["cz", "iswap"] = "cz"):
    """Transmon that is flux-tuned for the selected two-qubit interaction.

    For both CZ and iSWAP, the higher-frequency qubit is always identified first.
    δ = f_high − f_low  (always positive, qubit-role-agnostic).

    iSWAP:
        The high-freq qubit tunes down to meet the low-freq qubit at resonance.

    CZ:
        - δ > |α_high|  →  high-freq qubit moves  (|11⟩↔|20⟩ avoided crossing)
        - δ ≤ |α_high|  →  low-freq qubit moves   (|11⟩↔|02⟩ avoided crossing)
    """
    if qp.qubit_control.f_01 >= qp.qubit_target.f_01:
        high_q, low_q = qp.qubit_control, qp.qubit_target
    else:
        high_q, low_q = qp.qubit_target, qp.qubit_control

    if gate_type == "iswap":
        return high_q
    if gate_type == "cz":
        delta = high_q.f_01 - low_q.f_01
        if delta > abs(high_q.anharmonicity):
            return high_q
        return low_q
    raise ValueError(f"Invalid gate_type value: {gate_type}")


def get_stationary_qubit(qp, gate_type: Literal["cz", "iswap"] = "cz"):
    """Partner transmon of the moving qubit (stays at its idle frequency)."""
    if get_moving_qubit(qp, gate_type) is qp.qubit_target:
        return qp.qubit_control
    return qp.qubit_target


def verify_moving_qubit(
    qp,
    gate_type: Literal["cz", "iswap"] = "cz",
    log_callable: Optional[Callable[[str], None]] = None,
) -> None:
    """Verify that qp.moving_qubit matches the physics-based calculation; update in-place if not.

    Recalculates the moving qubit from qubit frequencies and anharmonicity, then compares
    it to the ``moving_qubit`` field stored on the QUAM qubit-pair object. If they disagree,
    logs a warning, updates ``qp.moving_qubit`` in memory, and relies on the caller's state save
    to persist the correction to disk.

    Parameters
    ----------
    qp:
        QUAM qubit-pair object with ``qubit_control``, ``qubit_target``, and ``moving_qubit``.
    gate_type:
        ``"cz"`` (default) or ``"iswap"`` — selects the physics rule used by
        :func:`get_moving_qubit`.
    log_callable:
        Callable that accepts a single string and emits it to the node's output
        (e.g. ``node.log``). Defaults to ``logging.getLogger(__name__).info``.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    computed_mq = get_moving_qubit(qp, gate_type)
    computed_sq = get_stationary_qubit(qp, gate_type)
    computed_label = "control" if computed_mq is qp.qubit_control else "target"

    # Build a compact physics context string for logging
    if qp.qubit_control.f_01 >= qp.qubit_target.f_01:
        high_q, low_q = qp.qubit_control, qp.qubit_target
    else:
        high_q, low_q = qp.qubit_target, qp.qubit_control
    delta_mhz = (high_q.f_01 - low_q.f_01) / 1e6
    alpha_mhz = high_q.anharmonicity / 1e6
    physics_ctx = (
        f"high={high_q.name} ({high_q.f_01/1e9:.4f} GHz), "
        f"low={low_q.name} ({low_q.f_01/1e9:.4f} GHz), "
        f"δ={delta_mhz:.1f} MHz, α_high={alpha_mhz:.1f} MHz"
    )

    stored = getattr(qp, "moving_qubit", None)
    stored_mq = qp.qubit_control if stored == "control" else qp.qubit_target
    stored_sq = qp.qubit_target if stored == "control" else qp.qubit_control

    if stored is None:
        log_callable(
            f"WARNING Pair {qp.name}: qp.moving_qubit is not set. "
            f"Setting moving={computed_mq.name}, stationary={computed_sq.name} "
            f"from recalculation ({gate_type} gate). [{physics_ctx}]"
        )
        qp.moving_qubit = computed_label
    elif stored != computed_label:
        log_callable(
            f"WARNING Pair {qp.name}: moving={stored_mq.name}, stationary={stored_sq.name} "
            f"disagrees with recalculation: moving={computed_mq.name}, stationary={computed_sq.name} "
            f"for {gate_type} gate — updating. "
            f"State will be persisted at the end of the node. [{physics_ctx}]"
        )
        qp.moving_qubit = computed_label
    else:
        log_callable(
            f"Pair {qp.name}: moving={computed_mq.name}, stationary={computed_sq.name}. "
            f"Consistent with recalculation ({gate_type} gate). [{physics_ctx}]"
        )


def estimate_qubit_flux_shift(
    parameters: NodeSpecificParameters,
    qp,
    log_callable: Optional[Callable[[str], None]] = None,
) -> float:
    """Centre the qubit flux sweep on saved or estimated detuning (CZ or iSWAP)."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    if parameters.use_saved_detuning:
        if qp.detuning is None:
            raise ValueError(
                f"Pair {qp.name}: detuning is unset. Set qubit_pair.detuning in state or disable "
                "use_saved_detuning to estimate from frequencies."
            )
        centre = qp.detuning
        source = "from qubit_pair.detuning"
    else:
        moving_q = get_moving_qubit(qp, parameters.cz_or_iswap)
        stationary_q = get_stationary_qubit(qp, parameters.cz_or_iswap)
        quad = moving_q.freq_vs_flux_01_quad_term
        if quad == 0:
            raise ValueError(
                f"Pair {qp.name}: moving qubit '{moving_q.name}' has freq_vs_flux_01_quad_term=0. "
                f"Run 09a_ramsey_vs_flux_calibration on {moving_q.name} first, or set "
                "qubit_pair.detuning and use_saved_detuning=True."
            )

        if parameters.cz_or_iswap == "iswap":
            detuning_hz = moving_q.xy.RF_frequency - stationary_q.xy.RF_frequency
            if detuning_hz < 0:
                raise ValueError(
                    f"Pair {qp.name} [iSWAP]: moving qubit '{moving_q.name}' "
                    f"({moving_q.xy.RF_frequency/1e9:.4f} GHz) is below partner '{stationary_q.name}' "
                    f"({stationary_q.xy.RF_frequency/1e9:.4f} GHz) by {abs(detuning_hz)/1e6:.1f} MHz. "
                    "iSWAP requires the flux-tunable qubit to tune down to the partner."
                )
        elif parameters.cz_or_iswap == "cz":
            detuning_hz = moving_q.xy.RF_frequency - stationary_q.xy.RF_frequency + stationary_q.anharmonicity
        else:
            raise ValueError(f"Invalid cz_or_iswap value: {parameters.cz_or_iswap}")

        ratio = -detuning_hz / quad
        if ratio < 0:
            raise ValueError(
                f"Pair {qp.name}: cannot estimate flux shift (sqrt of negative ratio {ratio:.3g}). "
                "Check qubit frequencies, anharmonicity, and freq_vs_flux_01_quad_term."
            )
        centre = float(np.sqrt(ratio))
        source = f"calculated for {parameters.cz_or_iswap}"
        if qp.detuning is not None:
            source += f", ignoring qubit_pair.detuning={qp.detuning:.6f} V"

    log_callable(f"Pair {qp.name}: qubit flux centre = {centre:.6f} V ({source})")
    return centre
