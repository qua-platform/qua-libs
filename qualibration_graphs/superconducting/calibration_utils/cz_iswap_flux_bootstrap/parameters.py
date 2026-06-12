"""Parameter definitions for CZ / iSWAP flux bootstrap (node 30)."""

# pylint: disable=too-few-public-methods

import logging
from typing import Callable, ClassVar, Literal, NamedTuple, Optional
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


class QubitRoles(NamedTuple):
    """Resolved qubit roles for a two-qubit gate on a pair.

    moving:     transmon flux-tuned to the interaction point
    stationary: partner that stays at its idle frequency
    leakage:    transmon whose |2⟩ state is the dominant leakage channel
    high:       higher-frequency qubit (control/target-agnostic)
    low:        lower-frequency qubit (control/target-agnostic)
    """

    moving: object
    stationary: object
    leakage: object
    high: object
    low: object

    @staticmethod
    def _high_low(qp):
        """(high_freq_qubit, low_freq_qubit) by f_01, control/target-agnostic."""
        if qp.qubit_control.f_01 >= qp.qubit_target.f_01:
            return qp.qubit_control, qp.qubit_target
        return qp.qubit_target, qp.qubit_control

    @classmethod
    def resolve(cls, qp, gate_type: Literal["cz", "iswap"] = "cz") -> "QubitRoles":
        """Resolve moving / stationary / leakage / high / low qubits for the interaction.

        δ = f_high − f_low ≥ 0.

        iSWAP (|10⟩↔|01⟩ at δ = 0):
            High frequency qubit tunes down to the low frequency qubit. Leakage attributed to the
            high frequency qubit (weak, off-resonant effect).

        CZ via |11⟩↔|20⟩ (both photons in the HIGH frequency qubit, crossing at δ = |α_high|):
            δ >  |α_high|  →  HIGH frequency qubit moves (δ shrinks to |α_high|)
            δ ≤  |α_high|  →  LOW frequency qubit moves (δ grows   to |α_high|)
            Leaker is ALWAYS the high frequency qubit, regardless of which qubit moves.
        """
        high_q, low_q = cls._high_low(qp)

        if gate_type == "iswap":
            return cls(moving=high_q, stationary=low_q, leakage=high_q, high=high_q, low=low_q)
        if gate_type == "cz":
            delta = high_q.f_01 - low_q.f_01
            moving = high_q if delta > abs(high_q.anharmonicity) else low_q
            stationary = low_q if moving is high_q else high_q
            return cls(moving=moving, stationary=stationary, leakage=high_q, high=high_q, low=low_q)
        raise ValueError(f"Invalid gate_type value: {gate_type!r}")


def verify_moving_qubit(
    qp,
    gate_type: Literal["cz", "iswap"] = "cz",
    operation: str = "cz_unipolar",
    repair_routing: bool = False,
    log_callable: Optional[Callable[[str], None]] = None,
) -> None:
    """Verify and reconcile ``qp.moving_qubit``, then ``macros[operation]`` flux Z-line routing.

    Recalculates the moving qubit from qubit frequencies and anharmonicity via
    :func:`QubitRoles.resolve`, updates ``qp.moving_qubit`` on the pair if needed, then
    checks that ``macros[operation].flux_pulse_qubit`` is registered on that qubit's Z
    channel. ``CZGate.apply()`` reads ``qp.moving_qubit`` to pick the Z line, so the
    label is corrected before pulse routing is verified or repaired.

    Flux routing is verified using ``flux_pulse_qubit.id`` as the Z-line operation name
    (the same name passed to ``moving_qubit.z.play(...)`` at runtime). If the pulse is
    missing from the moving qubit or present only on the stationary qubit's Z line,
    raises :class:`ValueError` unless ``repair_routing=True``.

    With ``repair_routing=True``, a new Z-line entry is created on the moving qubit and wired
    to ``macros[operation].flux_pulse_qubit`` (same pattern as ``populate_quam``); any existing
    entry on the stationary Z line is left unchanged.

    Parameters
    ----------
    qp:
        QUAM qubit-pair object with ``qubit_control``, ``qubit_target``, and ``moving_qubit``.
    gate_type:
        ``"cz"`` (default) or ``"iswap"`` — selects the physics rule used by
        :func:`QubitRoles.resolve`.
    operation:
        Pair macro to verify (e.g. ``"cz_unipolar"``), matching ``node.parameters.operation``.
    repair_routing:
        If True, add the flux pulse on the moving Z line instead of raising.
    log_callable:
        Callable that accepts a single string and emits it to the node's output
        (e.g. ``node.log``). Defaults to ``logging.getLogger(__name__).info``.

    Raises
    ------
    ValueError
        If ``operation`` is not in ``qp.macros``, if its flux pulse is not on the moving
        qubit's Z line (and ``repair_routing`` is False), if the moving qubit has no Z
        channel, or if ``qp.moving_qubit`` is set to a value other than ``"control"`` or
        ``"target"``.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    roles = QubitRoles.resolve(qp, gate_type)
    computed_label = "control" if roles.moving is qp.qubit_control else "target"

    # Build a compact physics context string for logging
    delta_mhz = (roles.high.f_01 - roles.low.f_01) / 1e6
    alpha_mhz = roles.high.anharmonicity / 1e6
    physics_ctx = (
        f"high={roles.high.name} ({roles.high.f_01/1e9:.4f} GHz), "
        f"low={roles.low.name} ({roles.low.f_01/1e9:.4f} GHz), "
        f"δ={delta_mhz:.1f} MHz, α_high={alpha_mhz:.1f} MHz"
    )

    if operation not in qp.macros:
        raise ValueError(f"Pair {qp.name}: macro '{operation}' not found.")

    stored = getattr(qp, "moving_qubit", None)
    label_updated = False
    if stored is None:
        log_callable(
            f"WARNING Pair {qp.name}: qp.moving_qubit is not set. "
            f"Setting moving={roles.moving.name}, stationary={roles.stationary.name} "
            f"from recalculation ({gate_type} gate). [{physics_ctx}]"
        )
        qp.moving_qubit = computed_label
        label_updated = True
    elif stored not in ("control", "target"):
        raise ValueError(
            f"Pair {qp.name}: qp.moving_qubit={stored!r} is invalid; expected 'control' or 'target'. "
            f"[{physics_ctx}]"
        )
    elif stored != computed_label:
        stored_mq = qp.qubit_control if stored == "control" else qp.qubit_target
        stored_sq = qp.qubit_target if stored == "control" else qp.qubit_control
        log_callable(
            f"WARNING Pair {qp.name}: moving={stored_mq.name}, stationary={stored_sq.name} "
            f"disagrees with recalculation: moving={roles.moving.name}, stationary={roles.stationary.name} "
            f"for {gate_type} gate — updating qp.moving_qubit to '{computed_label}'. [{physics_ctx}]"
        )
        qp.moving_qubit = computed_label
        label_updated = True

    moving_qubit = qp.qubit_control if qp.moving_qubit == "control" else qp.qubit_target
    stationary_qubit = qp.qubit_target if qp.moving_qubit == "control" else qp.qubit_control
    macro = qp.macros[operation]
    flux_pulse = macro.flux_pulse_qubit
    pulse_name = flux_pulse.id

    moving_z = getattr(moving_qubit, "z", None)
    stationary_z = getattr(stationary_qubit, "z", None)
    on_moving = moving_z is not None and pulse_name in moving_z.operations
    on_stationary = stationary_z is not None and pulse_name in stationary_z.operations
    if not on_moving:
        if moving_z is None:
            raise ValueError(
                f"Pair {qp.name}: moving qubit {moving_qubit.name} has no Z channel."
            )
        if repair_routing:
            ref = flux_pulse.get_reference()
            init = {
                attr: getattr(flux_pulse, attr)
                for attr in ("length", "amplitude", "flat_length", "sigma")
                if hasattr(flux_pulse, attr) and isinstance(getattr(flux_pulse, attr), (int, float))
            }
            if flux_pulse.id is not None:
                init["id"] = flux_pulse.id
            moving_z.operations[pulse_name] = type(flux_pulse)(**init)
            z_pulse = moving_z.operations[pulse_name]
            for attr in ("length", "amplitude", "flat_length", "digital_marker", "axis_angle"):
                if hasattr(flux_pulse, attr):
                    setattr(z_pulse, attr, f"{ref}/{attr}")
            if on_stationary:
                log_callable(
                    f"WARNING Pair {qp.name}: added macro '{operation}' flux pulse '{pulse_name}' "
                    f"to {moving_qubit.name} Z line (wired to macro; copy still on "
                    f"{stationary_qubit.name})."
                )
            else:
                log_callable(
                    f"WARNING Pair {qp.name}: added macro '{operation}' flux pulse '{pulse_name}' "
                    f"to {moving_qubit.name} Z line (wired to macro)."
                )
        elif on_stationary:
            raise ValueError(
                f"Pair {qp.name}: macro '{operation}' flux pulse '{pulse_name}' is on stationary qubit "
                f"{stationary_qubit.name} Z line but moving qubit is {moving_qubit.name}. "
                f"Re-run populate_quam or set repair_routing=True."
            )
        else:
            raise ValueError(
                f"Pair {qp.name}: macro '{operation}' flux pulse '{pulse_name}' not found on moving qubit "
                f"{moving_qubit.name} Z line. Re-run populate_quam or set repair_routing=True."
            )
    elif not label_updated:
        log_callable(
            f"Pair {qp.name}: moving={moving_qubit.name}, stationary={stationary_qubit.name}. "
            f"Label and flux-pulse routing consistent with recalculation ({gate_type} gate)."
        )


def estimate_qubit_flux_shift(
    parameters: NodeSpecificParameters,
    qp,
    log_callable: Optional[Callable[[str], None]] = None,
) -> float:
    """Return the flux bias centre for the moving-qubit sweep (CZ or iSWAP).

    If ``parameters.use_saved_detuning`` is True, reads ``qp.detuning`` directly from state json file.
    Otherwise, resolves the qubit roles via :func:`QubitRoles.resolve` and estimates the flux
    shift needed to reach the interaction point from the moving qubit's
    ``freq_vs_flux_01_quad_term``.

    Conventions
    -----------
    ``detuning_hz`` is the POSITIVE "distance to tune the moving qubit DOWN"
    (freq_now − target_freq ≥ 0). With ``quad < 0`` (upper sweet spot), the flux solves
    ``Φ² = -detuning_hz / quad ≥ 0``.

    iSWAP (|10⟩↔|01⟩ at δ = 0):
        Tune the moving (high) qubit down to the stationary (low) qubit:
            detuning_hz = f_moving − f_stationary = δ ≥ 0.

    CZ via the |11⟩↔|20⟩ avoided crossing (both photons in the HIGH qubit,
    crossing at δ = |α_high|):
        - HIGH qubit moves (δ > |α_high|): target = f_low + |α_high|
              detuning_hz = (f_high − f_low) − |α_high| = δ − |α_high| ≥ 0
        - LOW  qubit moves (δ ≤ |α_high|): target = f_high − |α_high|
              detuning_hz = |α_high| − (f_high − f_low) = |α_high| − δ ≥ 0

    Parameters
    ----------
    parameters:
        Node parameters carrying ``cz_or_iswap`` and ``use_saved_detuning``.
    qp:
        QUAM qubit-pair object.
    log_callable:
        Callable for progress messages (e.g. ``node.log``). Defaults to the module logger.

    Returns
    -------
    float
        Flux bias centre in volts for the moving-qubit sweep axis.

    Raises
    ------
    ValueError
        If ``use_saved_detuning`` is True but ``qp.detuning`` is unset, if
        ``freq_vs_flux_01_quad_term`` is zero on the moving qubit, if the iSWAP
        moving qubit is below its partner, or if the computed flux ratio is
        negative (inconsistent frequencies / anharmonicity / quad term).
    """
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
        roles = QubitRoles.resolve(qp, parameters.cz_or_iswap)
        quad = roles.moving.freq_vs_flux_01_quad_term
        if quad == 0:
            raise ValueError(
                f"Pair {qp.name}: moving qubit '{roles.moving.name}' has freq_vs_flux_01_quad_term=0. "
                f"Run 09a_ramsey_vs_flux_calibration on {roles.moving.name} first, or set "
                "qubit_pair.detuning and use_saved_detuning=True."
            )

        if parameters.cz_or_iswap == "iswap":
            detuning_hz = roles.moving.xy.RF_frequency - roles.stationary.xy.RF_frequency
            if detuning_hz < 0:
                raise ValueError(
                    f"Pair {qp.name} [iSWAP]: moving qubit '{roles.moving.name}' "
                    f"({roles.moving.xy.RF_frequency/1e9:.4f} GHz) is below partner '{roles.stationary.name}' "
                    f"({roles.stationary.xy.RF_frequency/1e9:.4f} GHz) by {abs(detuning_hz)/1e6:.1f} MHz. "
                    "iSWAP requires the flux-tunable qubit to tune down to the partner."
                )
        elif parameters.cz_or_iswap == "cz":
            # |11>-|20> crossing, always referenced to |alpha_high| (abs() so this is
            # correct whether anharmonicity is stored signed or as a positive magnitude):
            #   high moves: target = f_low + |a_high|;  detuning = (f_high - f_low) - |a_high|
            #   low  moves: target = f_high - |a_high|;  detuning = |a_high| - (f_high - f_low)
            alpha_high = abs(roles.high.anharmonicity)
            delta_rf = roles.high.xy.RF_frequency - roles.low.xy.RF_frequency
            if roles.moving is roles.high:
                detuning_hz = delta_rf - alpha_high
            else:
                detuning_hz = alpha_high - delta_rf
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
