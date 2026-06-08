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
    """Minimum coupler flux bias relative to the coupler set point. Default is -0.05."""
    coupler_flux_max: float = 0.1
    """Maximum coupler flux bias relative to the coupler set point. Default is 0.05."""
    coupler_flux_step: float = 0.002
    """Step size for the coupler flux sweep. Default is 0.0005."""
    qubit_flux_span: float = 0.1
    """Total qubit flux detuning span, relative to the known/calculated detuning between the qubits. Default is 0.05."""
    qubit_flux_step: float = 0.002
    """Step size for the qubit flux detuning sweep. Default is 0.0005."""
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


def moving_qubit(qp):
    """Transmon that carries ``flux_pulse_qubit`` (``qubit_pair.moving_qubit``)."""
    if qp.moving_qubit == "target":
        return qp.qubit_target
    return qp.qubit_control


def stationary_qubit(qp):
    """Partner transmon of the moving qubit (stays at its idle frequency)."""
    if qp.moving_qubit == "target":
        return qp.qubit_control
    return qp.qubit_target


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
        moving_q = moving_qubit(qp)
        stationary_q = stationary_qubit(qp)
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
