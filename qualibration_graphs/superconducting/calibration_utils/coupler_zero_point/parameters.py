"""Parameter definitions for coupler zero-point calibration.

This module defines the parameters used in coupler zero-point calibration
experiments, including sweep ranges and analysis options.
"""

# pylint: disable=too-few-public-methods

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters specific to a single calibration node for coupler zero-point characterization
    and flux detuning scans between two superconducting qubits.

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

    operation : Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"], default = "cz_unipolar"
        Macro-based CZ operation to use for the flux pulses. The node scales the qubit and
        coupler pulse amplitudes through this macro interface during the sweep.

    cz_or_iswap : Literal["cz", "iswap"], default = "cz"
        Specifies which entangling interaction is being characterized: controlled-Z ("cz")
        or iSWAP ("iswap"). May alter pulse shape, amplitude constraints, and analysis
        routines.

    use_saved_detuning : bool, default = False
        If True, reuse previously extracted qubit detuning value rather than recalculating.
        Speeds up subsequent runs but risks drift if the system has changed.

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
    coupler_flux_min: float = -0.05
    """Minimum coupler flux bias relative to the coupler set point. Default is -0.05."""
    coupler_flux_max: float = 0.05
    """Maximum coupler flux bias relative to the coupler set point. Default is 0.05."""
    coupler_flux_step: float = 0.0005
    """Step size for the coupler flux sweep. Default is 0.0005."""
    qubit_flux_span: float = 0.05
    """Total qubit flux detuning span, relative to the known/calculated detuning between the qubits. Default is 0.05."""
    qubit_flux_step: float = 0.0005
    """Step size for the qubit flux detuning sweep. Default is 0.0005."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Type of CZ operation to perform. Options are 'cz_flattop', 'cz_unipolar', 'cz_bipolar', 'cz_flattop_erf', or 'cz_SNZ'. Default is 'cz_unipolar'."""
    cz_or_iswap: Literal["cz", "iswap"] = "cz"
    """Entangling interaction to characterise: 'cz' or 'iswap'. Default is 'cz'."""
    use_saved_detuning: bool = False
    """Whether to reuse the previously extracted qubit detuning instead of recalculating. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for coupler zero-point calibration."""

    targets_name: ClassVar[str] = "qubit_pairs"
