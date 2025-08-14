import math

from calibration_utils.crosstalk_spectroscopy_vs_flux import Parameters
from quam_builder.architecture.superconducting import FluxTunableTransmon


def get_qubit_flux_detuning_for_target_slope(
    target_qubit: FluxTunableTransmon,
    target_slope_in_hz_per_v: float = 10e6/100e-3,
    expected_crosstalk: float = 1,
) -> float:
    """
    Returns the required flux detuning in V to achieve a target slope given
    some expected crosstalk percentage. Use `expected_crosstalk` = 1 if
    the target qubit is the same as the aggressor qubit.

        Uses f_q_target = A*phi^2 + C, where
            - A is the freq_vs_flux_01_quad_term, and
            - C is the RF_frequency

    And sets the derivative df/dphi_aggressor using the chain rule to the target slope

    """
    return (
        1 / expected_crosstalk * target_slope_in_hz_per_v / (2 * target_qubit.freq_vs_flux_01_quad_term)
    )


def get_target_slope_from_parameter_ranges(parameters: Parameters) -> float:
    """ Sets the target slope according to what would "fill" the 2D map. """
    return (
        parameters.frequency_span_in_mhz*1e6 / parameters.flux_offset_span_in_v
    )


def get_expected_frequency_at_flux_detuning(target_qubit: FluxTunableTransmon, flux_detuning: float) -> int:
    """ Returns the expected qubit frequency in flux at a given flux detuning in V. """
    return round(
        target_qubit.freq_vs_flux_01_quad_term * flux_detuning**2 + target_qubit.xy.RF_frequency
    )