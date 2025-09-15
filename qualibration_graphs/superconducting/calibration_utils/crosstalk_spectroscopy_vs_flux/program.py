import math

from calibration_utils.crosstalk_spectroscopy_vs_flux import Parameters
from quam_builder.architecture.superconducting import FluxTunableTransmon


def get_flux_detuning_in_v(node_parameters: Parameters, target_qubit: FluxTunableTransmon) -> float:
    """
    Returns the flux detuning of the target qubit in units of Volts according
    to the specified strategy in the node parameters. In essence,
        "manual"
            sets the flux detuning to a fixed value specified in the config.
        "auto_for_linear_response"
            sets the flux detuning so that the frequency response is "linear" within 1%,
            given some expected crosstalk.
        "auto_fill_sweep_window"
            sets the flux detuning so that it fills the flux vs. freq. sweep window,
            given some expected crosstalk.
    """
    if node_parameters.flux_detuning_mode == "manual":
        flux_detuning = node_parameters.manual_flux_detuning_in_v

    elif node_parameters.flux_detuning_mode == "auto_for_linear_response":
        # set target qubit to just-right flux-sensitive point
        flux_detuning_linear = get_qubit_flux_detuning_for_one_percent_tangent_error_on_parabola(
            node_parameters.flux_span_in_v, node_parameters.expected_crosstalk
        )
        target_slope = get_target_slope_from_parameter_ranges(node_parameters)

        flux_detuning_fill = get_qubit_flux_detuning_for_target_slope(
            target_qubit=target_qubit,
            target_slope_in_hz_per_v=target_slope,
            expected_crosstalk=node_parameters.expected_crosstalk,
        )

        flux_detuning = max(flux_detuning_linear, flux_detuning_fill)
    else:
        raise NotImplementedError(f"Unrecognized flux detuning mode {node_parameters.flux_detuning_mode}")

    return flux_detuning


def get_qubit_flux_detuning_for_one_percent_tangent_error_on_parabola(
    flux_span_in_v: float, expected_crosstalk: float
):
    """
    Returns the required flux detuning on the target qubit so that the specified
    flux span measures a window of the parabola whose tangent line has no more than
    1% error with respect to the parabola itself, given some expected crosstalk. If 
    the crosstalk is *higher* than expected, this condition will not be met.
    
    The result stems from the following formula, which divides the frequency response
    parabola as a function of aggressor qubit flux (x) by the tangent line formula.
    | (𝑨(𝑪𝑥−𝑩)²)⁄((2𝑨(−𝑩))(𝑪𝑥)+𝑨𝑩²) − 1 | < T.

    Interestingly, it doesn't depend on the frequency vs. flux quadrature term.
    """
    T = 0.01  # error threshold
    C = expected_crosstalk
    x = flux_span_in_v / 2

    return C * x / (-T + math.sqrt(T*(T+1)))


def get_qubit_flux_detuning_for_target_slope(
    target_qubit: FluxTunableTransmon,
    target_slope_in_hz_per_v: float = 10e6/100e-3,
    expected_crosstalk: float = 1,
) -> float:
    """
    Returns the required flux detuning in V on the target qubit required to achieve
    a specified target slope, given some expected crosstalk percentage. Use
    `expected_crosstalk` = 1 if the target qubit is the same as the aggressor qubit.

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
        parameters.frequency_span_in_mhz*1e6 / parameters.flux_span_in_v
    )


def get_expected_frequency_at_flux_detuning(target_qubit: FluxTunableTransmon, flux_detuning: float) -> int:
    """ Returns the expected qubit frequency in flux at a given flux detuning in V. """
    return round(
        target_qubit.freq_vs_flux_01_quad_term * flux_detuning**2 + target_qubit.xy.RF_frequency
    )