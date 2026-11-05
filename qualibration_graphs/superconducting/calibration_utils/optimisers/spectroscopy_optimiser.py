"""Adaptive spectroscopy parameter resolvers and retry validators."""

from qualibrate import QualibrationNode
from typing import Any


def resolve_resspec_params(node: QualibrationNode, target: str) -> bool:
    """Narrow the span and step after a successful fit; otherwise widen the span."""
    fit = node.results.get("fit_results", {}).get(target, {})
    curr_span = node.parameters.frequency_span_in_mhz
    curr_step = node.parameters.frequency_step_in_mhz
    print(f"ran previously with {curr_span}")
    if fit.get("success"):
        return {"frequency_span_in_mhz": curr_span * 0.6, "frequency_step_in_mhz": curr_step * 0.6}

    return {"frequency_span_in_mhz": curr_span * 1.3}


def resolve_qspec_params(node: QualibrationNode, target: str) -> bool:
    """Return a drive amplitude factor scaled by 0.75 and capped at 0.4."""
    amp_factor = node.parameters.operation_amplitude_factor
    curr_span = node.parameters.frequency_span_in_mhz
    print(f"ran previously with {amp_factor} and {curr_span}")

    max_power_to_rerun = 0.4
    amp_factor_to_return = min(max_power_to_rerun, amp_factor * 0.75)
    return {"operation_amplitude_factor": amp_factor_to_return}


def resolve_qspec_params_pulse_len(node: QualibrationNode, target: str) -> bool:
    """Scale an existing pulse length by 1.5, with a 10,000 ns minimum and fallback."""
    fit = node.results.get("fit_results", {}).get(target, {})
    amp_factor = node.parameters.operation_amplitude_factor
    prev_pulse_len = node.parameters.operation_len_in_ns
    curr_span = node.parameters.frequency_span_in_mhz
    print(f"ran previously with {amp_factor} and {curr_span}")
    if prev_pulse_len:
        pulse_len_to_return = max(prev_pulse_len * 1.5, 10000)
    else:
        pulse_len_to_return = 10000

    return {"operation_len_in_ns": pulse_len_to_return}


def resolve_qspec_params_advanced(node: QualibrationNode, target: str) -> bool:
    """Return amplitude, pulse-length, or shot-count overrides from fit results."""
    max_power_to_rerun = 0.4
    min_pulse_len_to_rerun = 10000

    qubit_fit = node.results.get("fit_results", {}).get(target, {})

    amp_factor = node.parameters.operation_amplitude_factor
    curr_span = node.parameters.frequency_span_in_mhz
    prev_num_shots = node.parameters.num_shots
    prev_pulse_len = node.parameters.operation_len_in_ns

    # Use the machine operation length when no explicit pulse length was supplied.
    if prev_pulse_len is None:
        qubit = node.machine.qubits[target]
        operation = node.parameters.operation
        prev_pulse_len = qubit.xy.operations[operation].length
        min_pulse_len_to_rerun = prev_pulse_len

    print(f"ran previously with {amp_factor} for {prev_pulse_len} ns over {curr_span} MHz range")

    r2_threshold = 0.7
    snr_threshold = 6.0

    return_dict = {}

    # Missing fit data or a successful fit selects fixed amplitude and length overrides.
    if not qubit_fit or qubit_fit.get("success"):
        return {
            "operation_amplitude_factor": max_power_to_rerun,
            "operation_len_in_ns": min_pulse_len_to_rerun,
            "num_shots": max(500, int(prev_num_shots * 1.2)),
        }

    # A truthy R2 below 0.7 reduces amplitude and increases averaging.
    if qubit_fit.get("r2", float("inf")) and qubit_fit.get("r2", float("inf")) < r2_threshold:
        return_dict["operation_amplitude_factor"] = min(max_power_to_rerun, amp_factor * 0.6)
        return_dict["num_shots"] = max(300, int(prev_num_shots * 1.2))

    # This branch checks peak_snr truthiness but compares R2 with the SNR threshold.
    # Its overrides replace any amplitude and shot count set by the R2 branch above.
    if qubit_fit.get("peak_snr", float("inf")) and qubit_fit.get("peak_snr", float("inf")) < snr_threshold:
        return_dict["num_shots"] = max(400, int(prev_num_shots * 1.5))
        return_dict["operation_amplitude_factor"] = min(max_power_to_rerun, amp_factor * 0.8)

    # If neither quality branch applies, cap amplitude and increase averaging.
    if len(return_dict) < 1:
        return_dict = {
            "operation_amplitude_factor": min(max_power_to_rerun, amp_factor),
            "operation_len_in_ns": max(min_pulse_len_to_rerun, prev_pulse_len),
            "num_shots": max(600, int(prev_num_shots * 1.2)),
        }

    return return_dict


def validate_qspec(node: QualibrationNode, target: str) -> bool:
    """Return True to retry when the target fit is missing or unsuccessful."""
    fit = node.results.get("fit_results", {}).get(target, {})

    if fit.get("success"):
        return False
    else:
        return True


def validate_qspec_fwhm(node: QualibrationNode, target: str) -> bool:
    """Retry unless the fit succeeds with a FWHM in the inclusive 12-25 MHz range."""
    fit = node.results.get("fit_results", {}).get(target, {})
    fwhm_min = 12e6
    fwhm_max = 25e6

    if not fit.get("success"):
        node.log(f"Fit was not successful the loop will rerun.")
        return True

    fwhm = fit.get("fwhm", float("inf"))

    in_range = fwhm_min <= fwhm <= fwhm_max
    should_rerun = not in_range

    node.log(
        f"FWHM is {fwhm } Hz and "
        f"{'is' if in_range else 'is not'} in the range "
        f"[{fwhm_min }, {fwhm_max}] Hz. "
        f"The loop {'will' if should_rerun else 'will not'} run."
    )

    return should_rerun


def validate_qspec_fine(node: QualibrationNode, target: str) -> bool:
    """Retry unless the fit succeeds with FWHM in 6-25 MHz inclusive and R2 >= 0.8."""
    fit = node.results.get("fit_results", {}).get(target, {})
    fwhm_min = 6e6
    fwhm_max = 25e6
    r2_min = 0.8
    if not fit.get("success"):
        node.log(f"Fit was not successful the loop will rerun.")
        return True

    fwhm = fit.get("fwhm", float("inf"))
    r2 = fit.get("r2", float("nan"))

    in_range = fwhm_min <= fwhm <= fwhm_max
    r2_valid = r2 >= r2_min
    should_rerun = not (in_range and r2_valid)

    return should_rerun
