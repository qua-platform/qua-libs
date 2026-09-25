
from qualibrate.core import GraphParameters, QualibrationGraph, QualibrationLibrary


# --- resolve_params callbacks ---
# Each function is called once per qubit after its source node finishes.
# It receives the source node instance and the qubit name, and returns a
# dict of parameter overrides that are applied to the destination node.


def _rabi_retry_params(rabi_node, qubit):
    """Narrow the amplitude sweep on each Rabi retry.

    The first Rabi attempt sweeps a broad amplitude
    range. If it finds a rough pi-amplitude estimate (fit succeeded), the
    next attempt can zoom in around that value — converging faster and with
    higher SNR. If the fit failed entirely, widening the range slightly
    avoids getting stuck in a local window.
    """
    fit = rabi_node.results.get("fit_results", {}).get(qubit, {})
    if fit.get("success"):
        # Center the next sweep around 2× the found pi amplitude and
        # add more points for a better fit on the narrower window.
        return {
            "max_amplitude": fit["pi_amplitude"] * 2.2,
            "num_points": 60,
        }
    # No fit: try a slightly wider range to catch a shifted resonance.
    return {"max_amplitude": rabi_node.parameters.max_amplitude * 1.3}


def _refined_rabi_params(rabi_node, qubit):
    """Pass the coarse pi amplitude to the refined scan as a search center.

    The coarse Rabi gives a rough estimate of the
    pi-pulse amplitude. The refined scan only needs to cover a narrow window
    around that value. Without resolve_params this handoff would require
    manual post-processing between nodes.
    """
    fit = rabi_node.results["fit_results"][qubit]
    return {
        "center_amplitude": fit["pi_amplitude"],
        # Scan ±20 % around the coarse estimate for a tight, high-precision fit.
        "amplitude_span": fit["pi_amplitude"] * 0.4,
    }


def _t1_diagnostic_params(rabi_node, qubit):
    """Set T1 max_delay based on why the Rabi calibration failed.

    If Rabi oscillations never appeared, the qubit
    may have a very short T1 (fast relaxation washes out the signal). Starting
    the T1 scan with a compressed delay range gets a faster diagnosis.
    If the fit partially succeeded but the node still failed for another
    reason, keep the standard range to avoid missing a longer T1.
    """
    fit = rabi_node.results.get("fit_results", {}).get(qubit, {})
    if fit.get("success"):
        # Partial success: T1 is likely in the normal range.
        return {"max_delay": 50e-6}
    # No fit at all: suspect short T1 — compressed range for quick diagnosis.
    return {"max_delay": 20e-6}

