from qualibrate import QualibrationNode, QualibrationGraph

def should_repeat_ramsey(graph: QualibrationGraph,node: QualibrationNode, target: str) -> bool:
    """Retry until T2* crosses the target threshold."""
    fit = node.results.get("fit_results", {}).get(target, {})
    t2_star = fit.get("t2_star")

    # Retry if fit failed or T2* is below target
    if not fit.get("success") or t2_star is None:
        return True

    t2_us = t2_star / 1e-6
    target_us = graph.parameters.target_t2_star_us
    if t2_us < target_us:
        return True

    return False  # Target met, stop looping


# dummy function (the rabi retry but the numbers make no sense, it's just there to pass some v alue)
def ramsey_retry_params(graph: QualibrationGraph,node: QualibrationNode, target: str):
    fit = node.results.get("fit_results", {}).get(target, {})
    if fit.get("success"):
        # Narrow the sweep around the fitted pi amplitude
        return {"max_amplitude": fit["pi_amplitude"] * 2.2, "num_points": 60}
    # No fit yet — widen the range slightly
    return {"max_amplitude": node.parameters.max_amplitude * 1.3}
