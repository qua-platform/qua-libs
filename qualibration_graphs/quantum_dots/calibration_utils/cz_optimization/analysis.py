"""Analysis utilities for the CZ gate optimisation node."""


def summarise_cz_optimization(
    fit_results: dict,
    measurement_streams: dict,
) -> dict:
    """Enrich fit_results with CZ-specific best-score and measurement data.

    Parameters
    ----------
    fit_results : dict
        Per-pair summary from analyse_optimization().
    measurement_streams : dict
        Per-pair measurement stream data containing circuit_history.

    Returns
    -------
    dict
        The fit_results dict, mutated in-place with added keys:
        - "best_score": highest score across all generations
        - "best_measurements": circuit probabilities at the running-best point
    """
    for qp_name, summary in fit_results.items():
        streams = measurement_streams.get(qp_name, {})
        circuit_hist = streams.get("circuit_history", {})
        if circuit_hist:
            summary["best_score"] = (
                max(circuit_hist["score"]) if circuit_hist["score"] else 0.0
            )
            running_best = circuit_hist.get("running_best_measurements", [])
            if running_best:
                summary["best_measurements"] = running_best[-1]

    return fit_results
