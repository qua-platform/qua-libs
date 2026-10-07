import numpy as np
import xarray as xr

STATES = ("ground", "excited")
"""The two qubit states scanned: |g> (as-is after reset) and |e> (x180 before the DRACHMA pulse)."""

# Streams saved per (state, qubit): the combined quadratures and their per-shot second moments
# (the latter give Var(I), Var(Q) for the error bars). Shared by the node's stream_processing,
# fetch_round_traces and process_raw_dataset so the key list only changes in one place.
TRACE_KEYS = ("I", "Q", "I_sq", "Q_sq")

NOOP_SUFFIX = "noop"
"""Stream/variable suffix of the single no-operation reference: the probe measured right after reset with no
DRACHMA pulse. Saved as "<key>_noop" with only a qubit dimension and shared by both states."""


def fetch_round_traces(job, qubits, states: tuple) -> dict[str, np.ndarray]:
    """Fetch each (state, qubit) per-round I/Q average and second moments straight from
    job.result_handles and stack across qubits -> {"<key>_<state>": array(qubit, round)}.

    The no-operation reference is fetched too -> {"<key>_noop": array(qubit)}."""
    traces: dict[str, np.ndarray] = {}
    for state in states:
        for key in TRACE_KEYS:
            stacked = [
                np.asarray(job.result_handles.get(f"{key}_{state}{i + 1}").fetch_all()) for i in range(len(qubits))
            ]
            traces[f"{key}_{state}"] = np.stack(stacked, axis=0)
    for key in TRACE_KEYS:
        stacked = [
            np.asarray(job.result_handles.get(f"{key}_{NOOP_SUFFIX}{i + 1}").fetch_all()) for i in range(len(qubits))
        ]
        traces[f"{key}_{NOOP_SUFFIX}"] = np.stack(stacked, axis=0)
    return traces


def process_raw_dataset(ds: xr.Dataset, probe_length: int, num_shots: int) -> xr.Dataset:
    """Convert the probe's integrated I/Q to volts, stack the per-state streams into a "state" dimension
    and compute the residual probe power I^2 + Q^2 vs scan point (per state) with its standard error.

    The shot-noise bias (Var(I) + Var(Q)) / num_shots is subtracted from the power, so a fully depleted
    resonator gives ~0 instead of a positive noise floor. Volts conversion: 2x single-demod
    correction, 2**12 fixed-point scale, divided by the probe length; I_sq/Q_sq get the squared factor.
    """
    factor = 2 * 2**12 / probe_length
    scaled = {}
    for key in TRACE_KEYS:
        f = factor**2 if key.endswith("_sq") else factor
        for s in (*STATES, NOOP_SUFFIX):
            scaled[f"{key}_{s}"] = ds[f"{key}_{s}"] * f
    ds = ds.assign(scaled)

    state_coord = xr.DataArray(list(STATES), dims="state", name="state")
    ds = ds.assign({key: xr.concat([ds[f"{key}_{s}"] for s in STATES], dim=state_coord) for key in TRACE_KEYS})

    I, Q = ds["I"], ds["Q"]
    var_I = (ds["I_sq"] - I**2).clip(min=0)
    var_Q = (ds["Q_sq"] - Q**2).clip(min=0)
    ds = ds.assign(
        var_I=var_I,
        var_Q=var_Q,
        power=I**2 + Q**2 - (var_I + var_Q) / num_shots,
        # Delta method for P = I^2 + Q^2 (Cov(I,Q) dropped, as in 23c).
        power_err=2 * np.sqrt((I**2 * var_I + Q**2 * var_Q) / num_shots),
    )
    ds.power.attrs = {"long_name": "residual probe power", "units": "V^2"}
    ds.power_err.attrs = {"long_name": "residual probe power standard error", "units": "V^2"}

    # Test vs the single no-operation reference at every (state, point): chi^2 of the (I, Q) mean
    # difference with 2 degrees of freedom, p = exp(-chi2 / 2) in closed form. p > alpha -> the probe
    # field is indistinguishable from the empty-resonator reference -> depleted.
    I_noop, Q_noop = ds[f"I_{NOOP_SUFFIX}"], ds[f"Q_{NOOP_SUFFIX}"]
    var_I_noop = (ds[f"I_sq_{NOOP_SUFFIX}"] - I_noop**2).clip(min=0)
    var_Q_noop = (ds[f"Q_sq_{NOOP_SUFFIX}"] - Q_noop**2).clip(min=0)
    chi2 = num_shots * ((I - I_noop) ** 2 / (var_I + var_I_noop) + (Q - Q_noop) ** 2 / (var_Q + var_Q_noop))
    ds = ds.assign(p_value_noop=np.exp(-chi2 / 2))
    ds.p_value_noop.attrs = {"long_name": "test-vs-no-operation z-test p-value", "units": ""}
    return ds
