import numpy as np
import xarray as xr

__all__ = ["stream_var_name", "split_condition_maps", "analyze_reset_pulse_maps"]


def stream_var_name(ds_raw: xr.Dataset, prefix: str, qubit_name: str) -> str:
    """Resolve fetched variable names when qubit numeric suffix is normalized."""
    candidates = (
        f"{prefix}_{qubit_name}",
        f"{prefix}_{qubit_name.rstrip('0123456789')}",
    )
    for candidate in candidates:
        if candidate in ds_raw:
            return candidate
    raise KeyError(
        f"Could not find stream variable for prefix='{prefix}', qubit='{qubit_name}'. "
        f"Tried: {candidates}"
    )


def split_condition_maps(
    ds_raw: xr.Dataset,
    prefix: str,
    qubit_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D maps for [no_op_pulse, with_op_pulse].

    Reads per-condition streams saved as ``{prefix}_{qubit_name}_{cond}``
    (condition 0 = no-op, condition 1 = with-op).
    """
    def _get(cond: int) -> np.ndarray:
        for qn in (qubit_name, qubit_name.rstrip("0123456789")):
            key = f"{prefix}_{qn}_{cond}"
            if key in ds_raw:
                return np.asarray(ds_raw[key].values).squeeze()
        raise KeyError(
            f"Could not find stream variable for prefix='{prefix}', "
            f"qubit='{qubit_name}', condition={cond}."
        )

    map_no = _get(0)
    map_yes = _get(1)

    n_freq = len(ds_raw["frequency_detuning"])
    n_amp = len(ds_raw["amplitude_scale"])
    if map_no.shape == (n_amp, n_freq):
        map_no = map_no.T
        map_yes = map_yes.T
    elif map_no.shape != (n_freq, n_amp):
        map_no = map_no.reshape(n_freq, n_amp)
        map_yes = map_yes.reshape(n_freq, n_amp)
    return map_no, map_yes


def analyze_reset_pulse_maps(
    ds_raw: xr.Dataset,
    qubit_names: list[str],
) -> tuple[dict, dict, dict]:
    """Return fit_results, optimal_points and map_results."""
    fit_results = {}
    optimal_points = {}
    map_results = {}

    detunings = ds_raw["frequency_detuning"].values
    amplitudes = ds_raw["amplitude_scale"].values

    for q_name in qubit_names:
        init_state_no, init_state_yes = split_condition_maps(ds_raw, "init_state", q_name)
        init_i_no, init_i_yes = split_condition_maps(ds_raw, "init_I", q_name)

        state_no, state_yes = split_condition_maps(ds_raw, "state", q_name)
        i_no, i_yes = split_condition_maps(ds_raw, "I", q_name)
        state_diff = state_yes - state_no
        i_diff = i_yes - i_no

        state_flat = int(np.argmax(np.abs(state_diff)))
        state_df_idx, state_amp_idx = np.unravel_index(state_flat, state_diff.shape)
        i_flat = int(np.argmax(np.abs(i_diff)))
        i_df_idx, i_amp_idx = np.unravel_index(i_flat, i_diff.shape)

        fit_results[q_name] = {
            "success": True,
            "optimal_frequency_detuning_hz": int(detunings[state_df_idx]),
            "optimal_amplitude_scale": float(amplitudes[state_amp_idx]),
            "max_abs_state_diff": float(np.abs(state_diff[state_df_idx, state_amp_idx])),
            "state_diff_at_optimum": float(state_diff[state_df_idx, state_amp_idx]),
            "max_abs_i_diff": float(np.abs(i_diff[i_df_idx, i_amp_idx])),
            "i_diff_at_optimum": float(i_diff[i_df_idx, i_amp_idx]),
            "i_optimal_frequency_detuning_hz": int(detunings[i_df_idx]),
            "i_optimal_amplitude_scale": float(amplitudes[i_amp_idx]),
        }
        optimal_points[q_name] = {
            "frequency_detuning_hz": int(detunings[state_df_idx]),
            "amplitude_scale": float(amplitudes[state_amp_idx]),
            "max_abs_state_diff": float(np.abs(state_diff[state_df_idx, state_amp_idx])),
        }
        map_results[q_name] = {
            "init_state_no": init_state_no,
            "init_state_yes": init_state_yes,
            "init_state_avg": 0.5 * (init_state_no + init_state_yes),
            "init_i_no": init_i_no,
            "init_i_yes": init_i_yes,
            "init_i_avg": 0.5 * (init_i_no + init_i_yes),
            "state_no": state_no,
            "state_yes": state_yes,
            "state_diff": state_diff,
            "i_no": i_no,
            "i_yes": i_yes,
            "i_diff": i_diff,
        }

    return fit_results, optimal_points, map_results
