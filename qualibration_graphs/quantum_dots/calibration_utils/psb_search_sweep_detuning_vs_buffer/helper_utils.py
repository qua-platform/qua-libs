import numpy as np
import xarray as xr

__all__ = ["validate_and_build_arrays", "assemble_ds_raw"]


def validate_and_build_arrays(node):
    """Validate the 2D sweep settings and build both OPX and plotting axes."""
    buffer_min = int(node.parameters.buffer_duration_min)
    buffer_max = int(node.parameters.buffer_duration_max)
    buffer_step = int(node.parameters.buffer_duration_step)
    if buffer_min % 4 != 0 or buffer_max % 4 != 0 or buffer_step % 4 != 0:
        raise ValueError(
            "Buffer settings must be divisible by 4. Received "
            f"buffer_duration_min={buffer_min}, buffer_duration_max={buffer_max}, "
            f"buffer_duration_step={buffer_step}"
        )

    buffer_ns_array = np.arange(buffer_min, buffer_max, buffer_step, dtype=int)
    buffer_cc_array = (buffer_ns_array // 4).astype(int)
    if len(buffer_ns_array) == 0:
        raise ValueError("Empty buffer sweep: require min < max with a positive step.")

    detuning_array = np.linspace(
        node.parameters.detuning_min,
        node.parameters.detuning_max,
        int(node.parameters.detuning_points),
    )
    return detuning_array, buffer_cc_array, buffer_ns_array


def assemble_ds_raw(dataset: xr.Dataset, pair_names: list[str]) -> xr.Dataset:
    """Convert per-pair fetched streams into the canonical ``ds_raw`` layout."""
    i_arrays = []
    q_arrays = []
    for pair_name in pair_names:
        # The live fetcher uses the stream-buffer order from the QUA program:
        # ``buffer_duration -> detuning -> n_runs``. Transpose here so every
        # downstream consumer sees the same canonical order as the other PSB nodes:
        # ``n_runs -> detuning -> buffer_duration``.
        i_arrays.append(dataset[f"I_{pair_name}"].transpose("n_runs", "detuning", "buffer_duration"))
        q_arrays.append(dataset[f"Q_{pair_name}"].transpose("n_runs", "detuning", "buffer_duration"))

    i_arr = xr.concat(i_arrays, dim="qubit_pair").assign_coords(qubit_pair=pair_names)
    q_arr = xr.concat(q_arrays, dim="qubit_pair").assign_coords(qubit_pair=pair_names)

    return xr.Dataset({"I": i_arr, "Q": q_arr})