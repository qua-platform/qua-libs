"""Analysis module for two-qubit readout confusion matrix calibration."""

# pylint: disable=duplicate-code

from typing import Callable, Dict, Optional

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process the raw dataset for confusion matrix analysis.

    If the dataset has state_control and state_target only, derives state = state_control * 2 + state_target.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from the experiment
    node : QualibrationNode
        The calibration node containing qubit pairs information

    Returns
    -------
    xr.Dataset
        Processed dataset suitable for compute_confusion_matrices
    """
    if "state" in ds.data_vars:
        return ds

    if "state_control" in ds.data_vars and "state_target" in ds.data_vars:
        return ds.assign(state=ds.state_control * 2 + ds.state_target)

    return ds


def is_confusion_matrix_valid(conf: np.ndarray, col_sum_tol: float = 0.05) -> bool:
    """Return True if ``conf`` is a finite 4x4 matrix with column sums near unity.

    Expects ``conf[measured, prepared]``: each column (fixed prepared state) sums to ~1.
    """
    conf = np.asarray(conf)
    if conf.shape != (4, 4) or not np.all(np.isfinite(conf)):
        return False
    col_sums = conf.sum(axis=0)
    return bool(np.all(np.abs(col_sums - 1.0) <= col_sum_tol))


def compute_confusion_matrices(
    ds: xr.Dataset,
    node: QualibrationNode,
    log_callable: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute 4x4 confusion matrices from the measurement dataset.

    Counts joint readout outcomes for each prepared control/target state and
    normalizes by ``num_shots``. The returned matrix is ``conf[measured, prepared]``:
    row index = measured |00⟩,|01⟩,|10⟩,|11⟩ (``state = control*2 + target``);
    column index = prepared state in the same order.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with ``state`` and preparation coordinates.
    node : QualibrationNode
        The calibration node.
    log_callable : callable, optional
        Logger (defaults to ``node.log`` when available).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from qubit pair name to 4x4 confusion matrix ``conf[measured, prepared]``.

    Raises
    ------
    ValueError
        If required dataset variables or coordinates are missing.
    KeyError
        If a qubit pair or preparation coordinate cannot be selected.
    """
    if log_callable is None:
        log_callable = getattr(node, "log", None)

    if "state" not in ds.data_vars:
        raise ValueError("Dataset must contain 'state' after processing.")

    qubit_pairs = node.namespace["qubit_pairs"]
    num_shots = node.parameters.num_shots
    pair_dim = "qubit_pair" if "qubit_pair" in ds.dims else "qubit"
    shot_dim = "n" if "n" in ds.dims else "N"

    for coord in (pair_dim, shot_dim, "init_state_control", "init_state_target"):
        if coord not in ds.dims and coord not in ds.coords:
            raise ValueError(f"Dataset missing required dimension/coordinate '{coord}'.")

    states = [0, 1, 2, 3]
    confusions = {}
    for qp in qubit_pairs:
        conf = []
        for measured_state in states:
            row = []
            for init_control in [0, 1]:
                for init_target in [0, 1]:
                    sel = ds.sel(
                        **{
                            pair_dim: qp.name,
                            "init_state_control": init_control,
                            "init_state_target": init_target,
                        }
                    )
                    count = (sel.state == measured_state).sum(dim=shot_dim).values
                    row.append(float(count))
            conf.append(row)
        confusions[qp.name] = np.array(conf) / num_shots
        if log_callable is not None and not is_confusion_matrix_valid(confusions[qp.name]):
            log_callable(
                f"Pair {qp.name}: confusion matrix failed validation "
                f"(column sums = {confusions[qp.name].sum(axis=0)})."
            )
    return confusions
