"""QUA helpers and post-processing for joint-outcome parity streams.

Provides:
- declare_parity_streams  – declare QUA variables/streams inside a program() block
- save_parity_measurement – save one shot's parity outcome to the right stream
- buffer_parity_streams   – wire up stream_processing() for one item
- process_joint_streams   – post-process averaged joint-outcome datasets
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import xarray as xr


def declare_parity_streams(node, items, stream_fn: Optional[Callable] = None):
    """Declare QUA variables and streams for parity readout.

    Call inside a ``with program()`` block before the main loop.

    Args:
        node: QualibrationNode whose ``parameters.parity_pre_measurement`` flag
            controls which streams are created.
        items: Iterable of qubits or qubit-pairs; each must have a ``.name``.
        stream_fn: QUA stream constructor to use (default: ``declare_stream``).
            Pass ``declare_output_stream`` when the surrounding program uses
            output streams throughout.

    Returns:
        ``(p2, p1, streams)`` where

        * ``p2`` – QUA ``int`` variable for the post-sequence measurement.
        * ``p1`` – QUA ``int`` variable for the pre-sequence measurement, or
          ``None`` when ``parity_pre_measurement`` is ``False``.
        * ``streams`` – dict mapping stream-key → ``{item.name: stream}``.
          Keys are ``"p0_p0"``, ``"p0_p1"``, ``"p1_p0"``, ``"p1_p1"`` when
          ``parity_pre_measurement`` is ``True``, or ``"p"`` otherwise.
    """
    from qm.qua import declare, declare_stream

    if stream_fn is None:
        stream_fn = declare_stream

    p2 = declare(int)

    if node.parameters.parity_pre_measurement:
        p1 = declare(int)
        streams = {
            "p0_p0": {item.name: stream_fn() for item in items},
            "p0_p1": {item.name: stream_fn() for item in items},
            "p1_p0": {item.name: stream_fn() for item in items},
            "p1_p1": {item.name: stream_fn() for item in items},
        }
    else:
        p1 = None
        streams = {"p": {item.name: stream_fn() for item in items}}

    return p2, p1, streams


def save_parity_measurement(node, name: str, p1, p2, streams: dict) -> None:
    """Save one shot's parity outcome to the appropriate QUA streams.

    Call inside the inner measurement loop, once per item per shot.

    Args:
        node: QualibrationNode with the ``parity_pre_measurement`` flag.
        name: The ``.name`` of the qubit or qubit-pair being measured.
        p1: QUA variable holding the pre-sequence measurement result (ignored
            when ``parity_pre_measurement`` is ``False``).
        p2: QUA variable holding the post-sequence measurement result.
        streams: The ``streams`` dict returned by :func:`declare_parity_streams`.
    """
    from qm.qua import else_, if_, save

    if node.parameters.parity_pre_measurement:
        with if_(p1 == 0):
            with if_(p2 == 0):
                save(1, streams["p0_p0"][name])
                save(0, streams["p0_p1"][name])
                save(0, streams["p1_p0"][name])
                save(0, streams["p1_p1"][name])
            with else_():
                save(0, streams["p0_p0"][name])
                save(1, streams["p0_p1"][name])
                save(0, streams["p1_p0"][name])
                save(0, streams["p1_p1"][name])
        with else_():
            with if_(p2 == 0):
                save(0, streams["p0_p0"][name])
                save(0, streams["p0_p1"][name])
                save(1, streams["p1_p0"][name])
                save(0, streams["p1_p1"][name])
            with else_():
                save(0, streams["p0_p0"][name])
                save(0, streams["p0_p1"][name])
                save(0, streams["p1_p0"][name])
                save(1, streams["p1_p1"][name])
    else:
        save(p2, streams["p"][name])


def buffer_parity_streams(node, name: str, streams: dict, *buffer_dims: int) -> None:
    """Buffer and save parity streams inside a ``stream_processing()`` block.

    Args:
        node: QualibrationNode with the ``parity_pre_measurement`` flag.
        name: The ``.name`` of the qubit or qubit-pair.
        streams: The ``streams`` dict returned by :func:`declare_parity_streams`.
        *buffer_dims: Dimension(s) passed to ``.buffer()``.
    """
    if node.parameters.parity_pre_measurement:
        for key in ("p0_p0", "p0_p1", "p1_p0", "p1_p1"):
            streams[key][name].buffer(*buffer_dims).average().save(f"{key}_{name}")
    else:
        streams["p"][name].buffer(*buffer_dims).average().save(f"p_{name}")


def _name_without_trailing_digits(name: str) -> str:
    stripped = name.rstrip("0123456789")
    return stripped if stripped != name and stripped else name


def _candidate_stream_names(prefix: str, item_name: str) -> list[str]:
    """Return possible variable names for a stream and item.

    ``XarrayDataFetcher`` may stack handles ending in digits.  For example,
    QUA handle ``p_q1`` can become dataset variable ``p_q(qubit, ...)``.
    Per-item handles such as ``p0_p0_q1`` remain unstacked because their
    prefix contains digits.
    """
    candidates = [f"{prefix}_{item_name}"]
    item_base = _name_without_trailing_digits(item_name)
    if item_base != item_name:
        candidates.append(f"{prefix}_{item_base}")
    candidates.append(prefix)
    return list(dict.fromkeys(candidates))


def _select_item(
    da: xr.DataArray,
    item_name: str,
    item_names: Sequence[str],
    item_dim: str,
) -> xr.DataArray:
    if item_dim not in da.dims:
        return da

    coord_values = da.coords.get(item_dim)
    if coord_values is not None and item_name in set(coord_values.values.tolist()):
        return da.sel({item_dim: item_name}, drop=True)

    if da.sizes[item_dim] == 1:
        return da.isel({item_dim: 0}, drop=True)

    item_index = item_names.index(item_name)
    if item_index >= da.sizes[item_dim]:
        raise ValueError(
            f"Cannot select {item_name!r} from {da.name!r}: "
            f"dimension {item_dim!r} has size {da.sizes[item_dim]}."
        )
    return da.isel({item_dim: item_index}, drop=True)


def _normalize_stream_dataarray(
    da: xr.DataArray,
    *,
    item_name: str,
    item_names: Sequence[str],
    item_dim: str,
    sweep_dims: Optional[Sequence[str]],
) -> xr.DataArray:
    da = _select_item(da, item_name, item_names, item_dim)

    if sweep_dims is not None:
        missing_dims = [dim for dim in sweep_dims if dim not in da.dims]
        if missing_dims:
            raise ValueError(
                f"{da.name!r} for {item_name!r} is missing sweep dimension(s) "
                f"{missing_dims}; dims are {da.dims}."
            )

        for dim in list(da.dims):
            if dim in sweep_dims:
                continue
            if da.sizes[dim] != 1:
                raise ValueError(
                    f"{da.name!r} for {item_name!r} has unexpected non-singleton "
                    f"dimension {dim!r} with size {da.sizes[dim]}."
                )
            da = da.isel({dim: 0}, drop=True)

        da = da.transpose(*sweep_dims)

    return da


def _stream_dataarray_for_item(
    ds: xr.Dataset,
    prefix: str,
    item_name: str,
    item_names: Sequence[str],
    *,
    item_dim: str,
    sweep_dims: Optional[Sequence[str]],
) -> xr.DataArray:
    for candidate in _candidate_stream_names(prefix, item_name):
        if candidate not in ds.data_vars:
            continue
        return _normalize_stream_dataarray(
            ds[candidate],
            item_name=item_name,
            item_names=item_names,
            item_dim=item_dim,
            sweep_dims=sweep_dims,
        )

    raise KeyError(
        f"Missing parity stream for {item_name!r} with prefix {prefix!r}. "
        f"Tried {_candidate_stream_names(prefix, item_name)}."
    )


def get_parity_item_names(
    ds: xr.Dataset,
    analysis_signal: str = "E_p2_given_p1_0",
    *,
    item_names: Optional[Iterable[str]] = None,
    item_dim: str = "qubit",
    legacy_prefixes: Sequence[str] = ("p0_p0", "p"),
) -> list[str]:
    """Resolve item names from processed parity variables.

    Prefer the canonical ``{analysis_signal}_{item}`` variables produced by
    :func:`process_parity_streams`.  Raw joint streams and single-shot streams
    are used only as compatibility fallbacks.
    """
    fallback_names = list(item_names or [])
    prefixes = (analysis_signal, *legacy_prefixes)

    if fallback_names:
        for prefix in prefixes:
            names = []
            for name in fallback_names:
                try:
                    _stream_dataarray_for_item(
                        ds,
                        prefix,
                        name,
                        fallback_names,
                        item_dim=item_dim,
                        sweep_dims=None,
                    )
                except (KeyError, ValueError):
                    continue
                names.append(name)
            if names:
                return names
        return fallback_names

    for prefix in prefixes:
        full_prefix = f"{prefix}_"
        names = [
            var_name.replace(full_prefix, "", 1)
            for var_name in sorted(ds.data_vars)
            if var_name.startswith(full_prefix) and not var_name.endswith("_fit")
        ]
        if names:
            return names

    return []


def process_joint_streams(
    ds: xr.Dataset,
    qubit_names: List[str],
    *,
    item_dim: str = "qubit",
    sweep_dims: Optional[Sequence[str]] = None,
) -> xr.Dataset:
    """Compute conditional expectations from joint-outcome streams.

    For each qubit (or qubit pair) name, reads the four averaged joint-outcome
    variables and adds two conditional expectations to the dataset:

    - ``E_p2_given_p1_0_{name}`` = P(second=1 | first=0) = p0_p1 / (p0_p0 + p0_p1)
    - ``E_p2_given_p1_1_{name}`` = P(second=1 | first=1) = p1_p1 / (p1_p0 + p1_p1)

    Division by zero yields NaN.

    Args:
        ds: Dataset containing ``p0_p0_{name}``, ``p0_p1_{name}``,
            ``p1_p0_{name}``, ``p1_p1_{name}`` for each name.
        qubit_names: List of qubit (or qubit-pair) names.

    Returns:
        Dataset with the two conditional expectation variables added.
    """
    new_vars = {}
    for name in qubit_names:
        p0_p0_da = _stream_dataarray_for_item(
            ds, "p0_p0", name, qubit_names, item_dim=item_dim, sweep_dims=sweep_dims
        )
        p0_p1_da = _stream_dataarray_for_item(
            ds, "p0_p1", name, qubit_names, item_dim=item_dim, sweep_dims=sweep_dims
        )
        p1_p0_da = _stream_dataarray_for_item(
            ds, "p1_p0", name, qubit_names, item_dim=item_dim, sweep_dims=sweep_dims
        )
        p1_p1_da = _stream_dataarray_for_item(
            ds, "p1_p1", name, qubit_names, item_dim=item_dim, sweep_dims=sweep_dims
        )

        p0_p0 = p0_p0_da.values.astype(np.float64)
        p0_p1 = p0_p1_da.values.astype(np.float64)
        p1_p0 = p1_p0_da.values.astype(np.float64)
        p1_p1 = p1_p1_da.values.astype(np.float64)

        denom_0 = p0_p0 + p0_p1
        denom_1 = p1_p0 + p1_p1

        e_given_0 = np.where(denom_0 > 0, p0_p1 / denom_0, np.nan)
        e_given_1 = np.where(denom_1 > 0, p1_p1 / denom_1, np.nan)

        template = p0_p0_da
        new_vars[f"E_p2_given_p1_0_{name}"] = template.copy(data=e_given_0)
        new_vars[f"E_p2_given_p1_1_{name}"] = template.copy(data=e_given_1)

    return ds.assign(new_vars)


def process_parity_streams(
    ds: xr.Dataset,
    item_names: Iterable[str],
    *,
    parity_pre_measurement: bool,
    item_dim: str = "qubit",
    sweep_dims: Optional[Sequence[str]] = None,
) -> xr.Dataset:
    """Normalize parity streams into conditional-expectation variables.

    The returned dataset always contains ``E_p2_given_p1_0_<item>`` and
    ``E_p2_given_p1_1_<item>`` with only sweep dimensions.  When no parity
    pre-measurement was acquired, the single post-measurement stream is copied
    into both conditional variables as the one available measurement branch.
    """
    item_names = list(item_names)
    if parity_pre_measurement:
        existing_vars = {}
        try:
            for name in item_names:
                for prefix in ("E_p2_given_p1_0", "E_p2_given_p1_1"):
                    da = _stream_dataarray_for_item(
                        ds,
                        prefix,
                        name,
                        item_names,
                        item_dim=item_dim,
                        sweep_dims=sweep_dims,
                    ).astype(np.float64)
                    existing_vars[f"{prefix}_{name}"] = da.copy()
        except KeyError:
            existing_vars = {}
        if existing_vars:
            return ds.assign(existing_vars)

        return process_joint_streams(
            ds,
            item_names,
            item_dim=item_dim,
            sweep_dims=sweep_dims,
        )

    new_vars = {}
    for name in item_names:
        p2 = _stream_dataarray_for_item(
            ds, "p", name, item_names, item_dim=item_dim, sweep_dims=sweep_dims
        ).astype(np.float64)
        new_vars[f"E_p2_given_p1_0_{name}"] = p2.copy()
        new_vars[f"E_p2_given_p1_1_{name}"] = p2.copy()

    return ds.assign(new_vars)
