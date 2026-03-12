"""Helpers for parity-difference datasets across fetch conventions."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import xarray as xr


def get_qubit_names_from_pdiff(ds: xr.Dataset, fallback_qubits: Iterable | None = None) -> List[str]:
    """Resolve qubit names from grouped or per-variable parity-diff datasets."""
    if "pdiff" in ds.data_vars and "qubit" in ds["pdiff"].dims:
        return [str(name) for name in np.asarray(ds["pdiff"].coords["qubit"].values)]

    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_") and not v.endswith("_fit")]
    if pdiff_vars:
        return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]

    if fallback_qubits is None:
        return []
    return [getattr(q, "name", f"Q{i}") for i, q in enumerate(fallback_qubits)]


def get_pdiff_trace(ds: xr.Dataset, qname: str) -> np.ndarray | None:
    """Return 1D/2D parity-difference trace for one qubit name if available."""
    if "pdiff" in ds.data_vars and "qubit" in ds["pdiff"].dims:
        qubit_coords = [str(name) for name in np.asarray(ds["pdiff"].coords["qubit"].values)]
        if qname not in qubit_coords:
            return None
        return np.asarray(ds["pdiff"].sel(qubit=qname).values, dtype=float)

    pdiff_var = f"pdiff_{qname}"
    if pdiff_var not in ds.data_vars:
        return None
    return np.asarray(ds[pdiff_var].values, dtype=float)
