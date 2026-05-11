"""Analysis of gate set tomography data.

Raw hardware data are integer counts of outcome ``1`` per sequence over a fixed
shot budget ``num_shots``. These are converted to pyGSTi outcome histograms
``{'0': n0, '1': n1}`` and passed to :class:`pygsti.protocols.GateSetTomography`.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import pygsti
import pygsti.protocols as protocols
import xarray as xr
from pygsti.circuits import Circuit

from calibration_utils.gate_set_tomography.gst_sequences import build_gst_sequences

logger = logging.getLogger(__name__)


def _load_model_pack(model_name: str):
    """Return ``pygsti.modelpacks.<model_name>`` (same naming as sequence generation)."""
    if not model_name.isidentifier():
        raise ValueError(
            f"Invalid GST model name {model_name!r}; expected a model pack identifier (e.g. 'smq1Q_XY')."
        )
    try:
        return importlib.import_module(f"pygsti.modelpacks.{model_name}")
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Unknown GST model pack {model_name!r} (check pygsti.modelpacks)."
        ) from e


def build_pygsti_dataset(
    sequence_strings: list[str],
    counts_one: np.ndarray,
    num_shots: int,
) -> Any:
    """Build a pyGSTi :class:`~pygsti.data.DataSet` from outcome-1 counts.

    Parameters
    ----------
    sequence_strings
        GST circuit strings in the same order as the experiment (from
        :func:`~calibration_utils.gate_set_tomography.build_gst_sequences`).
    counts_one
        One-dimensional array of length ``len(sequence_strings)``. Each entry is
        the number of times outcome ``1`` was observed for that sequence.
    num_shots
        Total repetitions per sequence (same for all sequences).

    Returns
    -------
    pygsti.data.DataSet
        Static dataset with outcome labels ``\"0\"`` and ``\"1\"``.
    """
    if counts_one.ndim != 1:
        raise ValueError(f"counts_one must be 1-D, got shape {counts_one.shape}")
    if len(sequence_strings) != len(counts_one):
        raise ValueError(
            f"sequence_strings ({len(sequence_strings)}) and counts_one ({len(counts_one)}) "
            "must have the same length."
        )

    ds = pygsti.data.DataSet(outcome_labels=("0", "1"))
    for s_str, n1_raw in zip(sequence_strings, counts_one):
        n1 = int(n1_raw)
        if n1 < 0 or n1 > num_shots:
            raise ValueError(
                f"Invalid count {n1} for sequence {s_str!r} with num_shots={num_shots}."
            )
        n0 = num_shots - n1
        c = Circuit(s_str)
        ds.add_count_dict(c, {"0": n0, "1": n1})
    ds.done_adding_data()
    return ds


def _run_gst_single_qubit(
    model_name: str,
    sequence_strings: list[str],
    counts_one: np.ndarray,
    num_shots: int,
    verbosity: int,
) -> dict[str, Any]:
    """Run :class:`pygsti.protocols.GateSetTomography` for one qubit."""
    pack = _load_model_pack(model_name)
    target = pack.target_model() # the model that we want to fit to the data
    initial = target.depolarize(op_noise=0.01, spam_noise=0.01) # initial starting point for the GST fit

    ds = build_pygsti_dataset(sequence_strings, counts_one, num_shots)
    proto_data = protocols.ProtocolData(None, ds)

    gst = protocols.GateSetTomography(
        initial_model=initial,
        verbosity=verbosity,
    )
    protocol_results = gst.run(proto_data)

    report: dict[str, Any] = {
        "success": True,
        "protocol_results": protocol_results,
        "target_model": target,
        "pygsti_dataset": ds,
    }
    try:
        nd = protocol_results.to_nameddict()
        report["nameddict_summary"] = nd
    except Exception as exc:  # noqa: BLE001 — best-effort serialization helper
        logger.debug("Could not convert protocol results to NamedDict: %s", exc)
    return report


def analyse_raw_data(
    ds_raw: xr.Dataset,
    qubits: list[Any],
    *,
    model_name: str,
    lengths: list[int],
    num_shots: int,
    verbosity: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run pyGSTi gate set tomography on raw QUAlibrate data.

    Parameters
    ----------
    ds_raw
        Dataset with coordinate ``sequence`` and variables ``state_<qubit_name>``
        holding integer counts of outcome ``1`` per sequence index.
    qubits
        Qubit objects (each must have a ``.name`` attribute).
    model_name
        pyGSTi model pack name (e.g. ``\"smq1Q_XY\"``), matching ``Parameters.model``.
    lengths
        Same length list passed to :func:`~calibration_utils.gate_set_tomography.build_gst_sequences`
        when the experiment was built (``Parameters.get_lengths()``).
    num_shots
        Shots per sequence (``Parameters.num_shots``).
    verbosity
        pyGSTi log verbosity (0 is quiet).

    Returns
    -------
    dict
        ``{qubit_name: report}`` where ``report`` contains at least ``success``,
        ``protocol_results`` (pyGSTi object), ``target_model``, ``pygsti_dataset``,
        and optionally ``nameddict_summary``.
    """
    sequence_strings = build_gst_sequences(model_name, lengths)
    n_seq = len(sequence_strings)
    results: dict[str, dict[str, Any]] = {}

    for qubit in qubits:
        qname = qubit.name
        var_name = f"state_{qname}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No state variable for qubit %s — skipping.", qname)
            results[qname] = {"success": False, "error": f"missing {var_name}"}
            continue

        state_data = np.asarray(ds_raw[var_name].values).ravel()
        if state_data.size != n_seq:
            msg = (
                f"Length mismatch for {qname}: expected {n_seq} sequences (from model/lengths), "
                f"got {state_data.size} values in {var_name}."
            )
            logger.error(msg)
            results[qname] = {"success": False, "error": msg}
            continue

        try:
            results[qname] = _run_gst_single_qubit(
                model_name=model_name,
                sequence_strings=sequence_strings,
                counts_one=state_data,
                num_shots=num_shots,
                verbosity=verbosity,
            )
        except Exception as exc:
            logger.exception("GST analysis failed for qubit %s", qname)
            results[qname] = {
                "success": False,
                "error": str(exc),
            }

    return results


def log_gst_results(
    gst_results: dict[str, dict[str, Any]],
    node_logger: Any | None = None,
) -> None:
    """Log GST analysis status per qubit."""
    _log = node_logger or logger.info
    for qname, r in sorted(gst_results.items()):
        if not r.get("success", False):
            _log(f"  {qname}: FAILED — {r.get('error', 'unknown error')}")
            continue
        _log(f"  {qname}: GST finished (see protocol_results / nameddict_summary).")
