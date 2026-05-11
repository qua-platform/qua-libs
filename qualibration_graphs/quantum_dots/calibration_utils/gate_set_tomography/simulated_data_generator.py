"""Synthetic GST datasets via pyGSTi ``simulate_data`` (no OPX / hardware)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pygsti
import xarray as xr
from pygsti.circuits import Circuit

from calibration_utils.common_utils.experiment import get_qubits
from calibration_utils.gate_set_tomography.analysis import _load_model_pack
from calibration_utils.gate_set_tomography.gst_sequences import build_gst_sequences

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode

    from calibration_utils.gate_set_tomography.parameters import Parameters
    from quam_config import Quam


def generate_simulated_dataset(node: "QualibrationNode[Parameters, Quam]") -> xr.Dataset:
    """Generate GST-like counts using ``pygsti.data.simulate_data`` on a noisy target model.

    For each qubit, draws ``num_shots`` Bernoulli samples per GST circuit so that downstream
    :func:`~calibration_utils.gate_set_tomography.analyse_raw_data` receives the same
    ``state_<qubit>`` variables as after hardware execution.

    The simulation uses the model pack's ``target_model()`` with small depolarizing noise on
    gates and SPAM (aligned with the GST analysis starting model in ``analysis.py``).
    """
    qubits = get_qubits(node)
    node.namespace["qubits"] = qubits

    model_name = node.parameters.model
    lengths = node.parameters.get_lengths()
    num_shots = node.parameters.num_shots

    sequence_strings = build_gst_sequences(model_name, lengths)
    n_seq = len(sequence_strings)
    circuits = [Circuit(s) for s in sequence_strings]

    pack = _load_model_pack(model_name)
    target = pack.target_model()
    sim_model = target.depolarize(op_noise=0.01, spam_noise=0.01)

    seed_base = getattr(node.parameters, "random_seed", None)
    if seed_base is None:
        seed_base = getattr(node.parameters, "seed", None)

    data_vars: dict[str, Any] = {}
    for i, qubit in enumerate(qubits):
        rng_seed = None
        if seed_base is not None:
            rng_seed = (int(seed_base) + i * 100_003) % (2**31)

        sim_ds = pygsti.data.simulate_data(
            sim_model,
            circuits,
            num_samples=num_shots,
            seed=rng_seed,
        )
        counts_one = np.zeros(n_seq, dtype=np.int64)
        for j, c in enumerate(circuits):
            counts_one[j] = int(sim_ds[c]["1"])
        data_vars[f"state_{qubit.name}"] = (["sequence"], counts_one)

    ds = xr.Dataset(
        data_vars,
        coords={
            "sequence": xr.DataArray(
                np.arange(n_seq),
                attrs={"long_name": "sequence index"},
            ),
        },
    )
    return ds
