import inspect
from pathlib import Path
from typing import Optional, Union
import warnings

from qm.qua import *
from qm.qua._dsl import _ResultSource
from qm.qua._expressions import QuaVariable

from quam_config import QuAM

__all__ = [
    "qua_declaration",
    "node_save",
]


def qua_declaration(
    num_qubits,
) -> tuple[list[QuaVariable], list[_ResultSource], list[QuaVariable], list[_ResultSource], QuaVariable, _ResultSource]:
    """
    Macro to declare the necessary QUA variables

    :param num_qubits: Number of qubits used in this experiment
    :return:
    """
    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(num_qubits)]
    Q = [declare(fixed) for _ in range(num_qubits)]
    I_st = [declare_stream() for _ in range(num_qubits)]
    Q_st = [declare_stream() for _ in range(num_qubits)]
    return I, I_st, Q, Q_st, n, n_st

def node_save(
    quam: QuAM,
    name: str,
    data: dict,
    additional_files: Optional[Union[dict, bool]] = None,
):
    # Save results
    if isinstance(additional_files, dict):
        quam.data_handler.additional_files = additional_files
    elif additional_files is True:
        files = ["../calibration_db.json", "optimal_weights.npz"]

        try:
            files.append(inspect.currentframe().f_back.f_locals["__file__"])
        except Exception:
            warnings.warn("Could not find the script file path to save it in the data folder")

        additional_files = {}
        for file in files:
            filepath = Path(file)
            if not filepath.exists():
                warnings.warn(f"File {file} does not exist, unable to save file")
                continue
            additional_files[str(filepath)] = filepath.name
    else:
        additional_files = {}
    quam.data_handler.additional_files = additional_files
    quam.data_handler.save_data(data=data, name=name)

    # Save QuAM to the data folder
    quam.save(
        path=quam.data_handler.path / "state.json",
    )
    quam.save(
        path=quam.data_handler.path / "quam_state",
        content_mapping={"wiring.json": {"wiring", "network"}},
    )

    # Save QuAM to configuration directory / `state.json`
    quam.save(content_mapping={"wiring.json": {"wiring", "network"}})
