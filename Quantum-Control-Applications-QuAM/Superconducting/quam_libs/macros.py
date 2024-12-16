import inspect
import pathlib
from pathlib import Path
from typing import Optional, Union
import warnings
import xarray as xr

from qm.qua import *
from quam_libs.components import QuAM


__all__ = [
    "apply_all_flux_to_min",
    "apply_all_flux_to_idle",
    "qua_declaration",
    "multiplexed_readout",
    "node_save",
]


def apply_all_flux_to_min(quam: "QuAM"):
    align()
    for q in quam.active_qubits:
        q.z.to_min()
    align()


def apply_all_flux_to_idle(quam: "QuAM"):
    align()
    for q in quam.active_qubits:
        q.z.to_joint_idle()
    align()


def qua_declaration(num_qubits):
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
    # Workaround to manually assign the results variables to the readout elements
    # for i in range(num_qubits):
    #     assign_variables_to_element(f"rr{i}", I[i], Q[i])
    return I, I_st, Q, Q_st, n, n_st


def multiplexed_readout(qubits, I, I_st, Q, Q_st, sequential=False, amplitude=1.0, weights=""):
    """Perform multiplexed readout on two resonators"""

    for ind, q in enumerate(qubits):
        q.resonator.measure("readout", qua_vars=(I[ind], Q[ind]), amplitude_scale=amplitude)

        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])

        if sequential and ind < len(qubits) - 1:
            align(q.resonator.name, qubits[ind + 1].resonator.name)


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

def readout_state(qubit , state, pulse_name : str = 'readout', threshold : float = None, save_qua_var : StreamType = None):
    I = declare(fixed)
    Q = declare(fixed)
    if threshold is None:
        threshold = qubit.resonator.operations[pulse_name].threshold
    qubit.resonator.measure(pulse_name, qua_vars=(I, Q))
    assign(state, Cast.to_int(I > threshold))
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
    
    
def active_reset(
    quam: QuAM,
    name: str,
    save_qua_var: Optional[StreamType] = None,
    pi_pulse_name: str = "x180",
    readout_pulse_name: str = "readout"):
    
    qubit = quam.qubits[name]
    pulse = qubit.resonator.operations[readout_pulse_name]

    I = declare(fixed)
    Q = declare(fixed)
    state = declare(bool)
    attempts = declare(int, value=1)
    assign(attempts, 1)
    qubit.align()
    qubit.resonator.measure("readout", qua_vars=(I, Q))
    assign(state, I > pulse.threshold)
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
    qubit.xy.play(pi_pulse_name, condition=state)
    qubit.align()
    with while_(I > pulse.rus_exit_threshold):
        qubit.align()
        qubit.resonator.measure("readout", qua_vars=(I, Q))
        assign(state, I > pulse.threshold)
        wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
        qubit.xy.play(pi_pulse_name, condition=state)
        qubit.align()
        assign(attempts, attempts + 1)
    wait(500,qubit.xy.name)
    qubit.align()    
    if save_qua_var is not None:
        save(attempts, save_qua_var)


def load_xarray_data(data_path: str | pathlib.PosixPath):
    if isinstance(data_path.name, str):
        data_path = Path(data_path)

    # Load the dataset
    ds = xr.open_dataset(data_path / "ds.h5", engine="netcdf4")  # or engine="netcdf4" if needed

    # View the dataset
    print(ds)

    return ds