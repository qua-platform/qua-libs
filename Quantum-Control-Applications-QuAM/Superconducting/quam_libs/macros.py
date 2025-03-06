import inspect
from pathlib import Path
from typing import Optional, Union
import warnings

from qm.qua import *
from qm.qua._dsl import Scalar
from quam_libs.components import QuAM
from quam_libs.components import Transmon

__all__ = [
    "qua_declaration",
    "multiplexed_readout",
    "node_save",
    "active_reset",
    "readout_state",
]


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


def readout_state(qubit, state, pulse_name: str = "readout", threshold: float = None, save_qua_var: StreamType = None):
    I = declare(fixed)
    Q = declare(fixed)
    if threshold is None:
        threshold = qubit.resonator.operations[pulse_name].threshold
    qubit.resonator.measure(pulse_name, qua_vars=(I, Q))
    assign(state, Cast.to_int(I > threshold))
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)


def readout_state_gef(
    qubit: Transmon, 
    state: Scalar[int], # : QuaVariableType, # TODO: Fix this type hinting error. for qua 1.2.2rc there is an import error
    pulse_name: str = "readout", save_qua_var: StreamType = None
):
    I = declare(fixed)
    Q = declare(fixed)
    diff = declare(fixed, size=3)

    qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift)
    qubit.resonator.measure(pulse_name, qua_vars=(I, Q))
    qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency)

    gef_centers = [qubit.resonator.gef_centers[0], qubit.resonator.gef_centers[1], qubit.resonator.gef_centers[2]]
    for p in range(3):
        assign(
            diff[p],
            (I - gef_centers[p][0]) * (I - gef_centers[p][0]) + (Q - gef_centers[p][1]) * (Q - gef_centers[p][1]),
        )
    assign(state, Math.argmin(diff))
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)


def active_reset_gef(
    qubit: Transmon,
    readout_pulse_name: str = "readout",
    pi_01_pulse_name: str = "x180",
    pi_12_pulse_name: str = "EF_x180",
    max_attempts: int = 10,
):
    res_ar = declare(int)
    success = declare(int)
    assign(success, 0)
    attempts = declare(int)
    assign(attempts, 0)
    qubit.align()
    with while_((success < 2) & (attempts < max_attempts)):
        readout_state_gef(qubit, res_ar, readout_pulse_name)
        qubit.align()
        with if_(res_ar == 0):
            assign(success, success + 1)  # we need to measure 'g' two times in a row to increase our confidence
        with if_(res_ar == 1):
            update_frequency(qubit.xy.name, int(qubit.xy.intermediate_frequency))
            qubit.xy.play(pi_01_pulse_name)
            assign(success, 0)
        with if_(res_ar == 2):
            update_frequency(
                qubit.xy.name,
                int(qubit.xy.intermediate_frequency - qubit.anharmonicity),
            )
            qubit.xy.play(pi_12_pulse_name)
            update_frequency(qubit.xy.name, int(qubit.xy.intermediate_frequency))
            qubit.xy.play(pi_01_pulse_name)
            assign(success, 0)
        qubit.align()
        assign(attempts, attempts + 1)

def active_reset_simple(
        qubit: Transmon,
        save_qua_var: Optional[StreamType] = None,
        pi_pulse_name: str = "x180",
        readout_pulse_name: str = "readout"):
    """
    Simple active reset for a qubit
    """
    pulse = qubit.resonator.operations[readout_pulse_name]

    I = declare(fixed)
    Q = declare(fixed)
    state = declare(bool)
    qubit.align()
    qubit.resonator.measure("readout", qua_vars=(I, Q))
    assign(state, I > pulse.threshold)
    wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
    qubit.align()
    qubit.xy.play(pi_pulse_name, condition=state)
    qubit.align()


def active_reset(
        qubit: Transmon,
        save_qua_var: Optional[StreamType] = None,
        pi_pulse_name: str = "x180",
        readout_pulse_name: str = "readout",
        max_attempts: int = 15):
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
    qubit.align()
    qubit.xy.play(pi_pulse_name, condition=state)
    qubit.align()
    with while_((I > pulse.rus_exit_threshold) & (attempts < max_attempts)):
        qubit.align()
        qubit.resonator.measure("readout", qua_vars=(I, Q))
        assign(state, I > pulse.threshold)
        wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)
        qubit.align()
        qubit.xy.play(pi_pulse_name, condition=state)
        qubit.align()
        assign(attempts, attempts + 1)
    wait(500, qubit.xy.name)
    qubit.align()
    if save_qua_var is not None:
        save(attempts, save_qua_var)
