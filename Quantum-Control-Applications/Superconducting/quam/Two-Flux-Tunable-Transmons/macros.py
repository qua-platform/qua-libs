from qm.qua import *
from components import FluxLine, QuAM


def node_save(name: str, data: dict, quam: QuAM):
    quam.data_handler.save_data(data=data, name=name)
    quam.save(path=quam.data_handler.path / "state.json")
    quam.save(path="state.json")


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


def qua_declaration(nb_of_qubits):
    """
    Macro to declare the necessary QUA variables

    :param nb_of_qubits: Number of qubits used in this experiment
    :return:
    """
    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(nb_of_qubits)]
    Q = [declare(fixed) for _ in range(nb_of_qubits)]
    I_st = [declare_stream() for _ in range(nb_of_qubits)]
    Q_st = [declare_stream() for _ in range(nb_of_qubits)]
    # Workaround to manually assign the results variables to the readout elements
    # for i in range(nb_of_qubits):
    #     assign_variables_to_element(f"rr{i}", I[i], Q[i])
    return I, I_st, Q, Q_st, n, n_st


def multiplexed_readout(quam: "QuAM", I, I_st, Q, Q_st, sequential: bool = False, amplitude_scale=None, weights=""):
    """Perform multiplexed readout on two resonators"""

    for ind, q in enumerate(quam.active_qubits):
        q.resonator.measure("readout", qua_vars=(I[ind], Q[ind]), amplitude_scale=amplitude_scale)

        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])

        if sequential and ind < len(quam.active_qubits) - 1:
            align(q.resonator.name, quam.active_qubits[ind + 1].resonator.name)
