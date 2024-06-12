from qm.qua import *
from quam_components import QuAM


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


def multiplexed_readout(qubits, I, I_st, Q, Q_st, sequential=False, amplitude=1.0, weights=""):
    """Perform multiplexed readout on two resonators"""

    for ind, q in enumerate(qubits):
        # TODO: demod.accumulated?
        q.resonator.measure("readout", qua_vars=(I[ind], Q[ind]))  # TODO: implement amplitude sweep

        if I_st is not None:
            save(I[ind], I_st[ind])
        if Q_st is not None:
            save(Q[ind], Q_st[ind])

        if sequential and ind < len(qubits) - 1:
            align(q.resonator.name, qubits[ind + 1].resonator.name)


def node_save(name: str, data: dict, quam: QuAM):
    # Save results
    quam.data_handler.save_data(data=data, name=name)

    # Save QuAM to the data folder
    quam.save(path=quam.data_handler.path / "quam_state", content_mapping={"wiring.json": {"wiring", "network"}})

    # Save QuAM to current working directory `state.json`
    quam.save(path="quam_state", content_mapping={"wiring.json": {"wiring", "network"}})
