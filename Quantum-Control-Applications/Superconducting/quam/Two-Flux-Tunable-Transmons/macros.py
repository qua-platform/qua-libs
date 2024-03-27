from qm.qua import *
from components import FluxLine, QuAM

# def wait_depletion_time(quam: "QuAM"):
#

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


def to_independent_idle(z: "FluxLine.FluxLine"):
    set_dc_offset(z.name, "single", z.independent_offset)


def to_joint_idle(z: "FluxLine.FluxLine"):
    set_dc_offset(z.name, "single", z.joint_offset)


def apply_z_to_min(z: "FluxLine.FluxLine"):
    set_dc_offset(z.name, "single", z.min_offset)



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
