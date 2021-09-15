import numpy as np
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools import baking
from qualang_tools.bakery.bakery import Baking
from aws_config import config, cat_frequencies


def get_IF(element):
    if element in config["elements"]:
        return config["elements"][element]["intermediate_frequency"]


detuning_X_gate = 2e5
delta_SWAP = 3e6
omega_p = (
    cat_frequencies["beta"]
    - cat_frequencies["rho"]
    - get_IF("beta_1_buffer")
    - delta_SWAP
)
n_parity = 2
t_parity = 100
th = -0.01
elements_list = [
    "gamma_1_ATS",
    "gamma_1_buffer",
    "gamma_2_ATS",
    "gamma_2_buffer",
    "alpha_1_ATS",
    "alpha_1_buffer",
    "alpha_2_ATS",
    "alpha_2_buffer",
    "beta_1_ATS",
    "beta_1_buffer",
    "beta_2_ATS",
    "beta_2_buffer",
    "CNOT_gamma_1_ATS",
    "CNOT_gamma_2_ATS",
    "CNOT_alpha_1_ATS",
    "CNOT_alpha_2_ATS",
]


def get_buffer_samples(pulse: str):
    waveform_I = config["pulses"][pulse]["waveforms"]["I"]
    waveform_Q = config["pulses"][pulse]["waveforms"]["Q"]
    samples_I = config["waveforms"][waveform_I]["samples"]
    samples_Q = config["waveforms"][waveform_Q]["samples"]

    return [samples_I, samples_Q]


def stabilize(qubit: str):
    """
    Initiate a stabilization procedure for qubit (tune up the two photon dissipation process)
    """
    ats = qubit + "_ATS"
    buffer = qubit + "_buffer"
    play("Pump_Op", ats)
    play("drive", buffer)


def Z(qubit: str):
    ats = qubit + "_ATS"
    buffer = qubit + "_buffer"
    if qubit == "gamma_1" or qubit == "gamma_2":
        update_frequency(buffer, cat_frequencies["gamma"])
    elif qubit == "alpha_1" or qubit == "alpha_2":
        update_frequency(buffer, cat_frequencies["alpha"])
    elif qubit == "beta_1" or qubit == "beta_2":
        update_frequency(buffer, cat_frequencies["beta"])
    play("Pump_Op", ats)
    play("drive", buffer)
    update_frequency(buffer, get_IF(buffer))


def X_gate(qubit: str, phi_function: list):
    ats = qubit + "_ATS"
    buffer = qubit + "_buffer"
    linear_wf_I = config["pulses"]["linear_pulse"]["waveforms"]["I"]
    linear_wf_Q = config["pulses"]["linear_pulse"]["waveforms"]["Q"]
    if "samples" in config["waveforms"][linear_wf_I]:
        samples_I = config["waveforms"][linear_wf_I]["samples"]
    else:
        samples_I = [config["waveforms"][linear_wf_I]["sample"]] * config["pulses"][
            "linear_pulse"
        ]["length"]
    if "samples" in config["waveforms"][linear_wf_Q]:
        samples_Q = config["waveforms"][linear_wf_Q]["samples"]
    else:
        samples_Q = [config["waveforms"][linear_wf_Q]["sample"]] * config["pulses"][
            "linear_pulse"
        ]["length"]
    with baking(config) as X_baked:
        for i in range(len(samples_I)):
            Op = f"Op{i}"
            X_baked.add_Op(Op, buffer, [[samples_I[i]], [samples_Q[i]]])
            X_baked.frame_rotation(phi_function[i], buffer)
            X_baked.play(Op, buffer)
        X_baked.play("Pump_Op", ats)
        # b.align(ats, buffer)

    return X_baked


def X(qubit: str, b: Baking):
    """
    Apply a non-adiabatic X gate (with compensating Hamiltonian)
    """
    ats = qubit + "_ATS"
    buffer = qubit + "_buffer"
    update_frequency(ats, get_IF(ats) - 2 * detuning_X_gate)
    update_frequency(buffer, get_IF(buffer) + detuning_X_gate)
    # play("Pump_Op", ats)
    # play("linear_Op", buffer, chirp=(400, 'Hz/nsec'))
    b.run()
    update_frequency(ats, get_IF(ats))
    update_frequency(buffer, get_IF(buffer))


def initialize_state(qubit: str):
    stabilize(qubit)


def CNOT(ctrl: str, tgt: str):
    ats_ctrl = ctrl + "_ATS"
    buffer_ctrl = ctrl + "_buffer"
    buffer_tgt = tgt + "_buffer"
    ats_tgt = tgt + "_ATS"
    CNOT_el = "CNOT_" + tgt + "_ATS"

    play("Pump_Op", CNOT_el)    # condition X rotation with (w_control - w_b_tgt) / 2 (ATS)
    play("Pump_Op", ats_tgt)    # stabilize target with 2w_tgt - w_b_tgt (ATS
    play("drive", buffer_tgt, chirp=(20, "Hz/nsec"))   # rotate target with w_b_tgt chirped,


# # Program test
# phi_func = list(np.linspace(0, np.pi, config["pulses"]["linear_pulse"]["length"]))
# b = X_gate("gamma_1", phi_func)
#
# baked_pulse_I = config["waveforms"]["gamma_1_buffer_baked_wf_I_0"]["samples"]
# baked_pulse_Q = config["waveforms"]["gamma_1_buffer_baked_wf_Q_0"]["samples"]
# # plt.plot(t, baked_pulse_I, t, baked_pulse_Q)
#
# # with program() as prog:
# #     a = declare(int, value=2)
# #     X("gamma_1", b)
# #     play("X",'t0')
# #     # CNOT("beta_1", "alpha_1")
# #     # CNOT("beta_1", "gamma_1")
# #     # CNOT("beta_2", "alpha_2")
# #     # CNOT("beta_2", "gamma_2")
# #
# #
# # qmm = QuantumMachinesManager()
# # job = qmm.simulate(config, prog, simulate=SimulationConfig(1500))
# # samples = job.get_simulated_samples()
# # # plt.figure()
# # # plt.plot(samples.con1.analog['10'])
# # # plt.plot(samples.con1.analog['9'])
# # samples.con1.plot()
# # results = job.result_handles
