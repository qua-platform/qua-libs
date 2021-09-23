"""
optimizing_phase_path.py: Script for generating real time phase modulation profiles for optimizing X gate fidelity
Author: Arthur Strauss - Quantum Machines
Created: 22/09/2021
Created on QUA version: 1.10.1904
This script aims to propose an optimizing scheme to be realized for finding the  optimal real time phase modulation
profile of the buffer drive to realize the adiabatic code deformation necessary to realize a X and a CNOT gate. We show
here multiple alternatives (1. using frequency chirp profiles, 2. using baking tool, 3. using frame rotations)
"""
from configuration import *
from qm import SimulationConfig, LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager, QmJob
from qm.QuantumMachine import QuantumMachine
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy.optimize import minimize
from qualang_tools.bakery.bakery import baking
from Examples.Papers.cat_qubits.qua_macros import *

simulation_config = SimulationConfig(
    duration=int(2e5),  # need to run the simulation for long enough to get all points
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
        noisePower=0.05 ** 2,
    ),
)

revival_time = int(np.pi / chi_qa / 4) * 4  # get revival time in multiples of 4 ns
T = 1e-6  # Assumed time for X gate
shots = 100
Delta = np.pi / T
Delta2 = 1e6  # SWAP frequency
threshold = 0.0

n_pieces = 4
init_rates = [199, 550, -997, 1396]
init_times = [0, 15196, 25397, 56925]
assert len(init_rates) == len(init_times) == n_pieces


def Z_measurement_scheme():
    pass


def state_estimation(state):
    assign(state, True)


""" Option 1: Optimization of frequency chirping profile for real time phase modulation"""


def encode_freq_in_IO(r: List[int], t: List[int], QM: QuantumMachine, job: QmJob):
    """
    Insert chirping parameter values using IOs
    
    :param r: List of frequency rates 
    :param t: List of times
    :param QM: Quantum Machine instance
    :param job: Job instance
    """
    assert len(r) == len(t)
    for j in range(len(r)):
        while not job.is_paused():
            time.sleep(0.1)
        QM.set_io1_value(r[j])
        QM.set_io2_value(t[j])
        job.resume()


with program() as X_gate_freq_chirp:
    n = declare(int)
    i = declare(fixed)
    q = declare(fixed)
    I = declare(fixed)
    a = declare(int)
    rates = declare(int, size=n_pieces)
    times = declare(int, size=n_pieces)
    state = declare(bool)
    state_stream = declare_stream()

    """ Infinite loop part and rates/times assignement used for frequency chirping profile optimization, comment out if
        using baking
        """
    with infinite_loop_():
        pause()
        with for_(n, 0, n < n_pieces, n + 1):
            pause()
            assign(rates[n], IO1)
            assign(times[n], IO2)

        with for_(n, 0, n < shots, n + 1):  # shots for averaging
            with for_(a, -1, a <= 1, a + 2):
                update_frequency("pump", int(omega_p))
                update_frequency("buffer", int(omega_b))
                # State prep: displace cavity to state |0> = |α> or |1> (assuming direct control of storage)
                play("Displace_Op" * amp(a), "storage")
                align()

                # Apply X gate according to https://arxiv.org/abs/1904.09474
                # Option 1: frequency chirping
                update_frequency("pump", int(omega_p - 2 * Delta))
                update_frequency("buffer", int(omega_b + Delta))
                play("pump", "ATS")
                play("drive", "buffer", chirp=(rates, times, 'Hz/nsec'))
                align()

                # Z Measurement
                update_frequency("pump", int(omega_p + Delta2))
                update_frequency("buffer", int(Delta2))
                Z_measurement_scheme()
                state_estimation(state)

                # Active reset of transmon and storage
                align()
                with if_(state):
                    play("X", "transmon")
                    g_one_to_g_zero()

                save(state, state_stream)

    with stream_processing():
        state_stream.boolean_to_int().buffer(2).average().save("state")


def cost_function_freq_chirp(angles, QM: QuantumMachine, job: QmJob):
    r = angles[0: 2 * n_pieces: 2]
    t = angles[1: 2 * n_pieces: 2]
    job.resume()

    encode_freq_in_IO(r, t, QM, job)

    # Implement routine here to set IO values and resume the program, and adapt prior part (or remove it)

    results = job.result_handles
    while not job.is_paused():
        time.sleep(0.1)
    res = results.get("state").fetch_all()
    P_exc_starting_0 = res[0]
    P_exc_starting_1 = res[1]

    return P_exc_starting_1 - P_exc_starting_0  # Result to be minimized (ideal target is -1)


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

job = qm.execute(X_gate_freq_chirp)

params = []
for i in range(n_pieces):
    params.append(rates[i])
    params.append(times[i])

Result = minimize(cost_function_freq_chirp, np.array(params), args=(qm, job), method="Nelder-Mead")
job.halt()

qmm.close_all_quantum_machines()

""""
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
"""

""" Option 2: Optimization of direct phase profile using baking frame rotations"""


def X_baked(phi_function: list, baking_index: int = None):
    ats = "pump"
    buffer = "buffer"
    wf_I = config["pulses"]["drive_pulse"]["waveforms"]["I"]
    wf_Q = config["pulses"]["drive_pulse"]["waveforms"]["Q"]
    if "samples" in config["waveforms"][wf_I]:
        samples_I = config["waveforms"][wf_I]["samples"]
    else:
        samples_I = [config["waveforms"][wf_I]["sample"]] * config["pulses"]["drive_pulse"]["length"]
    if "samples" in config["waveforms"][wf_Q]:
        samples_Q = config["waveforms"][wf_Q]["samples"]
    else:
        samples_Q = [config["waveforms"][wf_Q]["sample"]] * config["pulses"]["drive_pulse"]["length"]

    with baking(config, padding_method="right", override=True, baking_index=baking_index) as x_baked:
        for i in range(len(samples_I)):
            Op = f"Op{i}"
            x_baked.add_Op(Op, buffer, [[samples_I[i]], [samples_Q[i]]])
            x_baked.frame_rotation(phi_function[i], buffer)
            x_baked.play(Op, buffer)
        x_baked.play("Pump_Op", ats)
        # b.align(ats, buffer)

    return x_baked


# Initialize first baking object (optimization will override the waveform created by this baked object)
phi_func = list(np.linspace(0, np.pi, config["pulses"]["drive_pulse"]["length"]))
b = X_baked(phi_function=phi_func)
qm2 = qmm.open_qm(config)

with program() as X_gate_baked_profile:
    n = declare(int)
    I = declare(fixed)
    a = declare(int)
    rates = declare(int, size=n_pieces)
    times = declare(int, size=n_pieces)
    state = declare(bool)
    state_stream = declare_stream()

    with for_(n, 0, n < shots, n + 1):  # shots for averaging
        with for_(a, -1, a <= 1, a + 2):
            # State prep: displace cavity to state |0> = |α> or |1> (assuming direct control of storage)
            play("Displace_Op" * amp(a), "storage")
            align()

            # Option 2: baked drive pulse embedding incremental frame rotations based on optimized function phi(t)
            b.run()

            # Z Measurement
            update_frequency("pump", int(omega_p + Delta2))
            update_frequency("buffer", int(Delta2))
            Z_measurement_scheme()
            state_estimation(state)

            # Active reset of transmon and storage
            align()
            with if_(state):
                play("X", "transmon")
                g_one_to_g_zero()

            save(state, state_stream)

    with stream_processing():
        state_stream.boolean_to_int().buffer(2).average().save("state")


def cost_function_baked_phase_profile(phi_profile):
    baked_X = X_baked(phi_profile, b.get_baking_index()).get_waveforms_dict()
    pending_job = qm.queue.add_compiled(pid, overrides=baked_X)
    job = pending_job.wait_for_execution()
    results = job.result_handles
    results.wait_for_all_values()
    res = results.get("state").fetch_all()
    P_exc_starting_0 = res[0]
    P_exc_starting_1 = res[1]

    return P_exc_starting_1 - P_exc_starting_0  # Result to be minimized (ideal target is -1)


pid = qm.compile(X_gate_baked_profile)
phi_func = list(np.linspace(0, np.pi, config["pulses"]["drive_pulse"]["length"]))
Result2 = minimize(cost_function_baked_phase_profile, np.array(phi_func), method="Nelder-Mead")
