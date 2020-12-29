"""
hahn_echo.py: hahn echo experiment
Author: Arthur Strauss - Quantum Machines
Created: 16/11/2020
Created on QUA version: 0.5.138
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time

def hahn_echo(meas_len = 2000, t_len = 81, t_stop = 407, rep_num = 500000, f = 10e6, pi_time = 100):
    t_vec = [int(t_) for t_ in np.linspace(7, t_stop, t_len)]
    with program() as hahn_echo:
        update_frequency("qubit", int(f))
        n = declare(int)
        m = declare(int)
        t = declare(int)
        result1 = declare(int, size=int(meas_len / 500))
        resultLen1 = declare(int)
        result2 = declare(int, size=int(meas_len / 500))
        resultLen2 = declare(int)
        rabi_vec = declare(int, size=t_len)
        rabi_vec2 = declare(int, size=t_len)
        t_ind = declare(int)
        play('rabi', 'laser', duration=3000 // 4)
        with for_(n, 0, n < rep_num, n + 1):
            assign(t_ind, 0)
            with for_each_(t, t_vec):
                z_rotation(np.pi, 'qubit')  #Moving into the rotating frame of the qubit
                play('rabi', 'qubit', duration=((pi_time // 2) // 4)) #Initial π/2 pulse of echo sequence, divided by 4 at each time because of clock cycle duration (4 ns)
                wait(t, 'qubit')  #Wait time
                play('rabi', 'qubit', duration=(pi_time // 4)) # π-pulse
                wait(t, 'qubit') #Wait again
                play('rabi', 'qubit', duration=((pi_time // 2) // 4))  ## Play again a π/2 pulse to return at the top of the Bloch sphere, allowing deduction of T1 time
                align('qubit', 'readout1', 'readout2', "laser")  #Wait for all the elements to be ready for readout
                readout_QUA(result1, result2, resultLen1, resultLen2, meas_len, rabi_vec, t_ind, m) #Proceed to readout
                # Repeat sequence
                play('rabi', 'qubit', duration=((pi_time // 2) // 4))
                wait(t, 'qubit')
                play('rabi', 'qubit', duration=(pi_time // 4))
                wait(t, 'qubit')
                z_rotation(np.pi, 'qubit')
                play('rabi', 'qubit', duration=((pi_time // 2) // 4))
                align('qubit', 'readout1', 'readout2', "laser")
                readout_QUA(result1, result2, resultLen1, resultLen2, meas_len, rabi_vec2, t_ind, m)
                assign(t_ind, t_ind + 1)
        #Save all the results
        with for_(t_ind, 0, t_ind < rabi_vec.length(), t_ind + 1):
            save(rabi_vec[t_ind], "rabi")
        with for_(t_ind, 0, t_ind < rabi_vec2.length(), t_ind + 1):
            save(rabi_vec2[t_ind], "rabi2")
    # job = qm.simulate(hahn_echo, SimulationConfig(20000))
    # res = job.get_simulated_samples()
    job = qm.execute(hahn_echo, duration_limit=0, data_limit=0, force_execution=True)
    job.wait_for_all_results()
    res = job.get_results()
    plt.figure()
    plt.plot(np.linspace(7, t_stop, t_len) * 4, res.variable_results.rabi.values)
    plt.plot(np.linspace(7, t_stop, t_len) * 4, res.variable_results.rabi2.values)
    plt.figure()
    plt.plot(np.linspace(7, t_stop, t_len) * 4, res.variable_results.rabi.values - res.variable_results.rabi2.values)
    plt.plot(np.linspace(7, t_stop, t_len) * 4, res.variable_results.rabi.values - res.variable_results.rabi2.values, 'o')
    return res

