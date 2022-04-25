"""
intro-to-macros.py: An intro to usage of macros in QUA
Author: Gal Winer - Quantum Machines
Created: 26/12/2020
Revised by Tomer Feld - Quantum Machines
Revision date: 24/04/2022
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import config


qmm = QuantumMachinesManager(port=9510)


def declare_vars(stream_num=1):
    """
    A macro to declare QUA variables. `stream_num` showcases a way to declare multiple streams in an array
    Note that variables and streams don't have to be explicitly returned to the QUA program,
    but it is considered good practice to do so.
    """
    time_var = declare(int, value=100)
    amp_var = declare(fixed, value=0.2)
    stream_array = [declare_stream() for num in range(stream_num)]
    return [time_var, amp_var, stream_array]


def modify_var(addition=0.3):
    """
    A macro to modify a QUA variable. In this case, the variable does not
    need to be returned.
    """
    assign(b, b + addition)


def qua_function_calls(el):
    """
    A macro that calls a QUA play statements
    :param el: The quantum element used by the QUA statements
    :return:
    """
    play("const", el, duration=300)
    play("const" * amp(b), el, duration=300)  # b is a QUA variable


with program() as prog:
    [t, b, c_streams] = declare_vars()

    # Plays pulse with amplitude of 0.2 (from config) * b=0.2 (from declare_vars) for t=100ns (from declare_vars)
    save(b, c_streams[0])  # Saves b into stream for printing at the end
    play("const" * amp(b), "qe1", duration=t)

    # Plays pulse with amplitude of 0.2 (from config) * b=0.5 (after modify_var) for t=100ns (from declare_vars)
    modify_var()
    save(b, c_streams[0])  # Saves b into stream for printing at the end
    play("const" * amp(b), "qe1", duration=t)

    # Plays pulse twice, first with amplitude 0.2 (from config) for duration 300ns (from qua_function_calls).
    # Second with with 0.2 (from config) * b=0.5 (after modify_var) for duration 300ns (from qua_function_calls).
    qua_function_calls("qe1")

    with stream_processing():
        c_streams[0].save_all("out_stream")

job = qmm.simulate(config, prog, SimulationConfig(1500))
res = job.result_handles
out_str = res.out_stream.fetch_all()
samples = job.get_simulated_samples()
samples.con1.plot()

print("##################")
print("b is saved twice, once before the call to modify_var and once afterwards")
print(f"Before:{out_str[0][0]:.1f}, After:{out_str[1][0]:.1f}")
print("##################")
