"""
iq_blobs.py: performs the iq_blobs measurement
"""
import time

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from datetime import datetime
from qualang_tools.analysis.discriminator import two_state_discriminator

##################
# State and QuAM #
##################
experiment = "iq_blobs"
debug = True
simulate = False
qubit_list = [0, 1]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3

cooldown_time = 5 * u.us // 4

with program() as iq_blobs:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    a = declare(fixed)
    I_g = [declare(fixed) for _ in range(len(qubit_list))]
    Q_g = [declare(fixed) for _ in range(len(qubit_list))]
    I_g_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_g_st = [declare_stream() for _ in range(len(qubit_list))]
    I_e = [declare(fixed) for _ in range(len(qubit_list))]
    Q_e = [declare(fixed) for _ in range(len(qubit_list))]
    I_e_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_e_st = [declare_stream() for _ in range(len(qubit_list))]

    for i in range(len(qubit_list)):
        # bring other qubits to zero frequency
        machine.nullify_qubits(True, qubit_list, i)
        set_dc_offset(machine.qubits[i].name + "_flux", "single", machine.get_flux_bias_point(i, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            measure(
                "readout",
                machine.readout_resonators[i].name,
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I_g[i]),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q_g[i]),
            )
            wait(cooldown_time, machine.readout_resonators[i].name)
            save(I_g[i], I_g_st[i])
            save(Q_g[i], Q_g_st[i])

            align()

            play("x180", machine.qubits[i].name)
            align()
            measure(
                "readout",
                machine.readout_resonators[i].name,
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I_e[i]),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q_e[i]),
            )
            wait(cooldown_time, machine.readout_resonators[i].name)
            save(I_e[i], I_e_st[i])
            save(Q_e[i], Q_e_st[i])

            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(len(qubit_list)):
            I_g_st[i].save_all(f"Ig{i}")
            Q_g_st[i].save_all(f"Qg{i}")
            I_e_st[i].save_all(f"Ie{i}")
            Q_e_st[i].save_all(f"Qe{i}")
            n_st[i].save(f"iteration{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(iq_blobs)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]

    for q in range(len(qubit_list)):
        # Live plotting
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"Ig{q}", f"Qg{q}", f"Ie{q}", f"Qe{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["Ig"] = data[0]
            qubit_data[q]["Qg"] = data[1]
            qubit_data[q]["Ie"] = data[2]
            qubit_data[q]["Qe"] = data[3]
            qubit_data[q]["iteration"] = data[4]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            qubit_data[q]["Ig"],
            qubit_data[q]["Qg"],
            qubit_data[q]["Ie"],
            qubit_data[q]["Qe"],
            b_print=True,
            b_plot=True,
        )
