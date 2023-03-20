"""
Created on 08/10/2022
@author Jonathan R.
"""

from qm.qua import *
from macros import (
    round_to_fixed,
)
import numpy as np
from scipy import signal
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig, LoopbackInterface
from configuration import config, qop_ip
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qm.simulate.credentials import create_credentials
from qualang_tools.loops import from_array

##############################
# Program-specific variables #
##############################
n_avg = 10  # Number of averaging loops
wait_time = 16 // 4  # time to wait between start if voltage pulse and measurement

# Relevant elements

# 2D scan parameters
scale = config["waveforms"]["sweep"].get("sample")
x_amp = 0.1  # The scan is defined as +/- x_amp/2
y_amp = 0.1  # The scan is defined as +/- y_amp/2
x_res = 11  # Number of points along x
y_res = 11  # Number of points along y
x_axis = np.linspace(-x_amp / 2, x_amp / 2, x_res) / scale
y_axis = np.linspace(-y_amp / 2, y_amp / 2, y_res) / scale
int_time = 20 // 4  # integration time [ns] // 4 = [cycles]

ramp_to_zero_duration = 100
wait_time = 16 // 4


dx = round_to_fixed((x_amp / scale) / (x_res - 1))
dy = round_to_fixed((y_amp / scale) / (y_res - 1))

###################
# The QUA program #
###################
with program() as gate_scan:
    x = declare(fixed)  # a variable to keep track of the x coordinate
    y = declare(fixed)  # a variable to keep track of the x coordinate
    x_st = declare_stream()
    y_st = declare_stream()
    n = declare(int)  # an index variable for the average

    # declaring the measured variables and their streams
    I, Q = declare(fixed), declare(fixed)
    I_stream, Q_stream = declare_stream(), declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # set offsets to initial scan value
        set_dc_offset("G1_sticky", "single", x_axis[0])

        # for_ loop to move the required number of moves in the x direction
        with for_(*from_array(x, x_axis)):
            set_dc_offset("G2_sticky", "single", y_axis[0])
            play("sweep" * amp(dx), "G1_sticky", duration=int_time)
            with for_(*from_array(y, y_axis)):
                # Make sure that we measure after the pulse has settled
                if wait_time >= 4:  # if logic to enable wait_time = 0 without error
                    wait(wait_time, "RF")
                play("sweep" * amp(dy), "G2_sticky", duration=int_time)
                measure(
                    "measure",
                    "RF",
                    None,
                    demod.full("cos", I),
                    demod.full("sin", Q),
                )
                save(I, I_stream)
                save(Q, Q_stream)
                save(y, y_st)
                save(x, x_st)
            ramp_to_zero("G2_sticky", duration=ramp_to_zero_duration)
        ramp_to_zero("G1_sticky", duration=ramp_to_zero_duration)

    with stream_processing():
        for (
            stream_name,
            stream,
        ) in zip(["I", "Q"], [I_stream, Q_stream]):
            stream.buffer(x_res * y_res).average().save(stream_name)
        x_st.save_all("x")
        y_st.save_all("y")

#####################################
#  Open Communication with the QOP  #
#####################################
# qmm = QuantumMachinesManager(qop_ip)
qmm = QuantumMachinesManager(
    host="product-52ecaa43.dev.quantum-machines.co", port=443, credentials=create_credentials()
)

simulation = True

if simulation is True:
    simulation_duration = 200000  # ns

    job = qmm.simulate(
        config=config,
        program=gate_scan,
        simulate=SimulationConfig(
            duration=int(simulation_duration // 4),
        ),
    )
    # plotting the waveform outputted by the OPX
    plt.figure("simulated output samples")
    output_samples = job.get_simulated_samples()
    output_samples.con1.plot()
    plt.show()

    # fetching the data
    result_handles = job.result_handles
    I_handle, Q_handle, x_handle, y_handle = (
        result_handles.I,
        result_handles.Q,
        result_handles.x,
        result_handles.y,
    )
    I, Q, x, y = (
        I_handle.fetch_all(),
        Q_handle.fetch_all(),
        x_handle.fetch_all(),
        y_handle.fetch_all(),
    )

else:
    # Open a quantum machine
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(gate_scan)
    # fetch the data
    results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")
    # Live plot
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.start_time)
        # reshaping the data into the correct order and shape
        order = gate_scan(np.sqrt(I.size))
        I = I[order]
        Q = Q[order]
        # Plot results
        plt.cla()
        plt.pcolor(x_axis, y_axis, np.sqrt(I**2 + Q**2))
        plt.title("Stability diagram")
        plt.xlabel("G1" + "_scan [V]")
        plt.ylabel("G2" + "_scan [V]")
