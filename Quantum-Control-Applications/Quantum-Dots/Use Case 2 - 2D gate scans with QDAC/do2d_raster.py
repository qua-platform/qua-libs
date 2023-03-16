"""
Created on 30/10/2022
@author jdh

Performs a two-dimensional raster scan for a stability diagram measurement, for instance.

The x_amplitude and y_amplitude variables are set to the desired output of the OPX
channels. The amplitude of the waveform configured in the config is taken into account
and rescaled to make it the desired value set in these variables.

The program performs the measurement around the present dc set point, i.e. the measurement is from
-1/2 x_amplitude to +1/2 x_amplitude (likewise for y).

For each of the innermost iterations, we move to the next voltage value and play the measurement
pulse at this location. To prevent the first value of each axis from being skipped, there is a flag for
each axis that denotes whether the measurement is the first in the list. If so, the voltage update
step is skipped.
"""

import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from qm.qua import *
from macros import (
    round_to_fixed,
)

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig, LoopbackInterface
from configuration import config, qop_ip
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qm.simulate.credentials import create_credentials
import time
from qualang_tools.loops import from_array

# desired output amplitude for each axis
x_amplitude = 0.2
y_amplitude = 0.2

# resolution in number of pixels for each axis
x_resolution = 10
y_resolution = 10

# set up the axes including scaling to compensate for waveform amplitude
x_amplitude = x_amplitude / config["waveforms"]["constant"].get("sample")
y_amplitude = y_amplitude / config["waveforms"]["constant"].get("sample")
x_axis = np.linspace(-x_amplitude / 2, x_amplitude / 2, x_resolution)
y_axis = np.linspace(-y_amplitude / 2, y_amplitude / 2, y_resolution)

# step size for each axis
dx = round_to_fixed((x_amplitude) / (x_resolution - 1))
dy = round_to_fixed((y_amplitude) / (y_resolution - 1))

# number of averages
n_averages = 1

# other experimental parameters
integration_time = 16 // 4  # integration time [ns] // 4 = [cycles]
ramp_to_zero_duration = 100
wait_time = 0

with program() as do2d:
    # variables and streams for set data
    # (although they are not exactly right due to rounding errors!!!)
    x = declare(fixed)
    y = declare(fixed)
    x_stream = declare_stream()
    y_stream = declare_stream()

    # variable for averages
    n = declare(int)

    # variables and streams for the measured data
    I = declare(fixed)
    Q = declare(fixed)
    I_stream = declare_stream()
    Q_stream = declare_stream()

    # flags for checking if we are measuring the first element (in which case measure but do not move)
    y_move_flag = declare(bool)
    x_move_flag = declare(bool)

    # averaging loop
    with for_(n, 0, n < n_averages, n + 1):

        # set the x axis to the starting value
        play('constant' * amp(x_axis[0]), 'G1_sticky')

        # assign the x flag to false (= do not move for first iteration)
        assign(x_move_flag, False)

        with for_(*from_array(x, x_axis)):
            play('constant' * amp(dx), 'G1_sticky', condition=x_move_flag)

            # put the y axis at the initial value
            play('constant' * amp(y_axis[0]), 'G2_sticky')
            assign(y_move_flag, False)

            with for_(*from_array(y, y_axis)):
                # make sure that we measure after the pulse has settled
                if wait_time >= 4:  # if logic to enable wait_time = 0 without error
                    wait(wait_time, "RF")

                # update the y axis to the next value
                play("constant" * amp(dy), "G2_sticky", condition=y_move_flag)

                measure(
                    "measure",
                    "RF",
                    None,
                    demod.full("cos", I),
                    demod.full("sin", Q),
                )

                save(I, I_stream)
                save(Q, Q_stream)
                save(y, y_stream)
                save(x, x_stream)

                assign(y_move_flag, True)

            ramp_to_zero("G2_sticky")
            assign(x_move_flag, True)

        ramp_to_zero("G1_sticky")

    with stream_processing():
        for stream_name, stream, in zip(["I", "Q"], [I_stream, Q_stream]):
            stream.buffer(x_resolution * y_resolution).average().save(stream_name)
        x_stream.save_all("x")
        y_stream.save_all("y")


###




#####################################
#  Open Communication with the QOP  #
#####################################
simulation = True
if simulation:

    simulation_duration = 20000  # ns


    # qmm = QuantumMachinesManager(
    #     host='product-52ecaa43.dev.quantum-machines.co',
    #     port=443,
    #     credentials=create_credentials()
    # )
    qmm = QuantumMachinesManager(host="172.16.2.115", port=80)

    job = qmm.simulate(
        config=config,
        program=do2d,
        simulate=SimulationConfig(
            duration=int(simulation_duration // 4),
        ),
    )
    # plotting the waveform outputted by the OPX
    result_handles = job.result_handles

    # result_handles.wait_for_all_values()

    plt.figure("simulated output samples")
    output_samples = job.get_simulated_samples()
    output_samples.con1.plot()

    plt.show()

    # fetching the data
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
    qmm = QuantumMachinesManager(qop_ip, port=85)
    # Open a quantum machine
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(do2d)
    # fetch the data
    results = fetching_tool(job, ["I", "Q", 'x', 'y'])
    # Live plot
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch results
        I, Q, x, y = results.fetch_all()
        I = I.reshape_and_stitch(x_resolution, y_resolution)
        Q = Q.reshape_and_stitch(x_resolution, y_resolution)
        # Progress bar
        progress_counter(1, n_averages, start_time=results.start_time)

        # Plot results
        plt.cla()
        plt.pcolor(x_axis, y_axis, np.sqrt(I ** 2 + Q ** 2))
        plt.colorbar()
        plt.title("Stability diagram")
        plt.xlabel("G1" + "_scan [V]")
        plt.ylabel("G2" + "_scan [V]")
