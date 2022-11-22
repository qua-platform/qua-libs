"""
Created on 30/10/2022
@author jdh

Performs a raster scan over QDAC (or other DAC) values. At each point, performs an OPX raster
scan over the same axes. These plots are averaged and reshaped to return a large 2D dataset of the
OPX scans patched together.

The program makes use of the pause/resume functionality of the opx. The program is compiled, loaded, and launched
on the opx. When we need to send a command to the opx, the program is paused. In the mean time, a python program running
on the local pc is polling the opx to check the state of the program. When the program is paused, the local python
program sends a VISA command to the qdac, and then runs job.resume() to resume the program. The structure of the QUA
and python programs is the same so the correct commands are sent to the qdac.

A triggered version of this program is available in triggered_large_scan.py.

"""

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from qm.qua import *
from macros import (
    round_to_fixed,
)

from python_macros import TimingModule, reshape_for_do2d
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig, LoopbackInterface
from configuration import config, qop_ip
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qm.simulate.credentials import create_credentials
from qualang_tools.loops import from_array

from qdacii_visa import QDACII

import time
from macros import do2d

### variables for small scan

opx_x_amplitude = 0.05
opx_y_amplitude = 0.05

opx_x_resolution = 128
opx_y_resolution = 128

n_averages = 1


### variables for large scan

# TODO: We need to work out how large the opx scan is in terms of qdac voltage

# equivalent dc amplitudes for the do2d scan - i.e. how large is the amplitude of the fast
# scan in terms of the qdac output
do2d_x_dc_amplitude = 0.1
do2d_y_dc_amplitude = 0.1

# etc etc. For now just use values here:
qdac_x_resolution = 10
qdac_y_resolution = 10

# should be in volts as qdac driver is in volts
qdac_x_vals = np.linspace(0.5, 1, qdac_x_resolution)
qdac_y_vals = np.linspace(0.4, 0.9, qdac_y_resolution)

wait_time = 16 // 4


def reshape_and_stitch(data):
    """
    Reorder the data as measured into an array of
    (qdac_x_resolution * opx_x_resolution, qdac_y_resolution * opx_y_resolution)
    """
    return reshape_for_do2d(data, n_averages, qdac_x_resolution, qdac_y_resolution, opx_x_resolution,
                            opx_y_resolution)


with program() as do_large_2d:
    # for saving set data
    qdac_x = declare(fixed)
    qdac_y = declare(fixed)

    qdac_x_stream = declare_stream()
    qdac_y_stream = declare_stream()

    opx_x_stream = declare_stream()
    opx_y_stream = declare_stream()

    # variables and streams for the measured data
    I = declare(fixed)
    Q = declare(fixed)
    I_stream = declare_stream()
    Q_stream = declare_stream()

    iteration_counter = declare(fixed, value=0)
    iteration_stream = declare_stream()

    # probably best to average inside the movement of the qdac because
    # otherwise we incur overhead from having to move it a lot

    with for_(*from_array(qdac_x, qdac_x_vals)):

        # this pauses the program. A command will be sent to the qdac to set it to qdac_x, and the program will be
        # resumed.
        pause()

        with for_(*from_array(qdac_y, qdac_y_vals)):
            # the python program will set qdac_y and resume the program after this line:
            pause()

            do2d('G1_sticky', opx_x_amplitude, opx_x_resolution,
                 'G2_sticky', opx_y_amplitude, opx_y_resolution,
                 n_averages, I, Q, I_stream, Q_stream,
                 opx_x_stream, opx_y_stream, wait_time)

            save(iteration_counter, iteration_stream)
            assign(iteration_counter, iteration_counter + 1)

    with stream_processing():
        for stream_name, stream, in zip(["I", "Q"],
                                        [I_stream, Q_stream]):
            stream.buffer(opx_x_resolution, opx_y_resolution).save_all(stream_name)
        opx_x_stream.save_all("x")
        opx_y_stream.save_all("y")
        iteration_stream.save('iteration')

#####################################
#  Open Communication with the QOP  #
#####################################

simulation = False

if simulation:
    simulation_duration = 10000  # ns

    qmm = QuantumMachinesManager(
        host='product-52ecaa43.dev.quantum-machines.co',
        port=443,
        credentials=create_credentials()
    )

    job = qmm.simulate(
        config=config,
        program=do_large_2d,
        simulate=SimulationConfig(
            duration=int(simulation_duration // 4),
        ),
        # simulation_interface=LoopbackInterface([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])

    )

    # this is commented out because pause/resume does not work on the simulator at the time of writing
    # # pause/resume program
    #
    # for qdac_x_val in qdac_x_vals:
    #
    #     while not job.is_paused():
    #         time.sleep(1e-3)
    #
    #     qdac_x(qdac_x_val)
    #     job.resume()
    #
    #
    #     for qdac_y_val in qdac_y_vals:
    #
    #         while not job.is_paused():
    #             time.sleep(1e-3)
    #
    #         qdac_y(qdac_y_val)
    #         job.resume()

    plt.figure("simulated output samples")
    output_samples = job.get_simulated_samples()
    output_samples.con1.plot()

    plt.show()
    result_handles = job.result_handles

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

    I = reshape_and_stitch(I)
    Q = reshape_and_stitch(Q)

else:

    qmm = QuantumMachinesManager(qop_ip, port=85)
    # Open a quantum machine
    qm = qmm.open_qm(config)

    # connect to the qdac
    with QDACII() as qdac:

        # example channels from the dac
        x_channel = '1'
        y_channel = '2'

        # should do some qdac setup here (not sure if this is right as there isn't much info in the driver)
        qdac.write(f"sour{x_channel}:dc:mode fix")
        qdac.write(f"sour{y_channel}:dc:mode fix")

        # Execute the QUA program
        job = qm.execute(do_large_2d)

        # pause/resume program

        for qdac_x_val in qdac_x_vals:

            while not job.is_paused():
                time.sleep(1e-3)

            # TODO: check if this is a blocking command
            qdac.write(f'outp:sour{x_channel}:volt {qdac_x_val}')

            job.resume()

            for qdac_y_val in qdac_y_vals:

                while not job.is_paused():
                    time.sleep(1e-3)

                qdac.write(f'outp:sour{y_channel}:volt {qdac_y_val}')
                job.resume()

    # fetch the data
    results = fetching_tool(job, ["I", "Q", 'x', 'y', 'iteration'], mode="live")

    # Live plot
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch results
        I, Q, x, y, iteration = results.fetch_all()

        I = reshape_and_stitch(I)
        Q = reshape_and_stitch(Q)

        # Progress bar
        progress_counter(iteration, qdac_x_resolution * qdac_y_resolution, start_time=results.start_time)

        # Plot results
        plt.cla()
        # plt.pcolor(x_axis, y_axis, np.sqrt(I ** 2 + Q ** 2))
        plt.imshow(np.sqrt(I ** 2 + Q ** 2), aspect='auto', origin='lower')
        plt.title("Stability diagram")
        plt.colorbar()
        plt.xlabel("G1" + "_scan [V]")
        plt.ylabel("G2" + "_scan [V]")

