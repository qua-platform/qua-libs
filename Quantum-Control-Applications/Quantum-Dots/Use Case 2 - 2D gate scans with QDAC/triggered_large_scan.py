"""
Created on 31/10/2022
@author jdh

Performs a large 2d scan using both the op-x and the qdac in triggered list mode.

A list of voltages is sent to each qdac channel. These voltages are a grid; at each point in this grid, the opx
will perform a 2d scan. The scans are then stitched together to create an overall large scan.

To use, the size of the opx scan at the device and the deltas in the voltages sent to the qdac must be calibrated.
If not, there will be data missing or regions around the perimeter of the opx scans will be measured multiple times
(in multiple opx scans).

This file includes functions for setting up the qdac to the correct settings. 

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
from qualang_tools.loops import from_array
from python_macros import reshape_for_do2d

from qdacii_visa import QDACII

from macros import do2d

### variables for small scan

opx_x_amplitude = 0.3
opx_y_amplitude = 0.3

opx_x_resolution = 10
opx_y_resolution = 10

n_averages = 1


### variables for large scan

# equivalent dc amplitudes for the do2d scan - i.e. how large is the amplitude of the fast
# scan in terms of the qdac output
do2d_x_dc_amplitude = 0.1
do2d_y_dc_amplitude = 0.1


# etc etc. For now just use values here:
qdac_x_resolution = 10
qdac_y_resolution = 10


qdac_x_vals = np.linspace(0.5, 1, qdac_x_resolution)
qdac_y_vals = np.linspace(0.4, 0.9, qdac_y_resolution)

wait_time = 16 // 4

qdac_dwell_s = 5e-6

def reshape_and_stitch(data):
    """
    Reorder the data as measured into an array of
    (qdac_x_resolution * opx_x_resolution, qdac_y_resolution * opx_y_resolution)
    """
    return reshape_for_do2d(data, n_averages, qdac_x_resolution, qdac_y_resolution, opx_x_resolution,
                            opx_y_resolution)


# this should be in the driver not here but I don't want to make a custom driver that people aren't using
def setup_qdac_channels_for_triggered_list(qdac, channels, trigger_sources, dwell_s_vals):

    for channel, trigger, dwell_s in zip(channels, trigger_sources, dwell_s_vals):
        # Setup LIST connect to external trigger
        # ! Remember to set FIXed mode if you later want to set a voltage directly
        qdac.write(f"sour{channel}:dc:list:dwell {dwell_s}")
        qdac.write(f"sour{channel}:dc:list:tmode stepped")  # point by point trigger mode
        qdac.write(f"sour{channel}:dc:trig:sour {trigger}")
        qdac.write(f"sour{channel}:dc:init:cont on")

        # Always make sure that you are in the correct DC mode (LIST) in case you have switched to FIXed
        qdac.write(f"sour{channel}:dc:mode LIST")


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

    # iterations for the progress bar
    iteration_counter = declare(fixed, value=0)
    iteration_stream = declare_stream()


    # probably best to average inside the movement of the qdac because
    # otherwise we incur overhead from having to move it a lot

    with for_(*from_array(qdac_x, qdac_x_vals)):

        play('trig', 'trigger_x') # check if we need condition for first iteration

        with for_(*from_array(qdac_y, qdac_y_vals)):

            play('trig', 'trigger_y')

            # will save x and y streams for opx set values
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

    I = reshape_and_stitch(I)
    Q = reshape_and_stitch(Q)


else:

    # Open a quantum machine
    qmm = QuantumMachinesManager(qop_ip, port=85)
    qm = qmm.open_qm(config)

    # connect and set up the qdac
    with QDACII() as qdac:

        x_channel = 1
        y_channel = 2

        # prepare qdac for receiving a list of voltages to be set to on trigger
        setup_qdac_channels_for_triggered_list(qdac,
                                               (x_channel, y_channel),
                                               ('ext1', 'ext2'),
                                               (qdac_dwell_s, qdac_dwell_s))

        # write list of values to each qdac channel
        qdac.write_binary_values(f'sour{x_channel}:dc:list:volt', {qdac_x_vals})
        qdac.write_binary_values(f'sour{y_channel}:dc:list:volt', {qdac_y_vals})



        # Execute the QUA program
        job = qm.execute(do_large_2d)


    # fetch the data
    results = fetching_tool(job, ["I", "Q", 'x', 'y', "iteration"], mode="live")
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
        # reshaping the data into the correct order and shape
        # Plot results
        plt.cla()
        plt.imshow(np.sqrt(I ** 2 + Q ** 2), aspect='auto', origin='lower')

        # plt.pcolor(x_axis, y_axis, np.sqrt(I**2 + Q**2))
        plt.title("Stability diagram")
        plt.xlabel("G1" + "_scan [V]")
        plt.ylabel("G2" + "_scan [V]")



