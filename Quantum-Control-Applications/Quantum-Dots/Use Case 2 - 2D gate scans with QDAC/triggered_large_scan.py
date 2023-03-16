"""
Created on 31/10/2022
@author jdh

Performs a large 2d scan using both the op-x and the qdac in triggered list mode.

A list of voltages is sent to each qdac channel. These voltages are a grid; at each point in this grid, the opx
will perform a 2d scan. The scans are then stitched together to create an overall large scan.

To use, the size of the opx scan at the device must be scaled with respect to the qdac voltges by calibratring the 
voltage on the device. If not, there will be data missing or regions around the perimeter of the opx scans will be 
measured multiple times (in multiple opx scans).

This file includes functions for setting up the qdac to the correct settings. 

"""

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig, LoopbackInterface
from configuration import config, qop_ip
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qm.simulate.credentials import create_credentials
from qualang_tools.loops import from_array
from python_macros import reshape_for_do2d
from QDAC_II import QDACII
from macros import do2d

### variables for OPX scan

opx_amplitude = [0.3, 0.3]
opx_resolution = [4, 4]
n_averages = 1


### variables for large scan

# equivalent dc amplitudes for the do2d scan - i.e. how large is the amplitude of the fast
# scan in terms of the qdac output
opx_qdac_scale = [1, 1] # this needs to be calibrated and affected by attenuators, dividers etc. Not necessarly the same for x and y
number_of_tiles = [4, 4]
scan_center = [0, 0]  # in voltage
small_scan_window = [opx_amplitude[i] * opx_qdac_scale[i] for i in range(2)]
full_scan_window = [number_of_tiles[i] * opx_amplitude[i] * opx_qdac_scale[i] for i in range(2)]

qdac_x_vals = [scan_center[0] - full_scan_window[0]/2 + small_scan_window[0]/2 + i * small_scan_window[0] for i in range(number_of_tiles[0])]
qdac_y_vals = [scan_center[1] - full_scan_window[1]/2 + small_scan_window[1]/2 + i * small_scan_window[1] for i in range(number_of_tiles[1])]
qdac_wait_time = 7000 // 4  # in clock cycles

wait_time = 16 // 4  # voltage stabilization for OPX ramps
qdac_dwell_s = 5e-6 # voltage stabilization for qdac ramps

def reshape_and_stitch(data):
    """
    Reorder the data as measured into an array of
    (number_of_tiles_x * opx_resolution[0], number_of_tiles_y * opx_resolution[1])
    """
    return reshape_for_do2d(data, n_averages, number_of_tiles[0], number_of_tiles[1], opx_resolution[0],
                            opx_resolution[1])


with program() as do_large_2d:
    # for saving set points data
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
            wait(qdac_wait_time)

            # will save x and y streams for opx set values
            do2d('G1_sticky', opx_amplitude[0], opx_resolution[0],
                 'G2_sticky', opx_amplitude[1], opx_resolution[1],
                 n_averages, I, Q, I_stream, Q_stream,
                 opx_x_stream, opx_y_stream, wait_time)

            save(iteration_counter, iteration_stream)
            assign(iteration_counter, iteration_counter + 1)

    with stream_processing():
        for stream_name, stream, in zip(["I", "Q"],
                                        [I_stream, Q_stream]):
            stream.buffer(opx_resolution[0], opx_resolution[1]).save_all(stream_name)
        opx_x_stream.save_all("x")
        opx_y_stream.save_all("y")
        iteration_stream.save('iteration')


#####################################
#  Open Communication with the QOP  #
#####################################
simulation = True

if simulation:

    simulation_duration = 50000  # ns

    qmm = QuantumMachinesManager(
        host=qop_ip,
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

else:

    # Open a quantum machine
    qmm = QuantumMachinesManager(qop_ip)
    qm = qmm.open_qm(config)

    # connect and set up the qdac

    qdac = QDACII()
    x_channel = 7
    y_channel = 8

    # prepare qdac for receiving a list of voltages to be set to on trigger
    qdac.setup_qdac_channels_for_triggered_list((x_channel, y_channel),
                                           ('ext3', 'ext4'),
                                           (qdac_dwell_s, qdac_dwell_s))

    # write list of values to each qdac channel
    qdac.write(f'sour{x_channel}:dc:list:volt {",".join(map(str, qdac_x_vals.tolist()))}')
    qdac.write(f'sour{y_channel}:dc:list:volt {",".join(map(str, qdac_x_vals.tolist()))}')



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
        progress_counter(iteration, number_of_tiles_x * number_of_tiles_y, start_time=results.start_time)
        # reshaping the data into the correct order and shape
        # Plot results
        plt.cla()
        plt.imshow(np.sqrt(I ** 2 + Q ** 2), aspect='auto', origin='lower')

        # plt.pcolor(x_axis, y_axis, np.sqrt(I**2 + Q**2))
        plt.title("Stability diagram")
        plt.xlabel("G1" + "_scan [V]")
        plt.ylabel("G2" + "_scan [V]")



