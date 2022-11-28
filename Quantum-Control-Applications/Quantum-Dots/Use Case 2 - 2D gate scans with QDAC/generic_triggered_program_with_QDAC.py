"""
Created on 22/11/2022
@author jdh

Example program making use of triggering an external instrument. This program does not care about what happens
when that instrument receives the trigger; it is just a program shell to demonstrate the concept. After each trigger,
a generic macro is run on the opx.

As an example of how this could be used, imagine the macro performing a series of measurements that result in a
signal-to-noise ratio. For example, we could use this program to sweep the voltage supplied to an amplifier and run the SNR
measurement on the opx at each voltage on the amplifier.
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

import time
from macros import generic_macro

voltages_for_qdac = np.linspace(0, 0.1, 10)
qdac_wait_time = 2000 // 4  # in clock cycles
qdac_dwell_s = 5e-6



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


with program() as generic_qdac_triggering:
    # for saving the external instrument's set values, we will create a QUA variable and stream.
    # this will also help us structure the program to get the correct number of pause/resume commands.

    external_voltage = declare(fixed)

    external_voltage_stream = declare_stream()

    # variable and stream for the measured data
    SNR_variable = declare(fixed)
    SNR_stream = declare_stream()

    # iteration counter and stream so we can have a progress bar while the program is running.
    iteration_counter = declare(fixed, value=0)
    iteration_stream = declare_stream()

    with for_(*from_array(external_voltage, voltages_for_qdac)):

        # play a trigger command
        play('trig', 'trigger_x')
        wait(qdac_wait_time) # time for the new voltage value to settle

        # it's good practice to send variables and streams to the macro. The variables are global so would be available
        # anyway, but this helps us keep track of where variables are being used or modified.
        generic_macro(SNR_variable, SNR_stream)

        # put the iteration counter into the iteration stream and increment the counter
        save(iteration_counter, iteration_stream)
        assign(iteration_counter, iteration_counter + 1)

    with stream_processing():
        SNR_stream.save_all('SNR')
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
        program=generic_qdac_triggering,
        simulate=SimulationConfig(
            duration=int(simulation_duration // 4),
        ),
        # simulation_interface=LoopbackInterface([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])

    )

    plt.figure("simulated output samples")
    output_samples = job.get_simulated_samples()
    output_samples.con1.plot()

    plt.show()
    result_handles = job.result_handles

    # fetching the data
    SNR_handle = result_handles.SNR

    SNR = SNR_handle.fetch_all()



else:

    qmm = QuantumMachinesManager(qop_ip, port=85)
    # Open a quantum machine
    qm = qmm.open_qm(config)
    
    # connect and set up the qdac
    with qdacII() as qdac:

        channel = 1

        # prepare qdac for receiving a list of voltages to be set to on trigger
        setup_qdac_channels_for_triggered_list(qdac,
                                               channel,
                                               'ext1',
                                               qdac_dwell_s)

        # write list of values to each qdac channel
        qdac.write_binary_values(f'sour{channel}:dc:list:volt', {voltages_for_qdac})


    job = qm.execute(generic_qdac_triggering)

    # fetch the data
    results = fetching_tool(job, ['SNR', 'iteration'], mode="live")

    # Live plot
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch results
        SNR, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, len(voltages_for_qdac), start_time=results.start_time)

