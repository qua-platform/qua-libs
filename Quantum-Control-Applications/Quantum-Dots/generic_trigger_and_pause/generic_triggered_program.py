"""
Created on 22/11/2022
@author jdh

Example program making use of triggering an external instrument. This program does not care about what happens
when that instrument receives the trigger; it is just a program shell to demonstrate the concept. After each trigger,
a generic macro is run on the opx.

As an example of how this could be used, imagine the macro performing a series of measurements that result in a
signal-to-noise ratio. We could use this program to sweep the voltage supplied to an amplifier and run the SNR
measurement on the opx at each voltage on the amplifier.

This program requires that your instrument has an option whereby you send it a predefined list of parameter values
(here the variable set_variables_for_external_instrument) that are sequentially set after the instrument receives a
trigger. If your instrument does not have this functionality, the generic_pause_resume program may be more helpful.
"""

import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from qm.qua import *

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig, LoopbackInterface
from configuration import config, qop_ip
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qm.simulate.credentials import create_credentials
from qualang_tools.loops import from_array

from macros import generic_macro

# this array needs to be sent to your external instrument such that when it receives a trigger it sets the relevant
# parameter to the next value in the array
set_variables_for_external_instrument = np.linspace(0, 1, 10)

with program() as generic_pause_resume:
    # for saving the external instrument's set values, we will create a QUA variable and stream.
    # this will also help us structure the program to get the correct number of pause/resume commands.

    # external set variable for an instrument that is not the OPX
    set_variable = declare(fixed)

    # a stream to track this set variable
    set_variable_stream = declare_stream()

    # variable and stream for the measured data
    measured_variable = declare(fixed)
    measured_variable_stream = declare_stream()

    # iteration counter and stream so we can have a progress bar while the program is running.
    iteration_counter = declare(fixed, value=0)
    iteration_stream = declare_stream()

    with for_(*from_array(set_variable, set_variables_for_external_instrument)):

        # play a trigger command to trigger the external instrument to update its value
        play('trig', 'trigger_x')

        # it's good practice to send variables and streams to the macro. The variables are global so would be available
        # anyway, but this helps us keep track of where variables are being modified.
        # this macro will use the OPX to measure some data and store it in the measured_variable_stream
        generic_macro(measured_variable, measured_variable_stream)

        # put the iteration counter into the iteration stream and increment the counter
        save(iteration_counter, iteration_stream)
        assign(iteration_counter, iteration_counter + 1)

    with stream_processing():
        measured_variable_stream.save_all('measured_variable')
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
        program=generic_pause_resume,
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
    measured_variable_handle = result_handles.measured_variable

    measured_variable_data = measured_variable_handle.fetch_all()



else:

    qmm = QuantumMachinesManager(qop_ip)
    # Open a quantum machine
    qm = qmm.open_qm(config)

    job = qm.execute(generic_pause_resume)

    # fetch the data
    results = fetching_tool(job, ['measured_variable', 'iteration'], mode="live")

    # Live plot
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch results
        measured_variable_data, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, len(set_variables_for_external_instrument), start_time=results.start_time)

        plt.plot(measured_variable_data)
