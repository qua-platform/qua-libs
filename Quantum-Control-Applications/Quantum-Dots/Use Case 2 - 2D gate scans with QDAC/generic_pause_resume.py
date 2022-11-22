"""
Created on 22/11/2022
@author jdh

Example program making use of the pause/resume functionality of the opx. In this case, a list of voltages is
created that will be set on an external instrument. At each of these voltages, a generic macro is run.

As an example of how this could be used, imagine the macro performing a series of measurements that result in a
signal-to-noise ratio. We could use this program to sweep the voltage supplied to an amplifier and run the SNR
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

voltages_for_external_instrument = np.linspace(0, 1, 10)

def example_instrument_set_function(voltage):
    print(f'set fake instrument to {voltage}')

with program() as generic_pause_resume:
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

    with for_(*from_array(external_voltage, voltages_for_external_instrument)):
        pause()

        # it's good practice to send variables and streams to the macro. The variables are global so would be available
        # anyway, but this helps us keep track of where variables are being modified.
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
        program=generic_pause_resume,
        simulate=SimulationConfig(
            duration=int(simulation_duration // 4),
        ),
        # simulation_interface=LoopbackInterface([('con1', 1, 'con1', 1), ('con1', 2, 'con1', 2)])

    )

    # this is commented out because pause/resume does not work on the simulator at the time of writing
    # # pause/resume program

    # for external_voltage in voltages_for_external_instrument:
    #
    #     # poll the opx to check if the program is in the paused state. If not, wait 1 and repoll.
    #     while not job.is_paused():
    #         time.sleep(1e-3)
    #
    #     # send instruction to the external instrument
    #     example_instrument_set_function(external_voltage)
    #
    #     # resume the job. This python loop then iterates. We now await the next paused state
    #     job.resume()

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

    job = qm.execute(generic_pause_resume)

    #### pause/resume program ####

    for external_voltage in voltages_for_external_instrument:

        # poll the opx to check if the program is in the paused state. If not, wait 1 and repoll.
        while not job.is_paused():
            time.sleep(1e-3)

        # send instruction to the external instrument
        example_instrument_set_function(external_voltage)

        # resume the job. This python loop then iterates. We now await the next paused state
        job.resume()

    # fetch the data
    results = fetching_tool(job, ['SNR', 'iteration'], mode="live")

    # Live plot
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch results
        SNR, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, len(voltages_for_external_instrument), start_time=results.start_time)

