"""
        CHARGE SENSOR SWEEP
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter

###################
# The QUA program #
###################
n_avg = 100  # Number of averaging loops

offset_min = -0.2
offset_max = 0.2
d_offset = 0.01
offsets = np.arange(-0.2, 0.2, 0.01)

with program() as chargesensor_sweep:
    
    dc = declare(fixed)
    n = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    counter = declare(int, value=0)
    counter_st = declare_stream()

    play('bias'*amp(offset_min/charge_sensor_amp), 'charge_senstor_gate_sticky')

    with for_(*from_array(dc, offsets)):
        save(counter, counter_st)
        play('bias'*amp(d_offset/charge_sensor_amp), 'charge_senstor_gate_sticky')
        align('charge_sensor_sticky', 'charge_sensor_DC')
        with for_(n, 0, n < n_avg, n+1):
            measure('readout', 'charge_sensor_DC', None, integration.full('cos', I, 'out1'))
            save(I, I_st)
        assign(counter, counter+1)

    ramp_to_zero('charge_sensor_gate_sticky')

    with stream_processing():
        I.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(offsets)).save('I')
        counter.save('counter')

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, chargesensor_sweep, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(chargesensor_sweep)
    # Get results from QUA program
    my_results = fetching_tool(job, data_list=['counter'], mode="live")
    fit = plt.figure()
    while my_results.is_processing():
        counter = my_results.fetch_all()[0]
        progress_counter(counter, len(offsets), start_time=my_results.get_start_time())
    my_results = fetching_tool(job, data_list=['counter', 'I'])
    counter, I = my_results.fetch_all()
    plt.plot(offsets, I)
