"""
        CHARGE STABILITY DIAGRAM
"""

from importlib import reload
import configuration

reload(configuration)
from configuration import *

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close

#%matplotlib qt

###################
# The QUA program #
###################

n_avg = 10  # Number of averaging loops

offset_min_P1 = -0.04
offset_max_P1 = 0.04
d_offset_P1 = 0.001
offsets_P1 = np.arange(offset_min_P1, offset_max_P1, d_offset_P1)

offset_min_P2 = -0.04
offset_max_P2 = 0.04
d_offset_P2 = 0.001
offsets_P2 = np.arange(offset_min_P2, offset_max_P2, d_offset_P2)

with program() as charge_stability:
    dc_p1 = declare(fixed)
    dc_p2 = declare(fixed)
    n = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    counter = declare(int, value=0)
    counter_st = declare_stream()

    play('bias' * amp(offset_min_P1 / P1_amp), 'P1_sticky')  # -2 to 2
    wait(10 * u.ms, 'P1_sticky')

    with for_(*from_array(dc_p1, offsets_P1)):
        save(counter, counter_st)
        play('bias' * amp(offset_min_P2 / P2_amp), 'P2_sticky')
        wait(10 * u.ms, 'P2_sticky')

        align('P1_sticky', 'P2_sticky', 'charge_sensor_RF')
        with for_(*from_array(dc_p2, offsets_P2)):
            play('bias' * amp(d_offset_P2 / P2_amp), 'P2_sticky')
            wait(10 * u.ms, 'P2_sticky')

            align('P1_sticky', 'P2_sticky', 'charge_sensor_RF')
            with for_(n, 0, n < n_avg, n + 1):
                measure('readout', 'charge_sensor_RF', None, demod.full('cos', I, 'out2'), demod.full('sin', Q, 'out2'))
                save(I, I_st)
                save(Q, Q_st)
        align('P1_sticky', 'P2_sticky', 'charge_sensor_RF')
        ramp_to_zero('P2_sticky', 10 * u.ms)
        play('bias' * amp(d_offset_P1 / P1_amp), 'P1_sticky')
        wait(10 * u.ms, 'P1_sticky')
        assign(counter, counter + 1)

    ramp_to_zero('P1_sticky', 10 * u.ms)

    with stream_processing():
        I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(offsets_P2)).save_all('I')
        Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(len(offsets_P2)).save_all('Q')
        counter_st.save('counter')

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, charge_stability, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(charge_stability)
    # Get results from QUA program
    my_results = fetching_tool(job, data_list=['counter', 'I', 'Q'], mode="live")
    # Live plotting
    fig = plt.figure(figsize=(10, 4))
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while my_results.is_processing():
        counter, I, Q = my_results.fetch_all()
        I_volts = u.demod2volts(I, reflectometry_readout_length)
        Q_volts = u.demod2volts(Q, reflectometry_readout_length)
        progress_counter(counter, len(offsets_P1), start_time=my_results.get_start_time())
        plt.subplot(121)
        plt.cla()
        plt.pcolor(offsets_P2, offsets_P1[:len(I_volts)], I_volts)
        plt.xlabel('P2 [V]')
        plt.ylabel('P1 [V]')
        plt.pause(0.1)
        plt.subplot(122)
        plt.cla()
        plt.pcolor(offsets_P2, offsets_P1[:len(Q_volts)], Q_volts)
        plt.xlabel('P2 [V]')
        plt.ylabel('P1 [V]')
        plt.pause(0.1)
    plt.subplot(121)
    plt.colorbar()
    plt.subplot(122)
    plt.colorbar()
    plt.show()
