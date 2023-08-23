from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_rs import *
# from malab import *
import matplotlib.pyplot as plt
import numpy as np
import os

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###############
# QUA program #
###############

t_min = 16 // 4  # in units of clock cycles
t_max = 1200 // 4  # in units of clock cycles
dt = 8 // 4

times = np.arange(t_min, t_max + dt/2, dt)

# sideband modulation freq
f_min = 95e6
f_max = 110e6
df = 0.1e6

freqs = np.arange(f_min, f_max + df/2, df)  # + df/2 to add f_max to freqs

N_max = 50

cooldown_time = 100000 // 4
flux_dur = 100000 // 4

with program() as chevron_rabi:

    # Declare QUA variables
    ###################
    f = declare(int)  # variable for freqs sweep
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    tau = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    a = declare(fixed, value = ff_a)


    # Pulse sequence
    ################
    with for_(n, 0, n < N_max, n+1):

        with for_(tau, t_min, tau <= t_max, tau + dt):
            with for_(f, f_min, f <= f_max, f + df):
                update_frequency('flux1', f)  # update frequency of operations to the qubit
                wait(cooldown_time, 'qubit1', 'flux1')  # for qubit to decay
                play('pi' * amp(q1_ge_amp), 'qubit1')   # drive the qubit to the |e> state
                align('qubit1', 'flux1')
                play('offset' * amp(a), 'flux1', duration=tau) # apply red sideband flux modulation
                align('resonator1', 'flux1')
                measure('readout', 'resonator1', None,
                        dual_demod.full('cos', 'out1', 'minus_sin', 'out2', I),
                        dual_demod.full('sin', 'out1', 'cos', 'out2', Q))
                save(I, I_st)
                save(Q, Q_st)

            align()
        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save('iteration')
        I_st.buffer(len(times), len(freqs)).average().save('I')
        Q_st.buffer(len(times), len(freqs)).average().save('Q')

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(duration=100000
                                       , simulation_interface=LoopbackInterface(([('con1', 1, 'con1', 1)])))
    job = qmm.simulate(config, chevron_rabi, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(chevron_rabi)  # execute QUA program

    res_handles = job.result_handles  # get access to handles
    I_handle = res_handles.get('I')
    I_handle.wait_for_values(1)
    Q_handle = res_handles.get('Q')
    Q_handle.wait_for_values(1)
    iteration_handle = res_handles.get('iteration')
    iteration_handle.wait_for_values(1)

    while res_handles.is_processing():
        try:
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1
            plt.title('Chevron pattern')
            plt.pcolor(freqs, times*4, I)
            plt.xlabel('modulation freqs')
            plt.ylabel('Variable pulse length [ns]')
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    plt.title('Chevron pattern')
    plt.pcolor(freqs, times*4, I)
    plt.xlabel('modulation freqs')
    plt.ylabel('Variable pulse length [ns]')
