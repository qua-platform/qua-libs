from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###################
# The QUA program #
###################

n_avg = 1000  # number of averages

cooldown_time = 50000 // 4  # decay time for qubit

f_min = 30e6
f_max = 50e6
df = 1e6

freqs = np.arange(f_min, f_max + df / 2, df)  # + df/2 to add f_max to freqs

a_min = 0.3
a_max = 1.0
da = 0.1

amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

with program() as resonator_spec:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    f = declare(int)  # variable to sweep freqs
    a = declare(fixed)  # variable for amps sweep
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    I_st = declare_stream()  # stream for I
    Q_st = declare_stream()  # stream for Q

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):
        with for_(
            a, a_min, a < a_max + da / 2, a + da
        ):  # Notice it's + da/2 to include a_max (This is only for fixed!)
            with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
                update_frequency("resonator", f)  # update frequency of resonator element
                wait(cooldown_time, "resonator")  # wait for resonator to decay
                align("resonator", "flux")
                play("offset" * amp(a), "flux", duration=1500)  # duration of flux covers pulse to qubit and readout
                wait(1250, "resonator")  # wait for flux to settle due to filters
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)
                play("minus_offset" * amp(a), "flux", duration=1500)  # negative pulse to quickly remove the current
        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(len(amps), len(freqs)).average().save("I")
        Q_st.buffer(len(amps), len(freqs)).average().save("Q")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=10000,
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, resonator_spec, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(resonator_spec)  # execute QUA program

    res_handles = job.result_handles  # get access to handles
    I_handle = res_handles.get("I")
    I_handle.wait_for_values(1)
    Q_handle = res_handles.get("Q")
    Q_handle.wait_for_values(1)
    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)

    plt.figure()

    while res_handles.is_processing():
        try:
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1
            plt.title("resonator spectroscopy analysis")
            Z = I + Q * 1j
            plt.plot(freqs, np.sqrt(np.abs(Z)))
            # plt.plot(freqs, I)
            # plt.plot(freqs, Q)
            plt.xlabel("IF [Hz]")
            plt.ylabel("demod signal [a.u.]")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    plt.title("resonator spectroscopy analysis")
    Z = I + Q * 1j
    plt.plot(freqs, np.sqrt(np.abs(Z)))
    # plt.plot(freqs, I)
    # plt.plot(freqs, Q)
    plt.xlabel("IF [Hz]")
    plt.ylabel("demod signal [a.u.]")

    plt.figure()
    # If we want to plot the phase...
    # detrend removes the linear increase of phase
    plt.title("resonator spectroscopy analysis")
    phase = signal.detrend(np.unwrap(np.angle(Z)))
    plt.plot(freqs, phase)
    plt.xlabel("IF [Hz]")
    plt.ylabel("phase demod signal [a.u.]")
