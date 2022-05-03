from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

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
f_max = 40e6
df = 1e6

freqs = np.arange(f_min, f_max + df / 2, df)  # + df/2 to add f_max to freqs

a_min = 0.0
a_max = 1.0
da = 0.1

amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

with program() as qubit_spec:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    f = declare(int)  # variable for freqs sweep
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
                update_frequency("qubit", f)  # update frequency of operations to the qubit
                wait(cooldown_time, "qubit", "resonator")  # wait for qubit to decay
                wait(50, "qubit")  # wait 200 ns, so that resonator is played before qubit saturation pulse
                play("pi", "qubit")  # to create a mixed state between |g> and |e>
                measure(
                    "readout" * amp(a),
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)

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
        duration=100000,
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, qubit_spec, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(qubit_spec)  # execute QUA program

    res_handles = job.result_handles  # get access to handles
    I_handle = res_handles.get("I")
    I_handle.wait_for_values(1)
    Q_handle = res_handles.get("Q")
    Q_handle.wait_for_values(1)
    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)

    while res_handles.is_processing():
        try:
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1
            Z = I + Q * 1j
            plt.title("qubit spectroscopy analysis")
            plt.pcolor(freqs, amps * readout_amp, np.sqrt(np.abs(Z)))
            # plt.plot(freqs, np.sqrt(np.abs(Z)))
            # plt.plot(freqs, I)
            # plt.plot(freqs, Q)
            plt.xlabel("IF [Hz]")
            plt.ylabel("Voltage [V]")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    Z = I + Q * 1j
    plt.title("qb spec with variable readout amp")
    plt.plot(freqs, amps * readout_amp, np.sqrt(np.abs(Z)))
    # plt.plot(freqs, np.sqrt(np.abs(Z)))
    # plt.plot(freqs, I)
    # plt.plot(freqs, Q)
    plt.xlabel("IF [Hz]")
    plt.ylabel("Voltage [V]")

    # If we want to plot the phase...
    # detrend removes the linear increase of phase
    # phase = signal.detrend(np.unwrap(np.angle(I + 1j*Q)))
