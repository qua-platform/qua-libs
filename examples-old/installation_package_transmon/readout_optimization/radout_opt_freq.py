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

cooldown_time = 50000 // 4  # qubit decay time

f_min = 30e6
f_max = 40e6
df = 1e6

freqs = np.arange(f_min, f_max + df / 2, df)  # + df/2 to add f_max to freqs

with program() as rr_opt_f:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    f = declare(int)  # variable for freqs sweep
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    Ig_st = declare_stream()  # stream for Ig
    Qg_st = declare_stream()  # stream for Qg
    Ie_st = declare_stream()  # stream for Ie
    Qe_st = declare_stream()  # stream for Qe

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            update_frequency("resonator", f)
            # |g> IQ blob
            wait(cooldown_time, "qubit")  # wait for qubit to decay
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, Ig_st)
            save(Q, Qg_st)

            align()  # global align

            # |e> IQ blob
            wait(cooldown_time, "qubit")  # wait for qubit to decay
            play("pi", "qubit")  # to populate |e> state
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, Ie_st)
            save(Q, Qe_st)

        save(n, n_st)

    # Stream processing
    ###################
    with stream_processing():
        n_st.save("iteration")
        # mean values
        Ig_st.buffer(len(freqs)).average().save("Igavg")
        Qg_st.buffer(len(freqs)).average().save("Qgavg")
        Ie_st.buffer(len(freqs)).average().save("Ieavg")
        Qe_st.buffer(len(freqs)).average().save("Qeavg")
        # variances
        (
            ((Ig_st.buffer(len(freqs)) * Ig_st.buffer(len(freqs))).average())
            - (Ig_st.buffer(len(freqs)).average() * Ig_st.buffer(len(freqs)).average())
        ).save("Igvar")
        (
            ((Qg_st.buffer(len(freqs)) * Qg_st.buffer(len(freqs))).average())
            - (Qg_st.buffer(len(freqs)).average() * Qg_st.buffer(len(freqs)).average())
        ).save("Qgvar")
        (
            ((Ie_st.buffer(len(freqs)) * Ie_st.buffer(len(freqs))).average())
            - (Ie_st.buffer(len(freqs)).average() * Ie_st.buffer(len(freqs)).average())
        ).save("Ievar")
        (
            ((Qe_st.buffer(len(freqs)) * Qe_st.buffer(len(freqs))).average())
            - (Qe_st.buffer(len(freqs)).average() * Qe_st.buffer(len(freqs)).average())
        ).save("Qevar")


#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=100000,
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, rr_opt_f, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(rr_opt_f)  # execute QUA program

    res_handles = job.result_handles  # get access to handles
    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)
    Ig_avg_handle = res_handles.get("Igavg")
    Ig_avg_handle.wait_for_values(1)
    Qg_avg_handle = res_handles.get("Qgavg")
    Qg_avg_handle.wait_for_values(1)
    Ie_avg_handle = res_handles.get("Ieavg")
    Ie_avg_handle.wait_for_values(1)
    Qe_avg_handle = res_handles.get("Qeavg")
    Qe_avg_handle.wait_for_values(1)
    Ig_var_handle = res_handles.get("Igvar")
    Ig_var_handle.wait_for_values(1)
    Qg_var_handle = res_handles.get("Qgvar")
    Qg_var_handle.wait_for_values(1)
    Ie_var_handle = res_handles.get("Ievar")
    Ie_var_handle.wait_for_values(1)
    Qe_var_handle = res_handles.get("Qevar")
    Qe_var_handle.wait_for_values(1)

    while res_handles.is_processing():
        try:
            Ig_avg = Ig_avg_handle.fetch_all()
            Qg_avg = Qg_avg_handle.fetch_all()
            Ie_avg = Ie_avg_handle.fetch_all()
            Qe_avg = Qe_avg_handle.fetch_all()
            Ig_var = Ig_var_handle.fetch_all()
            Qg_var = Qg_var_handle.fetch_all()
            Ie_var = Ie_var_handle.fetch_all()
            Qe_var = Qe_var_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1

            Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
            var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
            SNR = ((np.abs(Z)) ** 2) / (2 * var)
            plt.plot(freqs, SNR, ".-")
            plt.title("Readout optimization")
            plt.xlabel("IF [Hz]")
            plt.ylabel("SNR")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

        except Exception as e:
            pass

    Ig_avg = Ig_avg_handle.fetch_all()
    Qg_avg = Qg_avg_handle.fetch_all()
    Ie_avg = Ie_avg_handle.fetch_all()
    Qe_avg = Qe_avg_handle.fetch_all()
    Ig_var = Ig_var_handle.fetch_all()
    Qg_var = Qg_var_handle.fetch_all()
    Ie_var = Ie_var_handle.fetch_all()
    Qe_var = Qe_var_handle.fetch_all()

    Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
    var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
    SNR = ((np.abs(Z)) ** 2) / (2 * var)
    plt.plot(freqs, SNR, ".-")
    plt.title("Readout optimization")
    plt.xlabel("IF [Hz]")
    plt.ylabel("SNR")
