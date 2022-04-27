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

a_min = 0.0
a_max = 1.0
da = 0.1

amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

with program() as rr_opt_a:

    # Declare QUA variables
    ###################
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    a = declare(fixed)  # variable for amps sweep
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    Ig_st = declare_stream()  # stream for Ig
    Qg_st = declare_stream()  # stream for Qg
    Ie_st = declare_stream()  # stream for Ie
    Qe_st = declare_stream()  # stream for Qe

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):
        with for_(
            a, a_min, a < a_max + da / 2, a + da
        ):  # Notice it's + da/2 to include a_max (This is only for fixed!)
            # |g> IQ blob
            wait(cooldown_time, "qubit")  # wait for qubit to decay
            align("qubit", "resonator")
            measure(
                "readout" * amp(a),
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
                "readout" * amp(a),
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
        Ig_st.buffer(len(amps)).average().save("Igavg")
        Qg_st.buffer(len(amps)).average().save("Qgavg")
        Ie_st.buffer(len(amps)).average().save("Ieavg")
        Qe_st.buffer(len(amps)).average().save("Qeavg")
        # variances
        (
            ((Ig_st.buffer(len(amps)) * Ig_st.buffer(len(amps))).average())
            - (Ig_st.buffer(len(amps)).average() * Ig_st.buffer(len(amps)).average())
        ).save("Igvar")
        (
            ((Qg_st.buffer(len(amps)) * Qg_st.buffer(len(amps))).average())
            - (Qg_st.buffer(len(amps)).average() * Qg_st.buffer(len(amps)).average())
        ).save("Qgvar")
        (
            ((Ie_st.buffer(len(amps)) * Ie_st.buffer(len(amps))).average())
            - (Ie_st.buffer(len(amps)).average() * Ie_st.buffer(len(amps)).average())
        ).save("Ievar")
        (
            ((Qe_st.buffer(len(amps)) * Qe_st.buffer(len(amps))).average())
            - (Qe_st.buffer(len(amps)).average() * Qe_st.buffer(len(amps)).average())
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
    job = qmm.simulate(config, rr_opt_a, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(rr_opt_a)  # execute QUA program

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

    plt.figure()

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
            plt.plot(amps * readout_amp, SNR, ".-")
            plt.title("Readout optimization")
            plt.xlabel("Voltage [V]")
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
    plt.plot(amps * readout_amp, SNR, ".-")
    plt.title("Readout optimization")
    plt.xlabel("Voltage [V]")
    plt.ylabel("SNR")
