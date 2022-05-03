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


def opt_len(time):  # will update time in nanoseconds

    # update the configuration file
    ###############################
    config["pulses"]["readout_pulse"]["length"] = time
    config["integration_weights"]["cos_weights"]["cosine"] = [(1.0, time)]
    config["integration_weights"]["cos_weights"]["sine"] = [(0.0, time)]
    config["integration_weights"]["sin_weights"]["cosine"] = [(0.0, time)]
    config["integration_weights"]["sin_weights"]["sine"] = [(1.0, time)]
    config["integration_weights"]["minus_sin_weights"]["cosine"] = [(0.0, time)]
    config["integration_weights"]["minus_sin_weights"]["sine"] = [(-1.0, time)]
    config["integration_weights"]["rotated_cos_weights"]["cosine"] = [(np.cos(rotation_angle), time)]
    config["integration_weights"]["rotated_cos_weights"]["sine"] = [(-np.sin(rotation_angle), time)]
    config["integration_weights"]["rotated_sin_weights"]["cosine"] = [(np.sin(rotation_angle), time)]
    config["integration_weights"]["rotated_sin_weights"]["sine"] = [(np.cos(rotation_angle), time)]
    config["integration_weights"]["rotated_minus_sin_weights"]["cosine"] = [(-np.sin(rotation_angle), time)]
    config["integration_weights"]["rotated_minus_sin_weights"]["sine"] = [(-np.cos(rotation_angle), time)]

    ###################
    # The QUA program #
    ###################

    n_avg = 1000  # number of averages

    cooldown_time = 50000 // 4  # qubit decay time

    with program() as rr_time:

        # Declare QUA variables
        ###################
        n = declare(int)  # variable for average loop
        n_st = declare_stream()  # stream for 'n'
        I = declare(fixed)  # demodulated and integrated signal
        Q = declare(fixed)  # demodulated and integrated signal
        Ig_st = declare_stream()  # stream for Ig
        Qg_st = declare_stream()  # stream for Qg
        Ie_st = declare_stream()  # stream for Ie
        Qe_st = declare_stream()  # stream for Qe

        # Pulse sequence
        ################
        with for_(n, 0, n < n_avg, n + 1):

            # ground IQ blob
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

            # excited IQ blob
            wait(cooldown_time, "qubit")  # wait for qubit to decay
            play("pi", "qubit")  # to populate |e> state
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
            )
            save(I, Ie_st)
            save(Q, Qe_st)

            save(n, n_st)

        # Stream processing
        ###################
        with stream_processing():
            n_st.save("iteration")
            # mean values
            Ig_st.average().save("Igavg")
            Qg_st.average().save("Qgavg")
            Ie_st.average().save("Ieavg")
            Qe_st.average().save("Qeavg")
            # variances
            (((Ig_st * Ig_st).average()) - (Ig_st.average() * Ig_st.average())).save("Igvar")
            (((Qg_st * Qg_st).average()) - (Qg_st.average() * Qg_st.average())).save("Qgvar")
            (((Ie_st * Ie_st).average()) - (Ie_st.average() * Ie_st.average())).save("Ievar")
            (((Qe_st * Qe_st).average()) - (Qe_st.average() * Qe_st.average())).save("Qevar")

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
        job = qmm.simulate(config, rr_time, simulate_config)  # do simulation with qmm
        job.get_simulated_samples().con1.plot()  # visualize played pulses

    else:
        job = qm.execute(rr_time)  # execute QUA program

        res_handles = job.result_handles  # get access to handles
        res_handles.wait_for_all_values()

        Ig_avg = res_handles.get("Igavg").fetch_all()
        Qg_avg = res_handles.get("Qgavg").fetch_all()
        Ie_avg = res_handles.get("Ieavg").fetch_all()
        Qe_avg = res_handles.get("Qeavg").fetch_all()
        Ig_var = res_handles.get("Igvar").fetch_all()
        Qg_var = res_handles.get("Qgvar").fetch_all()
        Ie_var = res_handles.get("Ievar").fetch_all()
        Qe_var = res_handles.get("Qevar").fetch_all()

        Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
        var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
        SNR = ((np.abs(Z)) ** 2) / (2 * var)

        return SNR


###############
# Python loop #
###############


SNRs = np.zeros(4)
lens = np.zeros(4)
j = 0

for i in range(200, 1000, 200):
    SNRs[j] = opt_len(i)
    lens[j] = i
    j += 1

plt.figure()
plt.plot(lens, SNRs, ".-")
plt.xlabel("Readout lenght [ns]")
plt.ylabel("SNR")
plt.title("Readout optimization")
