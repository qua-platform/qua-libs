"""
        READOUT OPTIMISATION: DURATION
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). The "demod.accumulated" method is employed to assess the state of the resonator over varying durations.
Reference: (https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/features/?h=accumulated#accumulated-demodulation)
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to determine
the Signal-to-Noise Ratio (SNR). The readout duration that offers the highest SNR is identified as the optimal choice.
Note: To observe the resonator's behavior during ringdown, the integration weights length should exceed the readout_pulse length.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit in focus (referred to as "resonator_spectroscopy").
    - Calibration of the qubit pi pulse (x180) using methods like qubit spectroscopy, rabi_chevron, and power_rabi,
      followed by an update to the configuration.
    - Calibration of both the readout frequency and amplitude, with subsequent configuration updates.

Before proceeding to the next node:
    - Adjust the readout duration setting, labeled as "readout_len", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
import matplotlib.pyplot as plt


####################
# Helper functions #
####################
def update_readout_length(new_readout_length, ringdown_length):
    config["pulses"]["readout_pulse"]["length"] = new_readout_length
    config["integration_weights"]["cosine_weights"] = {
        "cosine": [(1.0, new_readout_length + ringdown_length)],
        "sine": [(0.0, new_readout_length + ringdown_length)],
    }
    config["integration_weights"]["sine_weights"] = {
        "cosine": [(0.0, new_readout_length + ringdown_length)],
        "sine": [(1.0, new_readout_length + ringdown_length)],
    }
    config["integration_weights"]["minus_sine_weights"] = {
        "cosine": [(0.0, new_readout_length + ringdown_length)],
        "sine": [(-1.0, new_readout_length + ringdown_length)],
    }


###################
# The QUA program #
###################
n_avg = 1e4  # number of averages
# Set maximum readout duration for this scan and update the configuration accordingly
readout_len = 5 * u.us  # Readout pulse duration
ringdown_len = 0 * u.us  # integration time after readout pulse to observe the ringdown of the resonator
update_readout_length(readout_len, ringdown_len)
# Set the accumulated demod parameters
division_length = 10  # Size of each demodulation slice in clock cycles
number_of_divisions = int((readout_len + ringdown_len) / (4 * division_length))
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

# Time axis for the plots at the end
x_plot = np.arange(division_length * 4, readout_len + ringdown_len + 1, division_length * 4)


with program() as ro_duration_opt:
    n = declare(int)
    II = declare(fixed, size=number_of_divisions)
    IQ = declare(fixed, size=number_of_divisions)
    QI = declare(fixed, size=number_of_divisions)
    QQ = declare(fixed, size=number_of_divisions)
    I = declare(fixed, size=number_of_divisions)
    Q = declare(fixed, size=number_of_divisions)
    ind = declare(int)

    n_st = declare_stream()
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Measure the ground state.
        # With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
        measure(
            "readout",
            "resonator",
            None,
            demod.accumulated("cos", II, division_length, "out1"),
            demod.accumulated("sin", IQ, division_length, "out2"),
            demod.accumulated("minus_sin", QI, division_length, "out1"),
            demod.accumulated("cos", QQ, division_length, "out2"),
        )
        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], Ig_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Qg_st)
        # Wait for the qubit to decay to the ground state
        wait(thermalization_time * u.ns, "resonator")

        align()

        # Measure the excited state.
        # With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
        play("x180", "qubit")
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            demod.accumulated("cos", II, division_length, "out1"),
            demod.accumulated("sin", IQ, division_length, "out2"),
            demod.accumulated("minus_sin", QI, division_length, "out1"),
            demod.accumulated("cos", QQ, division_length, "out2"),
        )
        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], Ie_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Qe_st)

        # Wait for the qubit to decay to the ground state
        wait(thermalization_time * u.ns, "resonator")
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        # mean values
        Ig_st.buffer(number_of_divisions).average().save("Ig_avg")
        Qg_st.buffer(number_of_divisions).average().save("Qg_avg")
        Ie_st.buffer(number_of_divisions).average().save("Ie_avg")
        Qe_st.buffer(number_of_divisions).average().save("Qe_avg")
        # variances
        (
            ((Ig_st.buffer(number_of_divisions) * Ig_st.buffer(number_of_divisions)).average())
            - (Ig_st.buffer(number_of_divisions).average() * Ig_st.buffer(number_of_divisions).average())
        ).save("Ig_var")
        (
            ((Qg_st.buffer(number_of_divisions) * Qg_st.buffer(number_of_divisions)).average())
            - (Qg_st.buffer(number_of_divisions).average() * Qg_st.buffer(number_of_divisions).average())
        ).save("Qg_var")
        (
            ((Ie_st.buffer(number_of_divisions) * Ie_st.buffer(number_of_divisions)).average())
            - (Ie_st.buffer(number_of_divisions).average() * Ie_st.buffer(number_of_divisions).average())
        ).save("Ie_var")
        (
            ((Qe_st.buffer(number_of_divisions) * Qe_st.buffer(number_of_divisions)).average())
            - (Qe_st.buffer(number_of_divisions).average() * Qe_st.buffer(number_of_divisions).average())
        ).save("Qe_var")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_duration_opt, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_duration_opt)
    # Get results from QUA program
    results = fetching_tool(
        job,
        data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
        mode="live",
    )
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Derive the SNR
        ground_trace = Ig_avg + 1j * Qg_avg
        excited_trace = Ie_avg + 1j * Qe_avg
        var = (Ie_var + Qe_var + Ig_var + Qg_var) / 4
        SNR = (np.abs(excited_trace - ground_trace) ** 2) / (2 * var)
        # Plot results
        plt.subplot(221)
        plt.cla()
        plt.plot(x_plot, ground_trace.real, label="ground")
        plt.plot(x_plot, excited_trace.real, label="excited")
        plt.xlabel("Readout duration [ns]")
        plt.ylabel("demodulated traces [a.u.]")
        plt.title("Real part")
        plt.legend()

        plt.subplot(222)
        plt.cla()
        plt.plot(x_plot, ground_trace.imag, label="ground")
        plt.plot(x_plot, excited_trace.imag, label="excited")
        plt.xlabel("Readout duration [ns]")
        plt.title("Imaginary part")
        plt.legend()

        plt.subplot(212)
        plt.cla()
        plt.plot(x_plot, SNR, ".-")
        plt.xlabel("Readout duration [ns]")
        plt.ylabel("SNR")
        plt.title("SNR")
        plt.pause(0.1)
        plt.tight_layout()
    # Get the optimal readout length in ns
    opt_readout_length = int(np.round(np.argmax(SNR) * division_length / 4) * 4 * 4)
    print(f"The optimal readout length is {opt_readout_length} ns (SNR={max(SNR)})")
