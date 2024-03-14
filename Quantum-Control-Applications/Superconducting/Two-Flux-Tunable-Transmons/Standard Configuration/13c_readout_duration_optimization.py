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
    - Set the desired flux bias

Before proceeding to the next node:
    - Adjust the readout duration setting, labeled as "readout_len_q", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close


###################
# The QUA program #
###################

division_length = 10  # in clock cycles
number_of_divisions = int(readout_len / (4 * division_length))
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

n_avg = 1e4  # number of averages

with program() as ro_weights_opt:
    n = declare(int)  # QUA variable for the averaging loop
    ind = declare(int)  # QUA variable for the index used to save each element in the 'I' & 'Q' vectors
    II = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'II'
    IQ = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'IQ'
    QI = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'QI'
    QQ = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'QQ'
    I = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the full 'I'=II+IQ
    Q = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the full 'Q'=QI+QQ
    Ig_st = [declare_stream() for _ in range(2)]  # Stream for 'I' in the ground state
    Qg_st = [declare_stream() for _ in range(2)]  # Stream for 'Q' in the ground state
    Ie_st = [declare_stream() for _ in range(2)]  # Stream for 'I' in the excited state
    Qe_st = [declare_stream() for _ in range(2)]  # Stream for 'Q' in the excited state
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Measure the ground state.
        wait(thermalization_time * u.ns)
        # Loop over the two resonators
        for rr, res in enumerate([1, 2]):
            # With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
            measure(
                "readout",
                f"rr{res}",
                None,
                demod.accumulated("rotated_cos", II[rr], division_length, "out1"),
                demod.accumulated("rotated_sin", IQ[rr], division_length, "out2"),
                demod.accumulated("rotated_minus_sin", QI[rr], division_length, "out1"),
                demod.accumulated("rotated_cos", QQ[rr], division_length, "out2"),
            )
            # Save the QUA vectors to their corresponding streams
            with for_(ind, 0, ind < number_of_divisions, ind + 1):
                assign(I[rr][ind], II[rr][ind] + IQ[rr][ind])
                save(I[rr][ind], Ig_st[rr])
                assign(Q[rr][ind], QI[rr][ind] + QQ[rr][ind])
                save(Q[rr][ind], Qg_st[rr])

        # Measure the excited IQ blobs
        align()
        # Wait for the qubit to decay to the ground state
        wait(thermalization_time * u.ns)
        # Play the qubit drives
        play("x180", "q1_xy")
        play("x180", "q2_xy")
        align()
        # Loop over the two resonators
        for rr, res in enumerate([1, 2]):
            # With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
            measure(
                "readout",
                f"rr{res}",
                None,
                demod.accumulated("rotated_cos", II[rr], division_length, "out1"),
                demod.accumulated("rotated_sin", IQ[rr], division_length, "out2"),
                demod.accumulated("rotated_minus_sin", QI[rr], division_length, "out1"),
                demod.accumulated("rotated_cos", QQ[rr], division_length, "out2"),
            )
            # Save the QUA vectors to their corresponding streams
            with for_(ind, 0, ind < number_of_divisions, ind + 1):
                assign(I[rr][ind], II[rr][ind] + IQ[rr][ind])
                save(I[rr][ind], Ie_st[rr])
                assign(Q[rr][ind], QI[rr][ind] + QQ[rr][ind])
                save(Q[rr][ind], Qe_st[rr])

        # Wait for the qubit to decay to the ground state
        wait(thermalization_time * u.ns)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        # Loop over the two resonators
        for q in range(2):
            # mean values for |g> and |e>
            Ig_st[q].buffer(number_of_divisions).average().save(f"Ig_avg_q{q}")
            Qg_st[q].buffer(number_of_divisions).average().save(f"Qg_avg_q{q}")
            Ie_st[q].buffer(number_of_divisions).average().save(f"Ie_avg_q{q}")
            Qe_st[q].buffer(number_of_divisions).average().save(f"Qe_avg_q{q}")
            # variances for |g> and |e>
            (
                ((Ig_st[q].buffer(number_of_divisions) * Ig_st[q].buffer(number_of_divisions)).average())
                - (Ig_st[q].buffer(number_of_divisions).average() * Ig_st[q].buffer(number_of_divisions).average())
            ).save(f"Ig_var_q{q}")
            (
                ((Qg_st[q].buffer(number_of_divisions) * Qg_st[q].buffer(number_of_divisions)).average())
                - (Qg_st[q].buffer(number_of_divisions).average() * Qg_st[q].buffer(number_of_divisions).average())
            ).save(f"Qg_var_q{q}")
            (
                ((Ie_st[q].buffer(number_of_divisions) * Ie_st[q].buffer(number_of_divisions)).average())
                - (Ie_st[q].buffer(number_of_divisions).average() * Ie_st[q].buffer(number_of_divisions).average())
            ).save(f"Ie_var_q{q}")
            (
                ((Qe_st[q].buffer(number_of_divisions) * Qe_st[q].buffer(number_of_divisions)).average())
                - (Qe_st[q].buffer(number_of_divisions).average() * Qe_st[q].buffer(number_of_divisions).average())
            ).save(f"Qe_var_q{q}")

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
    job = qmm.simulate(config, ro_weights_opt, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_weights_opt)
    # Get results from QUA program
    data_list = [
        "Ig_avg_q0",
        "Qg_avg_q0",
        "Ie_avg_q0",
        "Qe_avg_q0",
        "Ig_var_q0",
        "Qg_var_q0",
        "Ie_var_q0",
        "Qe_var_q0",
    ] + [
        "Ig_avg_q1",
        "Qg_avg_q1",
        "Ie_avg_q1",
        "Qe_avg_q1",
        "Ig_var_q1",
        "Qg_var_q1",
        "Ie_var_q1",
        "Qe_var_q1",
        "iteration",
    ]
    results = fetching_tool(
        job,
        data_list=data_list,
        mode="live",
    )
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        (
            Ig_avg_q1,
            Qg_avg_q1,
            Ie_avg_q1,
            Qe_avg_q1,
            Ig_var_q1,
            Qg_var_q1,
            Ie_var_q1,
            Qe_var_q1,
            Ig_avg_q2,
            Qg_avg_q2,
            Ie_avg_q2,
            Qe_avg_q2,
            Ig_var_q2,
            Qg_var_q2,
            Ie_var_q2,
            Qe_var_q2,
            iteration,
        ) = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Derive the SNR
        ground_trace_q1 = Ig_avg_q1 + 1j * Qg_avg_q1
        excited_trace_q1 = Ie_avg_q1 + 1j * Qe_avg_q1
        var = (Ie_var_q1 + Qe_var_q1 + Ig_var_q1 + Qg_var_q1) / 4
        SNR_q1 = (np.abs(excited_trace_q1 - ground_trace_q1) ** 2) / (2 * var)
        ground_trace_q2 = Ig_avg_q2 + 1j * Qg_avg_q2
        excited_trace_q2 = Ie_avg_q2 + 1j * Qe_avg_q2
        var = (Ie_var_q2 + Qe_var_q2 + Ig_var_q2 + Qg_var_q2) / 4
        SNR_q2 = (np.abs(excited_trace_q2 - ground_trace_q2) ** 2) / (2 * var)
        # Plot results
        plt.subplot(231)
        plt.cla()
        plt.plot(ground_trace_q1.real, label="ground")
        plt.plot(excited_trace_q1.real, label="excited")
        plt.xlabel("Clock cycles [4ns]")
        plt.ylabel("demodulated traces [V]")
        plt.title("Real part qubit 1")
        plt.legend()
        plt.subplot(232)
        plt.cla()
        plt.plot(ground_trace_q2.imag, label="ground")
        plt.plot(excited_trace_q1.imag, label="excited")
        plt.xlabel("Clock cycles [4ns]")
        plt.title("Imaginary part qubit 1")
        plt.legend()
        plt.subplot(233)
        plt.cla()
        plt.plot(SNR_q1, ".-")
        plt.xlabel("Clock cycles [4ns]")
        plt.ylabel("SNR qubit 1")
        plt.title("SNR")
        # Qubit 2
        plt.subplot(234)
        plt.cla()
        plt.plot(ground_trace_q2.real, label="ground")
        plt.plot(excited_trace_q2.real, label="excited")
        plt.xlabel("Clock cycles [4ns]")
        plt.ylabel("demodulated traces [V]")
        plt.title("Real part qubit 2")
        plt.legend()
        plt.subplot(235)
        plt.cla()
        plt.plot(ground_trace_q2.imag, label="ground")
        plt.plot(excited_trace_q2.imag, label="excited")
        plt.xlabel("Clock cycles [4ns]")
        plt.title("Imaginary part qubit 2")
        plt.legend()
        plt.subplot(236)
        plt.cla()
        plt.plot(SNR_q2, ".-")
        plt.xlabel("Clock cycles [4ns]")
        plt.ylabel("SNR")
        plt.title("SNR qubit 2")
        plt.pause(0.1)
        plt.tight_layout()
        # Get the optimal readout length in ns
        opt_readout_length_q1 = int(np.round(np.argmax(SNR_q1) * division_length / 4) * 4 * 4)
        opt_readout_length_q2 = int(np.round(np.argmax(SNR_q2) * division_length / 4) * 4 * 4)
    print(f"The optimal readout length for qubit 1 is {opt_readout_length_q1} ns (SNR={max(SNR_q1)})")
    print(f"The optimal readout length for qubit 2 is {opt_readout_length_q2} ns (SNR={max(SNR_q2)})")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
