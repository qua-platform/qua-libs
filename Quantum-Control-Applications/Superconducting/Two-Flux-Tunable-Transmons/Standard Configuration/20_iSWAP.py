"""
        iSWAP CHEVRON - 4ns granularity
The goal of this protocol is to find the parameters of the iSWAP gate between two flux-tunable qubits.
The protocol consists in flux tuning one qubit (the one with the highest frequency) so that it becomes resonant with the second qubit.
If one qubit is excited, then they will start swapping their states by exchanging one photon when they are on resonance.
The process can be seen as an energy exchange between |10> and |01>.

By scanning the flux pulse amplitude and duration, the iSWAP chevron can be obtained and post-processed to extract the
iSWAP gate parameters corresponding to half an oscillation so that the states are fully swapped (flux pulse amplitude
and interation time).

This version sweeps the flux pulse duration using real-time QUA, which means that the flux pulse can be arbitrarily long
but the step must be larger than 1 clock cycle (4ns) and the minimum pulse duration is 4 clock cycles (16ns).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having found the qubits maximum frequency point (qubit_spectroscopy_vs_flux).
    - Having calibrated qubit gates (x180) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the configuration.
    - (Optional) having corrected the flux line distortions by running the Cryoscope protocol and updating the filter taps in the configuration.

Next steps before going to the next node:
    - Update the iSWAP gate parameters in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
from macros import qua_declaration, multiplexed_readout


###################
# The QUA program #
###################
qubit_in_e = 2  # Qubit number to put in |e> at the beginning of the sequence
qubit_to_flux_tune = 1  # Qubit number to flux-tune

n_avg = 1300  # The number of averages
ts = np.arange(4, 200, 1)  # The flux pulse durations in clock cycles (4ns) - Must be larger than 4 clock cycles.
amps = np.arange(-0.315, -0.298, 0.0002) / const_flux_amp  # The flux amplitude pre-factor
# Flux offset
flux_bias = config["controllers"]["con1"]["analog_outputs"][
    config["elements"][f"q{qubit_to_flux_tune}_z"]["singleInput"]["port"][1]
]["offset"]


with program() as iswap:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)  # QUA variable for the flux pulse duration
    a = declare(fixed)  # QUA variable for the flux pulse amplitude pre-factor.

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, ts)):
            with for_(*from_array(a, amps)):
                # Put one qubit in the excited state
                play("x180", f"q{qubit_in_e}_xy")
                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                # Play a flux pulse on the qubit with the highest frequency to bring it close to the excited qubit while
                # varying its amplitude and duration in order to observe the SWAP chevron.
                play("const" * amp(a), f"q{qubit_to_flux_tune}_z", duration=t)
                align()
                # Wait some time to ensure that the flux pulse will end before the readout pulse
                wait(20 * u.ns)
                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()
                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(len(ts)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(len(ts)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(len(ts)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(len(ts)).average().save("Q2")

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
    job = qmm.simulate(config, iswap, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(iswap)
    # Prepare the figure for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Convert the results into Volts
        I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
        I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot
        plt.suptitle(f"SWAP chevron sweeping the flux on qubit {qubit_to_flux_tune}")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + flux_bias, 4 * ts, I1)
        plt.title("q1 - I [V]")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + flux_bias, 4 * ts, Q1)
        plt.title("q1 - Q [V]")
        plt.xlabel("Flux amplitude (V)")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + flux_bias, 4 * ts, I2)
        plt.title("q2 - I [V]")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + flux_bias, 4 * ts, Q2)
        plt.title("q2 - Q [V]")
        plt.xlabel("Flux amplitude (V)")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # np.savez(save_dir / 'iswap', I1=I1, Q1=Q1, I2=I2, ts=ts, amps=amps)
