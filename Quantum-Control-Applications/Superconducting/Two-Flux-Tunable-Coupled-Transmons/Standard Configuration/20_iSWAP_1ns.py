"""
        iSWAP CHEVRON - 1ns granularity
The goal of this protocol is to find the parameters of the iSWAP gate between two flux-tunable qubits.
The protocol consists in flux tuning one qubit (the one with the highest frequency) so that it becomes resonant with the second qubit.
If one qubit is excited, then they will start swapping their states by exchanging one photon when they are on resonance.
The process can be seen as an energy exchange between |10> and |01>.

By scanning the flux pulse amplitude and duration, the iSWAP chevron can be obtained and post-processed to extract the
iSWAP gate parameters corresponding to half an oscillation so that the states are fully swapped (flux pulse amplitude
and interation time).

This version sweeps the flux pulse duration using the baking tool, which means that the flux pulse can be scanned with
a 1ns resolution, but must be shorter than ~260ns. If you want to measure longer flux pulse, you can either reduce the
resolution (do 2ns steps instead of 1ns) or use the 4ns version (iSWAP.py).

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
from qualang_tools.bakery import baking
from qualang_tools.results.data_handler import DataHandler


##########
# baking #
##########
def baked_waveform(waveform, pulse_duration, flux_qubit):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", f"q{flux_qubit}_z", wf)
            b.play("flux_pulse", f"q{flux_qubit}_z")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


##################
#   Parameters   #
##################
# Parameters Definition
qubit_in_e = 2  # Qubit number to put in |e> at the beginning of the sequence
qubit_to_flux_tune = 1  # Qubit number to flux-tune

n_avg = 1300  # The number of averages
amps = np.arange(-0.315, -0.298, 0.0002) / const_flux_amp

# FLux pulse waveform generation
# The variable const_flux_len is defined in the configuration
flux_waveform = np.array([const_flux_amp] * const_flux_len)
# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, const_flux_len, qubit_to_flux_tune)
# Flux offset
flux_bias = config["controllers"]["con1"]["analog_outputs"][
    config["elements"][f"q{qubit_to_flux_tune}_z"]["singleInput"]["port"][1]
]["offset"]
xplot = np.arange(0, const_flux_len + 0.1, 1)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "amps": amps,
    "flux_waveform": flux_waveform,
    "flux_bias": flux_bias,
    "config": config,
}

###################
# The QUA program #
###################
with program() as iswap:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the flux pulse amplitude pre-factor.
    segment = declare(int)  # QUA variable for the flux pulse segment index

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            with for_(segment, 0, segment <= const_flux_len, segment + 1):
                # Put one qubit in the excited state
                play("x180", f"q{qubit_in_e}_xy")
                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                # Play a flux pulse on the qubit with the highest frequency to bring it close to the excited qubit while
                # varying its amplitude and duration in order to observe the SWAP chevron with 1ns resolution.
                with switch_(segment):
                    for j in range(0, const_flux_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run(amp_array=[(f"q{qubit_to_flux_tune}_z", a)])
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
        I_st[0].buffer(const_flux_len + 1).buffer(len(amps)).average().save("I1")
        Q_st[0].buffer(const_flux_len + 1).buffer(len(amps)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(const_flux_len + 1).buffer(len(amps)).average().save("I2")
        Q_st[1].buffer(const_flux_len + 1).buffer(len(amps)).average().save("Q2")

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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, iswap, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
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
        plt.pcolor(xplot, amps * const_flux_amp + flux_bias, I1.T)
        plt.title("q1 - I")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(xplot, amps * const_flux_amp + flux_bias, Q1.T)
        plt.title("q1 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(xplot, amps * const_flux_amp + flux_bias, I2.T)
        plt.title("q2 - I")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(xplot, amps * const_flux_amp + flux_bias, Q2.T)
        plt.title("q2 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I1_data": I1})
    save_data_dict.update({"Q1_data": Q1})
    save_data_dict.update({"I2_data": I2})
    save_data_dict.update({"Q2_data": Q2})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
