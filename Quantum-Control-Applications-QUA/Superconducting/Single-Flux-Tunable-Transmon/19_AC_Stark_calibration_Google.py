"""
        AC STARK-SHIFT CALIBRATION WITH DRAG PULSES (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses successively for different DRAG
detunings. Here the detuning sweep has to be performed in python, because it involves changing the DRAG waveforms in a
non-linear manner.
After such a sequence, the qubit is expected to always be in the ground state if the AC Stark shift is
properly compensated by the DRAG detuning.
One can then take a line cut for a given number of pulse and fit the 1D trace with a parabola to get the optimum
detuning and update its value in the configuration.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Having calibrated the DRAG coefficient.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the DRAG detuning (AC_stark_detuning) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
from macros import readout_macro
import matplotlib.pyplot as plt


###################
# The QUA program #
###################

n_avg = 100
# Detuning to compensate for the AC STark-shift
detunings = np.arange(-10e6, 10e6, 1e6)
# Scan the number of pulses
iter_min = 0
iter_max = 25
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)

with program() as ac_stark_shift:
    n = declare(int)  # QUA variable for the averaging loop
    it = declare(int)  # QUA variable for the number of qubit pulses
    pulses = declare(int)  # QUA variable for counting the qubit pulses
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)  # QUA variable for the qubit state
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(it, iters)):  # QUA for_ loop for sweeping the number of pulses
            # Loop for error amplification (perform many qubit pulses with varying DRAG coefficients)
            with for_(pulses, iter_min, pulses <= it, pulses + d):
                play("x180" * amp(1), "qubit")
                play("x180" * amp(-1), "qubit")
            # Align the two elements to measure after playing the qubit pulses.
            align("qubit", "resonator")
            # Measure the resonator and extract the qubit state
            state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)
            save(state, state_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I_st.buffer(len(iters)).average().save("I")
        Q_st.buffer(len(iters)).average().save("Q")
        state_st.boolean_to_int().buffer(len(iters)).average().save("state")

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
    job = qmm.simulate(config, ac_stark_shift, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    xaxis = []
    I_tot = []
    Q_tot = []
    state_tot = []
    plt.figure()
    # Since the DRAG waveforms need to be changed, we have to do it in Python
    for det in detunings:
        xaxis.append(det)
        # Derive the DRAG waveforms with the new detuning
        x180_wf, x180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                x180_amp, x180_len, x180_sigma, alpha=drag_coef, anharmonicity=anharmonicity, detuning=det
            )
        )
        x180_I_wf = x180_wf
        x180_Q_wf = x180_der_wf
        # Update the config
        config["waveforms"]["x180_I_wf"]["samples"] = x180_I_wf.tolist()
        config["waveforms"]["x180_Q_wf"]["samples"] = x180_Q_wf.tolist()
        # Open the quantum machine with the updated config
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(ac_stark_shift)
        # Get results from QUA program
        results = fetching_tool(job, data_list=["I", "Q", "state"])
        # Fetch results
        I, Q, state = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        I_tot.append(I)
        Q_tot.append(Q)
        state_tot.append(state)
        # Plot results
        plt.suptitle("AC stark shift calibration")
        plt.subplot(231)
        plt.cla()
        plt.pcolor(iters, xaxis, I_tot)
        plt.xlabel("# of x180-x180 pulses")
        plt.ylabel("Detuning [Hz]")
        plt.title("I [V]")
        plt.subplot(232)
        plt.cla()
        plt.pcolor(iters, xaxis, Q_tot)
        plt.xlabel("# of x180-x180 pulses")
        plt.ylabel("Detuning [Hz]")
        plt.title("Q [V]")
        plt.subplot(233)
        plt.cla()
        plt.pcolor(iters, xaxis, state_tot)
        plt.xlabel("# of x180-x180 pulses")
        plt.ylabel("Detuning [Hz]")
        plt.title("state")
        plt.subplot(212)
        plt.cla()
        plt.plot(xaxis, np.sum(I_tot, axis=1))
        plt.xlabel("DRAG detuning [Hz]")
        plt.ylabel("Sum along the iterations")
        plt.tight_layout()
        plt.pause(0.01)
    print(f"Optimal DRAG detuning = {xaxis[np.argmin(np.sum(I_tot, axis=1))]:.0f} Hz")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
