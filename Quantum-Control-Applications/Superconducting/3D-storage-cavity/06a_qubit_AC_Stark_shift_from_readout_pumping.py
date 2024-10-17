"""
        QUBIT AC STARK SHIFT DUE TO READOUT PUMPING
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state, simultaneously with an off resonant readout pulse,
followed by a selective pi-pulse (x180_long) to qubit and measure across various qubit drive intermediate dfs.

The data is post-processed to determine the qubit AC start shift due to readout pumping.

Note that the pi-pulse should be long enough such that it will apply a pi-pulse only when the storage is at Fock state n=0.


Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the resonator drive line (whether it's an external mixer or an Octave port).
    - Identification of the qubit's resonance frequency (referred to as "qubit_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Specification of the measured thermalization_time of the storage in the configuration (referred to as "T1")

"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
import macros as macros

###################
# The QUA program #
###################
n_avg = 1000  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 100 * u.us  # In ns
saturation_amp = 0.005  # Pre-factor to the value defined in the config - restricted to [-2; 2)
# Adjust the pulse amplitude and frequency to the storage cavity
off_saturation_amp = 1  # Pre-factor to the value defined in the config - restricted to [-2; 2)
detuning = 20 * u.MHz  # Detuning frequency of the storage off pump pulse
# Qubit detuning sweep
center = 100 * u.MHz
span = 2 * u.MHz
df = 1 * u.kHz
dfs = np.arange(-span, +span + 0.1, df)


with program() as qubit_AC_stark_shift:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    state_st = declare_stream()
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the digital oscillator linked to the qubit element
            update_frequency("qubit", df + center)
            # Play the saturation pulse to put the qubit in a mixed state - Can adjust the amplitude on the fly [-2; 2)
            # simultaneously with the off pump pulse to the resonator (with a detuned frequency)
            update_frequency("resonator", resonator_IF - detuning)
            play("off_pump" * amp(off_saturation_amp), "resonator", duration=saturation_len * u.ns)
            play("saturation" * amp(saturation_amp), "qubit", duration=saturation_len * u.ns)
            # Align the two elements to measure after playing the storage pulse.
            # Measure the storage state by applying a selective pi-pulse to the qubit and measure the qubit state
            align("qubit", "resonator")
            update_frequency("resonator", resonator_IF)
            state, I, Q = macros.readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)

            # Wait for the qubit to decay to the ground state
            align("qubit", "resonator")
            wait(thermalization_time * u.ns, "qubit")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)
            save(state, state_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I_st.buffer(len(dfs)).average().save("I")
        Q_st.buffer(len(dfs)).average().save("Q")
        state_st.boolean_to_int().buffer(len(dfs)).average().save("state")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_AC_stark_shift, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(qubit_AC_stark_shift)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    # Live plotting
    fig1, ax1 = plt.subplots(2, 1)
    fig2, ax2 = plt.subplots(1, 1)
    interrupt_on_close(fig1, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, state, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle(f"Qubit AC Start Shift due to readout pumping - LO = {qubit_LO / u.GHz} GHz")
        plt.subplot(211)
        plt.cla()
        plt.plot((dfs + center) / u.MHz, R, ".")
        plt.xlabel("Qubit intermediate frequency [MHz]")
        plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.subplot(212)
        plt.cla()
        plt.plot((dfs + center) / u.MHz, phase, ".")
        plt.xlabel("Qubit intermediate frequency [MHz]")
        plt.ylabel("Phase [rad]")
        plt.pause(1)
        plt.tight_layout()

        ax2.clear()
        ax2.plot((dfs + center) / u.MHz, state, ".")
        ax2.set_ylabel(r"$P_e$")
        ax2.set_xlabel("Qubit intermediate frequency [MHz]")
        ax2.set_ylim(0, 1)
