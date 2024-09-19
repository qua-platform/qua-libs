from configuration import *
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
import matplotlib.pyplot as plt


# Section 1: Waveform generation
def supergaussian(length, order, cutoff):
    """
    Generate a super-Gaussian envelope.

    Parameters:
    - length (int): Number of points in the waveform.
    - order (int): The order of the super-Gaussian (controls the sharpness).
    - cutoff (float): The cutoff value for the super-Gaussian envelope.

    Returns:
    - numpy array: The super-Gaussian envelope.
    """
    x = np.linspace(-1, 1, length)
    return np.exp(x**order * np.log(cutoff))


def robust_wf(amp, length, mod=40e6, order=4, cutoff=1e-2):
    """
    Generate a robust waveform with phase modulation.

    Parameters:
    - amp (float): Amplitude of the pulse.
    - length (int): Number of points in the waveform.
    - mod (float): Modulation frequency in Hz (default 40 MHz).
    - order (int): Order of the super-Gaussian envelope.
    - cutoff (float): Cutoff value for the super-Gaussian envelope.

    Returns:
    - numpy array: The complex waveform with phase modulation applied.
    """
    robust_time = np.linspace(-length / 2, length / 2, length) * 1e-9
    robust_freq = np.linspace(-1, 1, length) * mod
    robust_envelope = supergaussian(length, order, cutoff)
    robust_rescaling = length / robust_envelope.sum()
    return robust_amp * robust_rescaling * robust_envelope * np.exp(1j * 2 * np.pi * robust_time * robust_freq)


# Parameters for the robust waveform
robust_len = 200
robust_amp = 0.2

# Select the type of pulse to generate using the match-case structure
pulse_flag = 2  # 0 for rectangular, 1 for super-Gaussian, 2 for robust pulse

# Generate the pulse based on the selected flag

if pulse_flag == 0:
    pulse = robust_wf(robust_amp, robust_len, mod=0, order=4, cutoff=1)  # rectangular pulse
elif pulse_flag == 1:
    pulse = robust_wf(robust_amp, robust_len, mod=0, order=4, cutoff=1e-2)  # super-Gaussian pulse
elif pulse_flag == 2:
    pulse = robust_wf(robust_amp, robust_len, mod=40e6, order=4, cutoff=1e-2)  # robust pulse

# Update the pulse waveforms in the config: I=real(pulse) and Q=imag(pulse)
config["elements"]["qubit"]["operations"].update({"robust_op": "robust_pulse"})
config["pulses"].update(
    {
        "robust_pulse": {
            "operation": "control",
            "length": robust_len,
            "waveforms": {
                "I": "robust_I_wf",
                "Q": "robust_Q_wf",
            },
        }
    }
)
config["waveforms"].update({"robust_I_wf": {"type": "arbitrary", "samples": np.real(pulse).tolist()}})
config["waveforms"].update({"robust_Q_wf": {"type": "arbitrary", "samples": np.imag(pulse).tolist()}})

# Plot the real and imaginary parts of the generated pulse
plt.plot(np.real(pulse))
plt.plot(np.imag(pulse))

# Section 2: QUA program
n_avg = 1000  # Number of averages
# Define parameters for frequency detuning sweeps
n_detuning = 321  # Number of detuning points
detuning_span = 160e6  # Total span of detuning in Hz
detuning_array = np.linspace(-detuning_span / 2, detuning_span / 2, n_detuning).astype(
    int
)  # Array of detuning values centered around zero

# Define parameters for amplitude sweeps
n_a = 101  # Number of amplitude points
a_array = np.linspace(0, 2 - 2**-16, n_a)  # Array of amplitude values ranging from 0 to just below 2

# Define the QUA program for a Rabi amplitude and frequency sweep experiment
with program() as rabi_amp_freq:
    n = declare(int)  # Declare an integer variable for loop iteration
    f = declare(int)  # Declare an integer variable for frequency detuning
    detuning = declare(int)  # Declare an integer variable for frequency detuning
    a = declare(fixed)  # Declare a fixed-point variable for amplitude
    I = declare(fixed)  # Declare a fixed-point variable for the I (in-phase) component of the signal
    Q = declare(fixed)  # Declare a fixed-point variable for the Q (quadrature) component of the signal

    I_st = declare_stream()  # Declare a stream to store I component data
    Q_st = declare_stream()  # Declare a stream to store Q component data

    # Outer loop over the number of averages
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over each amplitude value
        with for_each_(a, a_array.tolist()):
            # Loop over each detuning value
            with for_each_(detuning, detuning_array.tolist()):
                # Update the qubit frequency with the current detuning value
                update_frequency("qubit", detuning + qubit_IF)

                # Play the pulse with the current amplitude on the qubit
                play("robust_op" * amp(a), "qubit")

                # Align the qubit and resonator operations
                align("qubit", "resonator")

                # Perform dual demodulation measurement on the qubit signal
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                )

                # Save the I and Q measurement results to their respective streams
                save(I, I_st)
                save(Q, Q_st)

    # Stream processing block to buffer and average the measurement results
    with stream_processing():
        I_st.buffer(n_a, n_detuning).average().save("I")  # Buffer and average I data, then save
        Q_st.buffer(n_a, n_detuning).average().save("Q")  # Buffer and average Q data, then save


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, rabi_amp_freq, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi_amp_freq)
