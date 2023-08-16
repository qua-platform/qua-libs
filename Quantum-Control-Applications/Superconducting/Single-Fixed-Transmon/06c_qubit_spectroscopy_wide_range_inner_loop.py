"""
        QUBIT SPECTROSCOPY OVER A WIDE RANGE (INNER LOOP)
Performs a wide 1D frequency sweep on the qubit, measuring the resonator while also sweeping an external LO source.
In this version, the external LO source is being swept in the inner loop in order to minimize noise.
The user will need to update the LO source frequency using the corresponding API at the bottom of the script
(lo_source.set_freq(freqs_external[i])).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy.py).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Set the saturation pulse amplitude and duration in the configuration to put the qubit in a mixed state.
"""
from time import sleep
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.loops import from_array
import warnings
warnings.filterwarnings("ignore")

###################
# The QUA program #
###################

n_avg = 100  # The number of averages
# The intermediate frequency sweep parameters
f_min = 50 * u.MHz
f_max = 300 * u.MHz
df = 1000 * u.kHz
frequencies = np.arange(f_min, f_max + 0.1, df)  # The intermediate frequency vector (+ 0.1 to add f_max to frequencies)

# The LO frequency sweep parameters
f_min_external = 3e9 - f_min
f_max_external = 4e9 - f_max
df_external = f_max - f_min
freqs_external = np.arange(f_min_external, f_max_external + 0.1, df_external)
frequency = np.array(np.concatenate([frequencies + freqs_external[i] for i in range(len(freqs_external))]))

with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    i = declare(int)  # QUA variable for the LO frequency sweep
    f = declare(int)  # QUA variable for the qubit frequency
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(i, 0, i < len(freqs_external), i + 1):
            pause()  # This waits until it is resumed from python
            with for_(*from_array(f, frequencies)):
                # Update the frequency of the digital oscillator linked to the qubit element
                update_frequency("qubit", f)
                # Play the saturation pulse to put the qubit in a mixed state
                play("saturation", "qubit")
                # Align the two elements to measure after playing the qubit pulse.
                # One can also measure the resonator while driving the qubit (2-tone spectroscopy) by commenting the 'align'
                align("qubit", "resonator")
                # Measure the state of the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(frequencies)).buffer(len(freqs_external)).average().save("I")
        Q_st.buffer(len(frequencies)).buffer(len(freqs_external)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

def wait_until_job_is_paused(current_job):
    """
    Waits until the OPX FPGA reaches the pause statement.
    Used when the OPX sequence needs to be synchronized with an external parameter sweep.

    :param current_job: the job object.
    """
    while not current_job.is_paused():
        sleep(0.1)
        pass
    return True

###############
# Run Program #
###############
# Open the quantum machine
qm = qmm.open_qm(config)
# Send the QUA program to the OPX, which compiles and executes it. It will stop at the 'pause' statement.
job = qm.execute(qubit_spec)
# Creates results handles to fetch the data
res_handles = job.result_handles
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
n_handle = res_handles.get("iteration")

# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

for i in range(n_avg):
    for j in range(len(freqs_external)):
        # Set the frequency of the LO source
        lo_source.set_freq(freqs_external[j])  # Replace by your own function and add time.sleep() if needed
        # Resume the QUA program (escape the 'pause' statement)
        job.resume()
        # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
        wait_until_job_is_paused(job)
    # Live plot
    I_handle.wait_for_values(1)
    Q_handle.wait_for_values(1)
    n_handle.wait_for_values(1)
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    iteration = n_handle.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg)
    # Convert results into Volts
    S = u.demod2volts(np.concatenate(I + 1j * Q), readout_len)
    R = np.abs(S)  # Amplitude
    phase = np.angle(S)  # Phase
    # Plot results
    ax1 = plt.suptitle("Qubit spectroscopy")
    plt.subplot(211)
    plt.cla()
    plt.plot(frequency / u.MHz, R, ".")
    plt.xlabel("qubit frequency [MHz]")
    plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(212, share=ax1)
    plt.cla()
    plt.plot(frequency / u.MHz, signal.detrend(np.unwrap(phase)), ".")
    plt.ylabel("Phase [rad]")
    plt.pause(0.1)
    plt.tight_layout()
