from qm.qua import *
from configuration import *
from qm import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.optimize
from qualang_tools.bakery import baking
from filter_functions import expdecay, filter_calc

##############################
# Program-specific variables #
##############################
qubit_under_study = 1  # defines the qubit on which to do experiment
# Choose relevant elements
res_name = f"resonator{qubit_under_study}"
qbit_name = f"qubit{qubit_under_study}"
flux = f"flux_line{qubit_under_study}"
if qubit_under_study == 0:
    other_flux_element = f"flux_line{1}"
    flux_amp = 0.05
    # This threshold is further towards the ground IQ blob to increase the initialization fidelity
    initialization_threshold = -0.00025
else:
    other_flux_element = f"flux_line{0}"
    flux_amp = 0.05
    # This threshold is further towards the ground IQ blob to increase the initialization fidelity
    initialization_threshold = -0.00007

# FLux pulse waveform generation
flux_pulse = np.array([flux_amp] * const_flux_len)  # const_flux_len = 200 ns
zeros_before_pulse = 20  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = 20  # End of the flux pulse (after we put zeros to see the falling time)
flux_waveform = np.array([0.0] * zeros_before_pulse + list(flux_pulse) + [0.0] * zeros_after_pulse)


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()

            b.add_op("flux_pulse", flux, wf)
            b.play("flux_pulse", flux)
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
total_len = const_flux_len + zeros_before_pulse + zeros_after_pulse
square_pulse_segments = baked_waveform(flux_waveform, total_len)
step_response = [1.0] * const_flux_len
xplot = np.arange(0, total_len + 0.1, 1)
# Number of averages
n_avg = 1e3

###################
# The QUA program #
###################

with program() as cryoscope:
    n = declare(int)  # Variable for averaging
    n_st = declare_stream()
    I = declare(fixed)  # I quadrature for state measurement
    Q = declare(fixed)  # Q quadrature for state measurement
    state = declare(bool)  # Qubit state
    state_st = declare_stream()
    I_g = declare(fixed)  # I quadrature for qubit cooldown
    segment = declare(int)  # Flux pulse segment
    flag = declare(bool)  # Boolean flag to switch between x90 and y90 for state measurement

    # Set the flux line offset of the other qubit to 0
    set_dc_offset(other_flux_element, "single", 0)

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's <= to include t_max (This is only for integers!)
        with for_(segment, 0, segment <= total_len, segment + 1):
            with for_each_(flag, [True, False]):
                # Cooldown
                measure("readout", res_name, None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                with while_(I_g > initialization_threshold):
                    measure("readout", res_name, None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                align()
                wait(500)
                # Cryoscope protocol
                # Play the first pi/2 pulse
                play("x90", qbit_name)
                align(qbit_name, flux)
                # Play truncated flux pulse with 1ns resolution
                with switch_(segment):
                    for j in range(0, total_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run()
                # Wait some fixed time so that the whole protocol duration is constant
                wait(total_len // 4, qbit_name)
                # Play the second pi/2 pulse along x and y successively
                with if_(flag):
                    play("x90", qbit_name)
                with else_():
                    play("y90", qbit_name)
                # State readout
                align(qbit_name, res_name)
                measure(
                    "readout",
                    res_name,
                    None,
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                )
                # State discrimination
                if qubit_under_study == 0:
                    assign(state, I > ge_threshold)
                else:
                    assign(state, I > ge_threshold1)
                save(state, state_st)
        save(n, n_st)

    with stream_processing():
        state_st.boolean_to_int().buffer(2).buffer(total_len + 1).average().save("state")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager()
# Open quantum machine
qm = qmm.open_qm(config)
# Execute QUA program
job = qm.execute(cryoscope)
# Get results from QUA program
results = fetching_tool(job, data_list=["state", "iteration"], mode="live")
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
while results.is_processing():
    # Fetch results
    state, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Derive results
    Sxx = state[:, 0] * 2 - 1  # Bloch vector projection along X
    Syy = state[:, 1] * 2 - 1  # Bloch vector projection along Y
    S = Sxx + 1j * Syy  # Bloch vector
    # Qubit phase
    phase = np.unwrap(np.angle(S))
    phase = phase - phase[-1]
    # Qubit detuning
    detuning = signal.savgol_filter(
        phase[zeros_before_pulse : const_flux_len + zeros_before_pulse] / 2 / np.pi, 21, 2, deriv=1, delta=0.001
    )
    # Step response
    step_response_freq = detuning / np.average(detuning[-int(const_flux_len / 4)])
    step_response_volt = np.sqrt(detuning / np.average(detuning[-int(const_flux_len / 4)]))
    # plot results
    plt.subplot(121)
    plt.cla()
    plt.plot(xplot, Sxx, ".-", label="Sxx")
    plt.plot(xplot, Syy, ".-", label="Syy")
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Bloch vector components")
    plt.title("Cryoscope")
    plt.legend()
    plt.subplot(122)
    plt.cla()
    plt.plot(xplot[zeros_before_pulse : const_flux_len + zeros_before_pulse], detuning, ".-", label="Pulse")
    plt.title("Square pulse response")
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Qubit detuning [MHz]")
    plt.legend()


## Fit step response with exponential
[A, tau], _ = scipy.optimize.curve_fit(
    expdecay, xplot[zeros_before_pulse : const_flux_len + zeros_before_pulse], step_response_volt
)
print(f"A: {A}\ntau: {tau}")

## Derive IIR and FIR corrections
fir, iir = filter_calc(exponential=[(A, tau)])
print(f"FIR: {fir}\nIIR: {iir}")

## Derive responses and plots
# Ideal response
pulse = np.array([1.0] * const_flux_len)
# Response without filter
no_filter = expdecay(xplot, a=A, t=tau)
# Response with filters
with_filter = no_filter * signal.lfilter(fir, [1, iir[0]], pulse)  # Output filter , DAC Output

# Plot all data
plt.rcParams.update({"font.size": 13})
plt.figure()
plt.suptitle("Cryoscope with filter implementation")
plt.subplot(121)
plt.plot(xplot, step_response_volt, "o-", label="Data")
plt.plot(xplot, expdecay(xplot, A, tau), label="Fit")
plt.text(100, 0.95, f"A = {A:.2f}\ntau = {tau:.2f}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.axhline(y=1.01)
plt.axhline(y=0.99)
plt.xlabel("Flux pulse duration [ns]")
plt.ylabel("Step response")
plt.legend()

plt.subplot(122)
plt.plot()
plt.plot(no_filter, label="After Bias-T without filter")
plt.plot(with_filter, label="After Bias-T with filter")
plt.plot(pulse, label="Ideal WF")  # pulse
plt.plot(list(step_response_volt), label="Experimental data")
plt.text(40, 0.93, f"IIR = {iir}\nFIR = {fir}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.xlabel("Flux pulse duration [ns]")
plt.ylabel("Step response")
plt.legend(loc="upper right")
