from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
import macros as macros 
import numpy as np
import scipy.special as sp
import scipy.optimize as spo


###################
# The QUA program #
###################
n_avg = 500  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
# Qubit detuning sweep
t_min = 16 // 4
t_max = 100000 // 4
dt = 400 // 4
durations = np.arange(t_min, t_max, dt)


df1 = 178.104 * u.MHz

detuning = 30 * u.MHz
with program() as qubit_spec:
    n = declare(int)  # QUA variable for the averaging loop
    t = declare(int)  # QUA variable for the qubit pulse duration
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, durations)):
            update_frequency("storage", s_IF)
            update_frequency("qubit", qubit_IF)
            # Prepare the Storage Cavity in Fock state n=1
            play("beta1" , "storage")
            align()
            align("qubit", "storage")
            play("x360", "qubit")
            align("qubit", "storage")
            play("beta2" , "storage")
            align()
            # Play two off resonance pulses. One to the storage cavity and another one to the resonator
            update_frequency("storage", s_IF - detuning)
            update_frequency("resonator", resonator_IF - detuning)
            play("off_pump", "storage", duration=t)
            play("off_pump", "resonator", duration=t)
            align()
            # Measure
            update_frequency("resonator", resonator_IF)
            update_frequency("qubit", df1)
            play("x180_long", "qubit")
            align("qubit", "resonator")
            # Measure the state of the resonator
            state, I, Q = macros.readout_macro(threshold = ge_threshold, state=state, I=I, Q=Q)

            # Wait for the qubit to decay to the ground state
            wait(s_thermalization_time * u.ns, "resonator")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)
            save(state, state_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I_st.buffer(len(durations)).average().save("I")
        I_st.buffer(len(durations)).save_all("I_single")
        state_st.boolean_to_int().buffer(len(durations)).average().save("state")
        Q_st.buffer(len(durations)).average().save("Q")
        Q_st.buffer(len(durations)).save_all("Q_single")
        n_st.save("iteration")

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
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(qubit_spec)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I","state", "Q", "iteration","I_single","Q_single"], mode="live")
    # Live plotting
    fig1, ax1 = plt.subplots(2,1)
    fig2, ax2 = plt.subplots(1,1)
    interrupt_on_close(fig1, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I,state, Q, iteration,I_single,Q_single = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        fig1.suptitle("Swapping operation")
        ax1[0].clear()
        ax1[1].clear()
        ax1[0].cla()
        ax1[0].plot(4 * durations, R, ".")
        ax1[0].set_xlabel("Swapping duration [ms]")
        ax1[0].set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        ax1[1].cla()
        ax1[1].plot(4 * durations, phase, ".")
        ax1[1].set_xlabel("Swapping duration [ms]")
        ax1[1].set_ylabel("Phase [rad]")
        plt.pause(0.1)
        plt.tight_layout()


        ax2.clear()
        ax2.plot(4 * durations, state, ".")
        ax2.set_ylabel(r"$P_e$")
        ax2.set_xlabel("Swapping duration [ms]")
        ax2.set_ylim(0,1)

def func(t, A, alpha, kappa,offset, n=0):
    return A*np.exp(-np.abs(alpha)**2*np.exp(-kappa*t))+offset




x0 = [-max(state)+min(state),3, 6, max(state)]
popt, pcov = spo.curve_fit(func, durations*4/u.ms, state, p0=x0)
print(popt)

fig3, ax3 = plt.subplots(1,1)

x = 4*np.linspace(4e-3, np.max(durations))/u.ms
ax3.plot(4 * durations/u.ms, state, ".")
ax3.plot(x, func(x,*popt))
ax3.plot(x, func(x, *x0))
ax3.set_ylabel(r"$P_e$")
ax3.set_xlabel("Pulse duration [ns]")
plt.show()