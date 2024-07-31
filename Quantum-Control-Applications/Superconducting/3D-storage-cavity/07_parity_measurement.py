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
n_avg = 1000
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
tau_min = 16 // 4
tau_max = 1000 // 4
d_tau = 4 // 4
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus
# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
detuning = 0 * u.MHz  # in Hz

df1 = (178.12-1.5)* u.MHz
test = True
with program() as ramsey:
    n = declare(int)  # QUA variable for the averaging loop
    tau = declare(int)  # QUA variable for the idle time
    state1 = declare(bool)
    state2 = declare(bool)
    I1 = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q1 = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I1_st = declare_stream()  # Stream for the 'I' quadrature
    Q1_st = declare_stream()  # Stream for the 'Q' quadrature
    I2 = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q2 = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I2_st = declare_stream()  # Stream for the 'I' quadrature
    Q2_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state1_st = declare_stream()
    state2_st = declare_stream()

    # Shift the qubit drive frequency to observe Ramsey oscillations
    # update_frequency("qubit", qubit_IF + detuning)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(tau, taus)):
            play("cw" , "storage")

            align()

            play("x90", "qubit")
            # Wait a varying idle time
            wait(tau, "qubit")
            # 2nd x90 gate
            play("-x90", "qubit")

            align("qubit", "resonator")

            # Measure the state of the resonator
            state1, I1, Q1 = macros.readout_macro(threshold = ge_threshold, state=state1, I=I1, Q=Q1)

            # Wait for the qubit to decay to the ground state
            wait(s_thermalization_time * u.ns, "resonator")    
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I1, I1_st)
            save(Q1, Q1_st)
            save(state1, state1_st)

            align()


            update_frequency("qubit", qubit_IF)
            play("cw" , "storage")

            align()

            play("x90", "qubit")
            # Wait a varying idle time
            wait(tau, "qubit")
            # 2nd x90 gate
            play("-x90", "qubit")

            # measure the parity of n=1
            update_frequency("qubit", df1)

            play("x180_long", "qubit")
            align("qubit", "resonator")

            state2, I2, Q2 = macros.readout_macro(threshold = ge_threshold, state=state2, I=I2, Q=Q2)

            # Wait for the storage cavity to decay to the ground state
            wait(s_thermalization_time * u.ns, "resonator")    
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I2, I2_st)
            save(Q2, Q2_st)
            save(state2, state2_st)

        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        (I1_st.buffer(len(taus))-I2_st.buffer(len(taus))).average().save("I")
        (Q1_st.buffer(len(taus))-Q2_st.buffer(len(taus))).average().save("Q")
        (state1_st.boolean_to_int().buffer(len(taus))-state2_st.boolean_to_int().buffer(len(taus))).average().save("state")

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
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ramsey)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration", "state"], mode="live")
    # Live plotting
    fig1, ax1 = plt.subplots(2,1)
    fig2, ax2 = plt.subplots(1,1)
    interrupt_on_close(fig1, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration, state = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        fig1.suptitle(f"Parity measurement")
        ax1[0].clear()
        ax1[1].clear()
        ax1[0].plot(4 * taus, I, ".")
        ax1[0].set_ylabel("I quadrature [V]")
        ax1[1].plot(4 * taus, Q, ".")
        ax1[1].set_xlabel("Idle time [ns]")
        ax1[1].set_ylabel("Q quadrature [V]")
        plt.pause(0.1)
        plt.tight_layout()

        ax2.clear()
        ax2.plot(4 * taus, state, ".")
        ax2.set_ylabel(r"$P_e$")
        ax2.set_xlabel("Idle time [ns]")
        # ax2.set_ylim(0,1)


    # Fit the results to extract the qubit frequency and T2*
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        ramsey_fit = fit.ramsey(4 * taus, state, plot=True)
        qubit_T2 = np.abs(ramsey_fit["T2"][0])
        qubit_detuning = ramsey_fit["f"][0] * u.GHz - detuning
        plt.xlabel("Idle time [ns]")
        plt.ylabel("I quadrature [V]")
        print(f"Qubit detuning to update in the config: qubit_IF += {-qubit_detuning:.0f} Hz")
        print(f"T2* = {qubit_T2:.0f} ns")
        plt.legend((f"detuning = {-qubit_detuning / u.kHz:.3f} kHz", f"T2* = {qubit_T2:.0f} ns"))
        plt.title("Ramsey measurement with detuned gates")
        print(f"Detuning to add: {-qubit_detuning / u.kHz:.3f} kHz")
    except (Exception,):
        pass
