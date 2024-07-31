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

aIs = np.arange(-2, 2, 0.05)
aQs = np.arange(-2, 2, 0.05)

df1 = 178.12 * u.MHz
test = True
with program() as ramsey:
    n = declare(int)  # QUA variable for the averaging loop
    aI = declare(fixed)  # QUA variable for the idle time
    aQ = declare(fixed)
    state = declare(int)
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
    state_st = declare_stream()
    state1_st = declare_stream()
    state2_st = declare_stream()
    n_st = declare_stream()

    # Shift the qubit drive frequency to observe Ramsey oscillations
    # update_frequency("qubit", qubit_IF + detuning)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(aI, aIs)):
            with for_(*from_array(aQ, aQs)):
                # Prepare storage cavity in Fock state n=1
                play("beta1" , "storage")
                align()
                play("x360", "qubit")
                align()
                play("beta2" , "storage")
                align()

                play("cw"*amp(aI,0,aQ, 0) , "storage")

                align()

                play("x90", "qubit")
                # Wait a varying idle time
                wait(t_parity, "qubit")
                # 2nd x90 gate
                play("x90", "qubit")

                # update_frequency("qubit", df1)

                align("qubit", "resonator")

                # Measure the state of the resonator
                state1, I1, Q1 = macros.readout_macro(threshold = ge_threshold, state=state1, I=I1, Q=Q1)

                # Wait for the qubit to decay to the ground state
                wait(s_thermalization_time * u.ns, "resonator")
                save(I1, I1_st)
                save(Q1, Q1_st)
                save(state1, state1_st)

                align()

                play("beta1" , "storage")
                align()
                # align("qubit", "storage")
                play("x360", "qubit")
                align()
                play("beta2" , "storage")
                align()

                play("cw"*amp(aI,0,aQ, 0) , "storage")

                align()

                play("x90", "qubit")
                # Wait a varying idle time
                wait(t_parity, "qubit")
                # 2nd x90 gate
                play("-x90", "qubit")

                # update_frequency("qubit", df1)

                align("qubit", "resonator")

                # Measure the state of the resonator
                state2, I2, Q2 = macros.readout_macro(threshold = ge_threshold, state=state2, I=I2, Q=Q2)

                # Wait for the qubit to decay to the ground state
                wait(s_thermalization_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I2, I2_st)
                save(Q2, Q2_st)
                save(state2, state2_st)

                assign(state, Cast.to_int(state1) - Cast.to_int(state2))
                save(state, state_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I1_st.buffer(len(aQs)).buffer(len(aIs)).average().save("I")
        Q1_st.buffer(len(aQs)).buffer(len(aIs)).average().save("Q")
        state_st.buffer(len(aQs)).buffer(len(aIs)).average().save("state")
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

        ax1[0].clear()
        ax1[1].clear()
        try:
            c10b.remove()
            c11b.remove()
        except:
            pass
        ax1[0].cla()
        c10=ax1[0].pcolor(aIs*2, aQs*2,I)
        ax1[0].set_ylabel("Imag")
        ax1[1].cla()
        c11=ax1[1].pcolor(aIs*2, aQs*2,Q)
        ax1[1].set_xlabel("Real")
        ax1[1].set_ylabel("Imag")
        c10b = fig1.colorbar(c10, ax=ax1[0])
        c11b = fig1.colorbar(c11, ax=ax1[1])
        plt.pause(0.1)
        plt.tight_layout()


        ax2.clear()
        try:
            c2b.remove()
        except:
            pass
        c2=ax2.pcolor(aIs*2, aQs*2, state)
        ax2.set_xlabel("Real")
        ax2.set_ylabel("Imag")
        # ax2.set_ylim(0,1)
        c2b = fig2.colorbar(c2, ax=ax2)



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
