# %%
"""
        T1 at readout
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
import matplotlib.pyplot as plt
from qualang_tools.loops.loops import from_array
from qm import generate_qua_script
import copy

###################
# The QUA program #
###################

division_length = 250  # 1 microsecond so that it matches one period of 1 MHz drive
number_of_divisions = int((lock_in_readout_length) / (4 * division_length))
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

# Time axis for the plots at the end
x_plot = np.arange(division_length * 4, lock_in_readout_length + 1, division_length * 4)

local_config = copy.deepcopy(config)

seq = OPX_virtual_gate_sequence(local_config, ["P5_sticky", "P6_sticky"])
seq.add_points("dephasing", level_dephasing, duration_dephasing)
seq.add_points("readout", level_readout, duration_readout)

n_shots = 100

with program() as t1_at_readout:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    n_st = declare_stream()  # Stream for the iteration number (progress bar)

    I = declare(fixed, size=number_of_divisions)
    Q = declare(fixed, size=number_of_divisions)
    I_st = declare_stream()
    Q_st = declare_stream()

    ind = declare(int)

    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("QDS", I[0], Q[0])

    with for_(n, 0, n < n_shots, n + 1):

        save(n, n_st)

        # Play fast pulse
        seq.add_step(voltage_point_name="dephasing", ramp_duration=dephasing_ramp)
        seq.add_step(voltage_point_name="readout", ramp_duration=readout_ramp)  # duration in nanoseconds
        seq.add_compensation_pulse(duration=duration_compensation_pulse)
        # Ramp the voltage down to zero at the end of the triangle (needed with sticky elements)
        seq.ramp_to_zero()

        # Measure the dot right after the qubit manipulation
        wait((duration_dephasing + dephasing_ramp + readout_ramp) * u.ns, "QDS")

        measure(
            "readout",
            "QDS",
            None,
            demod.sliced("cos", I, division_length, "out2"),
            demod.sliced("sin", Q, division_length, "out2"),
        )

        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            save(I[ind], I_st)
            save(Q[ind], Q_st)

        align()

        # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
        # per µs to the stream processing.
        wait(1_000 * u.ns)  # in ns

    # Stream processing section used to process the data before saving it
    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(number_of_divisions).average().save("I")
        Q_st.buffer(number_of_divisions).average().save("Q")
        I_st.buffer(number_of_divisions).save_all("I_shots")
        Q_st.buffer(number_of_divisions).save_all("Q_shots")


# %%
#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(local_config, t1_at_readout, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    # Open the quantum machine
    qm = qmm.open_qm(local_config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(t1_at_readout)
    # Get results from QUA program and initialize live plotting
    results = fetching_tool(job, data_list=["I", "Q", "I_shots", "Q_shots", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, I_shots, Q_shots, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, division_length * 4)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_shots, start_time=results.start_time)
        plt.clf()
        plt.plot(x_plot, R)
        plt.xlabel('Readout time [ns]')
        plt.ylabel('Magnitude [V]')

    qm.close()
    S_shots = u.demod2volts(I_shots + 1j * Q_shots, division_length * 4)
    R_shots = np.abs(S_shots)  # Amplitude
    phase = np.angle(S_shots)  # Phase
    plt.figure()
    plt.pcolor(x_plot, range(n_shots), R_shots)
    plt.xlabel('Readout time [ns]')
    plt.ylabel('Shots')
        