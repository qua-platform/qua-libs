# %%
"""
        Pauli Spin Blockade search
The goal of the script is to find the PSB region according to the protocol described in xxx.
To do so, we fix the position in the charge stability diagram with the DC sources,
for example, it can be the charge transition region between (2,0) and (1,1). Then, we pulse
the fast AC lines with analog outputs from the OPX. The fast pulse navigates the following
regions: create a dephase/mixed state of S-T, then move along the detuning axis to cross
the (2,0) and (1,1) boundary, perform measurement along the detuning axis.

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
fast pulse so that the fast line of the bias-tees sees zero voltage in average. Otherwise, the high-pass filtering effect
of the bias-tee will distort accumulates charge over time and thus distortin the charge map. A function has been written for this.

In the current implementation, the OPX is also measuring (either with DC current sensing or RF-reflectometry) during the
readout window (last segment of the triangle).
The goal is to save_all data points and plot histogram of data.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the parameters of the external DC source using its driver.
    - Connect the two plunger gates (DC line of the bias-tee) to the external dc source.
    - Connect the OPX to the fast line of the plunger gates for playing the triangle pulse sequence.

Before proceeding to the next node:
    - Identify the PSB region and update the config.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
from macros import lock_in_macro
import matplotlib.pyplot as plt
from qm import generate_qua_script
import copy

###################
# The QUA program #
###################

p5_voltages = np.linspace(-0.1, 0.1, 20)
p6_voltages = np.linspace(-0.15, 0.15, 20)

buffer_len = len(p5_voltages)

# Points in the charge stability map [V1, V2]
level_dephasing = [-0.07, 0.25]
duration_dephasing = 2000  # nanoseconds

local_config = copy.deepcopy(config)

seq = OPX_virtual_gate_sequence(local_config, ["P5_sticky", "P6_sticky"])
seq.add_points("dephasing", level_dephasing, duration_dephasing)

n_shots = 100

with program() as PSB_search_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    x = declare(fixed)
    y = declare(fixed)

    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("QDS", I, Q)

    with for_(n, 0, n < n_shots, n + 1):

        save(n, n_st)

        with for_each_((x, y), (p5_voltages.tolist(), p6_voltages.tolist())):

            # Play fast pulse
            seq.add_step(voltage_point_name="dephasing", ramp_duration=dephasing_ramp)
            seq.add_step(duration=lock_in_readout_length, level=[x,y], ramp_duration=readout_ramp)  # duration in nanoseconds
            seq.add_compensation_pulse(duration=duration_compensation_pulse)
            # Ramp the voltage down to zero at the end of the triangle (needed with sticky elements)
            seq.ramp_to_zero()

            # Measure the dot right after the qubit manipulation
            wait((duration_dephasing + dephasing_ramp + readout_ramp) * u.ns, "QDS")
            lock_in_macro(I=I, Q=Q, I_st=I_st, Q_st=Q_st)

            align()

            # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
            # per µs to the stream processing.
            wait(1_000 * u.ns)  # in ns

    # Stream processing section used to process the data before saving it
    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(buffer_len).save_all("I")
        Q_st.buffer(buffer_len).save_all("Q")

# %%
#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(local_config, PSB_search_prog, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    # Open the quantum machine
    qm = qmm.open_qm(local_config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(PSB_search_prog)
    # Get results from QUA program and initialize live plotting
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, iteration = results.fetch_all()
        length_to_use = np.minimum(len(I), len(Q))
        # Convert results into Volts
        S = u.demod2volts(I[:length_to_use] + 1j * Q[:length_to_use], lock_in_readout_length)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_shots, start_time=results.start_time)
        R_flattened = R.flatten()
        voltages_expanded = np.tile(p5_voltages, length_to_use)
        plt.clf()
        plt.hist2d(voltages_expanded, R_flattened)
        plt.colorbar()
        plt.tight_layout()
        plt.pause(0.1)

    qm.close()
# %%
