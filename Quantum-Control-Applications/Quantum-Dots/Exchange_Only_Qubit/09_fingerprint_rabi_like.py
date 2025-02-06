# %%
"""
        Power-Time Rabi like / rotation along n
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
from macros import lock_in_macro
import matplotlib.pyplot as plt
from qm import generate_qua_script
import copy
from qualang_tools.loops.loops import from_array

local_config = copy.deepcopy(config)

level_init = [0.1, 0.1]
level_manipulation = [0.1, 0.1]
duration_init = 10_000
init_ramp = 100
duration_init_jumps = 16
manipulation_ramp = 100
duration_manipulation = step_length

seq = OPX_virtual_gate_sequence(local_config, ["P5_sticky", "P6_sticky"])
seq.add_points("dephasing", level_dephasing, duration_dephasing)
seq.add_points("readout", level_readout, duration_readout)
seq.add_points("initialization", level_init, duration_init)
seq.add_points("just_outsidePSB_20", [0.1, 0.1], duration_init_jumps)
seq.add_points("just_outsidePSB_11", [0.1, 0.1], duration_init_jumps)
seq.add_points("quick_return_11", [0.1, 0.1], duration_init_jumps)
seq.add_points("manipulation", [0.1, 0.1], duration_manipulation)

n_shots = 100

p5_voltages = np.linspace(-0.15, 0.15, 20)
p6_voltages = np.linspace(-0.15, 0.15, 20)

buffer_len = len(p5_voltages)

amps = np.arange(0, 1, 0.1)

with program() as finger_print:

    n = declare(int)  # QUA integer used as an index for the averaging loop
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    x = declare(fixed)
    y = declare(fixed)
    a = declare(fixed)
    t = declare(int)
    
    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("QDS", I, Q)

    with for_(n, 0, n < n_shots, n + 1):

        save(n, n_st)

        with for_each_((x, y), (p5_voltages.tolist(), p6_voltages.tolist())):
            
            with for_(*from_array(a, amps)):

                seq.add_step(voltage_point_name="initialization", ramp_duration=init_ramp)  # duration in nanoseconds
                seq.add_step(voltage_point_name="just_outsidePSB_20", ramp_duration=init_ramp)
                seq.add_step(voltage_point_name="just_outsidePSB_11", ramp_duration=init_ramp)
                seq.add_step(voltage_point_name="quick_return_11", ramp_duration=init_ramp)

                seq.add_step(duration=duration_manipulation, level=[x, y], ramp_duration=manipulation_ramp)  # to manipulate the barrier

                seq.add_step(voltage_point_name="readout", ramp_duration=readout_ramp)
                seq.add_compensation_pulse(duration=duration_compensation_pulse)
                # Ramp the voltage down to zero at the end of the triangle (needed with sticky elements)
                seq.ramp_to_zero()

                # pulse the barrier
                wait((duration_init + init_ramp*3 + manipulation_ramp) * u.ns, "X4")
                play("step"*amp(a), "X4")

                # Measure the dot right after the qubit manipulation
                wait((duration_init + readout_ramp + init_ramp*3 + duration_init_jumps*3 + manipulation_ramp) * u.ns, "QDS")
                lock_in_macro(I=I, Q=Q, I_st=I_st, Q_st=Q_st)

                align()

                # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
                # per µs to the stream processing.
                wait(1_000 * u.ns)  # in ns

    # Stream processing section used to process the data before saving it
    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(len(amps)).buffer(buffer_len).average().save("I")
        Q_st.buffer(len(amps)).buffer(buffer_len).average().save("Q")

print(generate_qua_script(finger_print))

# %%
        
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(local_config, finger_print, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    # Open the quantum machine
    qm = qmm.open_qm(local_config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(finger_print)
    # Get results from QUA program and initialize live plotting
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, lock_in_readout_length)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_shots, start_time=results.start_time)
        plt.clf()
        plt.pcolor(p5_voltages * np.sqrt(2), amps, R)
        plt.colorbar()
        plt.tight_layout()
        plt.xlabel('Detuning voltage [V]')
        plt.ylabel('X4 scaling amp [a.u.]')
        plt.pause(0.1)

    qm.close()