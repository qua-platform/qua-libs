# %%
"""
        Initialization search vs duration at initialization
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.loops import from_array

from configuration import *
from qualang_tools.results import progress_counter, fetching_tool, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
from macros import lock_in_macro
import matplotlib.pyplot as plt
from qm import generate_qua_script
import copy

local_config = copy.deepcopy(config)

seq = OPX_virtual_gate_sequence(local_config, ["P5_sticky", "P6_sticky"])
seq.add_points("dephasing", level_dephasing, duration_dephasing)
seq.add_points("readout", level_readout, duration_readout)
seq.add_points("just_outsidePSB_20", [0.1, 0.1], duration_init_jumps)
# seq.add_points("just_outsidePSB_11", [0.1, 0.1], duration_init_jumps)
# seq.add_points("quick_return_11", [0.1, 0.1], duration_init_jumps)

n_shots = 100

p5_voltages = np.linspace(-0.1, 0.1, 20)
p6_voltages = np.linspace(-0.15, 0.15, 20)

times = np.arange(16, 2500, 40)

with program() as init_search_duration_prog:

    n = declare(int)  # QUA integer used as an index for the averaging loop
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    Id = declare(fixed)
    Qd = declare(fixed)
    Id_st = declare_stream()
    Qd_st = declare_stream()
    Ii = declare(fixed)
    Qi = declare(fixed)
    Ii_st = declare_stream()
    Qi_st = declare_stream()
    x = declare(fixed)
    y = declare(fixed)
    t = declare(int)

    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("QDS", Id, Qd, Ii, Qi)

    with for_(n, 0, n < n_shots, n + 1):

        save(n, n_st)

        with for_(*from_array(x, p5_voltages.tolist())):
            with for_(*from_array(t, times)):
                # MEASURE DEPHASE

                # Play fast pulse
                seq.add_step(voltage_point_name="dephasing", ramp_duration=dephasing_ramp)
                seq.add_step(voltage_point_name="readout", ramp_duration=readout_ramp)
                seq.add_compensation_pulse(duration=duration_compensation_pulse)
                # Ramp the voltage down to zero at the end of the triangle (needed with sticky elements)
                seq.ramp_to_zero()

                # Measure the dot right after the qubit manipulation
                wait((duration_dephasing + dephasing_ramp + readout_ramp) * u.ns, "QDS")
                lock_in_macro(I=Id, Q=Qd, I_st=Id_st, Q_st=Qd_st)

                align()

                # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
                # per µs to the stream processing.
                wait(1_000 * u.ns)  # in ns

                # MEASURE INITIALIZAION

                seq.add_step(duration=t, level=[x, 0.01], ramp_duration=init_ramp)  # duration in nanoseconds
                seq.add_step(voltage_point_name="just_outsidePSB_20", ramp_duration=init_ramp)
                # seq.add_step(voltage_point_name="just_outsidePSB_11", ramp_duration=init_ramp)
                # seq.add_step(voltage_point_name="quick_return_11", ramp_duration=init_ramp)
                seq.add_step(voltage_point_name="readout", ramp_duration=readout_ramp)
                seq.add_compensation_pulse(duration=duration_compensation_pulse)
                # Ramp the voltage down to zero at the end of the triangle (needed with sticky elements)
                seq.ramp_to_zero()

                # Measure the dot right after the qubit manipulation
                # wait((duration_init + duration_readout + readout_ramp + init_ramp*4 + duration_init_jumps*3) * u.ns, "QDS")
                wait((duration_init + init_ramp + duration_init_jumps + init_ramp) * u.ns, "QDS")
                lock_in_macro(I=Ii, Q=Qi, I_st=Ii_st, Q_st=Qi_st)

                align()

                # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
                # per µs to the stream processing.
                wait(1_000 * u.ns)  # in ns

    # Stream processing section used to process the data before saving it
    with stream_processing():
        n_st.save("iteration")
        Id_st.buffer(len(times)).buffer(len(p5_voltages)).average().save("Id")
        Qd_st.buffer(len(times)).buffer(len(p5_voltages)).average().save("Qd")
        Ii_st.buffer(len(times)).buffer(len(p5_voltages)).average().save("Ii")
        Qi_st.buffer(len(times)).buffer(len(p5_voltages)).average().save("Qi")

print(generate_qua_script(init_search_duration_prog))

# %%
        
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(local_config, init_search_duration_prog, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    # Open the quantum machine
    qm = qmm.open_qm(local_config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(init_search_duration_prog)
    # Get results from QUA program and initialize live plotting
    results = fetching_tool(job, data_list=["Id", "Qd", "Ii", "Qi", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        Id, Qd, Ii, Qi, iteration = results.fetch_all()
        # Convert results into Volts
        Sd = u.demod2volts(Id + 1j * Qd, lock_in_readout_length)
        Rd = np.abs(Sd)  # Amplitude
        phase_d = np.angle(Sd)  # Phase
        Si = u.demod2volts(Ii + 1j * Qi, lock_in_readout_length)
        Ri = np.abs(Si)  # Amplitude
        phase_i = np.angle(Si)  # Phase
        # Progress bar
        progress_counter(iteration, n_shots, start_time=results.start_time)
        plt.clf()
        plt.pcolor(times * 4, p5_voltages, Ri - Rd)
        plt.colorbar()
        plt.tight_layout()
        plt.pause(0.1)

    qm.close()