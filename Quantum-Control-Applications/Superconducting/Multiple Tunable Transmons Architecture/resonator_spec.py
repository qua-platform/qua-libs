"""
resonator_spec.py: performs the 1D resonator spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool

u = unit()

###################
# The QUA program #
###################

n_avg = 20000

cooldown_time = 10 * u.us // 4

f_min = 30e6
f_max = 70e6
df = 0.5e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

num_qubits = 4

with program() as resonator_spec:
    n = [declare(int) for _ in range(num_qubits)]
    n_st = [declare_stream() for _ in range(num_qubits)]
    f = declare(int)
    I = [declare(fixed) for _ in range(num_qubits)]
    Q = [declare(fixed) for _ in range(num_qubits)]
    I_st = [declare_stream() for _ in range(num_qubits)]
    Q_st = [declare_stream() for _ in range(num_qubits)]

    for i in range(num_qubits):
        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
                update_frequency(f"rr{i}", f)
                measure(
                    "readout",
                    f"rr{i}",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                )
                wait(cooldown_time, f"rr{i}")
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(num_qubits):
            I_st[i].buffer(len(freqs)).average().save(f"I{i}")
            Q_st[i].buffer(len(freqs)).average().save(f"Q{i}")
            n_st[i].save(f'iteration{i}')

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host='172.16.2.103', port='85')

#######################
# Simulate or execute #
#######################

simulate = False

machine = QuAM("quam_bootstrap_state.json")
config = machine.build_config()

if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)

    # Get results from QUA program
    my_results = fetching_tool(job, data_list=['I0', 'Q0', 'iteration0'], mode='live')

    while job.result_handles.is_processing():
        # Fetch results
        I, Q, iteration = my_results.fetch_all()
        iteration1 = job.result_handles.get('iteration1').fetch_all()
        iteration2 = job.result_handles.get('iteration2').fetch_all()
        iteration3 = job.result_handles.get('iteration3').fetch_all()
        # Progress bar
        if iteration < n_avg-1:
            progress_counter(iteration, n_avg, start_time=my_results.get_start_time())
        if (iteration1 is not None) and (iteration1 < n_avg - 1):
            progress_counter(iteration1, n_avg, start_time=my_results.get_start_time())
        if (iteration2 is not None) and (iteration2 < n_avg - 1):
            progress_counter(iteration2, n_avg, start_time=my_results.get_start_time())
        if (iteration3 is not None) and (iteration3 < n_avg - 1):
            progress_counter(iteration3, n_avg, start_time=my_results.get_start_time())

    my_results = fetching_tool(job, data_list=['iteration0', 'iteration1', 'iteration2', 'iteration3'])
    iteration0, iteration1, iteration2, iteration3 = my_results.fetch_all()
    progress_counter(iteration0, n_avg)
    progress_counter(iteration1, n_avg)
    progress_counter(iteration2, n_avg)
    progress_counter(iteration3, n_avg)

    my_results = fetching_tool(job, data_list=['I0', 'Q0'])
    I, Q = my_results.fetch_all()
    fig = plt.figure()
    # Plot results
    plt.subplot(211)
    plt.cla()
    plt.title("resonator spectroscopy amplitude")
    plt.plot(freqs / u.MHz, np.sqrt(I ** 2 + Q ** 2), ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
    plt.subplot(212)
    plt.cla()
    # detrend removes the linear increase of phase
    phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
    plt.title("resonator spectroscopy phase")
    plt.plot(freqs / u.MHz, phase, ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.pause(0.1)
    plt.tight_layout()