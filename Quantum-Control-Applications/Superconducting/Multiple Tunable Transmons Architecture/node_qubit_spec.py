# ==================== DEFINE NODE ====================
import nodeio

nodeio.context(name="QbSpecNode", description="does qubit spectroscopy")

inputs = nodeio.Inputs()
inputs.stream("state", units="JSON", description="state after res spec")

outputs = nodeio.Outputs()
outputs.define(
    "IQ",
    units="list",
    description="measured IQ data",
    retention=2,
)

nodeio.register()

# ==================== DRY RUN DATA ====================

from quam import QuAM

# set inputs data for dry-run of the node
inputs.set(state="quam_bootstrap_state.json")

# =============== RUN NODE STATE MACHINE ===============

"""
qubit_spec.py: performs the 1D qubit spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool

u = unit()

while nodeio.status.active:

    state = inputs.get("state")

    print("Doing qubits spectroscopy...")

    ###################
    # The QUA program #
    ###################
    num_qubits = 4

    n_avg = 1e2

    cooldown_time = 50 * u.us // 4  # 50 microseconds

    f_min = [40e6, 80e6, 130e6, 190e6]
    f_max = [70e6, 110e6, 150e6, 210e6]
    df = 0.05e6

    freqs = [np.arange(f_min[i], f_max[i] + 0.1, df) for i in range(num_qubits)]

    freqs = [freqs[i].tolist() for i in range(num_qubits)]

    with program() as qubit_spec:
        n = [declare(int) for _ in range(num_qubits)]
        n_st = [declare_stream() for _ in range(num_qubits)]
        f = declare(int)
        I = [declare(fixed) for _ in range(num_qubits)]
        Q = [declare(fixed) for _ in range(num_qubits)]
        I_st = [declare_stream() for _ in range(num_qubits)]
        Q_st = [declare_stream() for _ in range(num_qubits)]

        for i in range(num_qubits):
            with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
                with for_(
                    f, f_min[i], f <= f_max[i], f + df
                ):  # Notice it's <= to include f_max (This is only for integers!)
                    update_frequency(f"q{i}", f)
                    play("saturation", f"q{i}")
                    align(f"q{i}", f"rr{i}")
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
                I_st[i].buffer(len(freqs[i])).average().save(f"I{i}")
                Q_st[i].buffer(len(freqs[i])).average().save(f"Q{i}")
                n_st[i].save(f"iteration{i}")

    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host="172.16.2.103", port="85")

    #######################
    # Simulate or execute #
    #######################

    simulate = False

    machine = QuAM(state)
    config = machine.build_config()

    for _ in range(len(machine.qubits)):
        print(f"Previous q{_}", machine.qubits[_].f_01)

    if simulate:
        simulation_config = SimulationConfig(duration=1000)
        job = qmm.simulate(config, qubit_spec, simulation_config)
        job.get_simulated_samples().con1.plot()

    else:
        qm = qmm.open_qm(config)
        job = qm.execute(qubit_spec)

        # Get results from QUA program
        my_results = fetching_tool(job, data_list=["I0", "Q0", "iteration0"], mode="live")

        while job.result_handles.is_processing():
            # Fetch results
            I, Q, iteration = my_results.fetch_all()
            iteration1 = job.result_handles.get("iteration1").fetch_all()
            iteration2 = job.result_handles.get("iteration2").fetch_all()
            iteration3 = job.result_handles.get("iteration3").fetch_all()
            # Progress bar
            if iteration < n_avg - 1:
                progress_counter(iteration, n_avg, start_time=my_results.get_start_time())
            if (iteration1 is not None) and (iteration1 < n_avg - 1):
                progress_counter(iteration1, n_avg, start_time=my_results.get_start_time())
            if (iteration2 is not None) and (iteration2 < n_avg - 1):
                progress_counter(iteration2, n_avg, start_time=my_results.get_start_time())
            if (iteration3 is not None) and (iteration3 < n_avg - 1):
                progress_counter(iteration3, n_avg, start_time=my_results.get_start_time())

        my_results = fetching_tool(job, data_list=["iteration0", "iteration1", "iteration2", "iteration3"])
        iteration0, iteration1, iteration2, iteration3 = my_results.fetch_all()
        progress_counter(iteration0, n_avg)
        progress_counter(iteration1, n_avg)
        progress_counter(iteration2, n_avg)
        progress_counter(iteration3, n_avg)

        my_results = fetching_tool(job, data_list=["I0", "Q0"])
        I, Q = my_results.fetch_all()

        data = [I.tolist(), Q.tolist(), freqs]
        print("Qubits spectroscopy finished...")
        outputs.set(IQ=data)

        nodeio.terminate_workflow()
