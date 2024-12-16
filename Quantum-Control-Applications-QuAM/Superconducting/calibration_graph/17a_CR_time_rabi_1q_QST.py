# %%
"""
                                    Cross-Resonance Time Rabi
The sequence consists two consecutive pulse sequences with the qubit's thermal decay in between.
In the first sequence, we set the control qubit in |g> and play a rectangular cross-resonance pulse to
the target qubit; the cross-resonance pulse has a variable duration. In the second sequence, we initialize the control
qubit in |e> and play the variable duration cross-resonance pulse to the target qubit. Note that in
the second sequence after the cross-resonance pulse we send a x180_c pulse. With it, the target qubit starts
in |g> in both sequences when CR lenght -> zero.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Reference: A. D. Corcoles et al., Phys. Rev. A 87, 030301 (2013)

"""

# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["q1-2"]
    num_averages: int = 20
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 1000
    wait_time_step_in_ns: int = 16
    use_state_discrimination: bool = False
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="17a_CR_time_rabi_1q_QST", parameters=Parameters())


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import (
    qua_declaration,
    multiplexed_readout,
    node_save,
    active_reset,
    readout_state,
)

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()


# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_time_ns = np.arange(
    node.parameters.min_wait_time_in_ns,
    node.parameters.max_wait_time_in_ns,
    node.parameters.wait_time_step_in_ns,
) // 4 * 4
idle_time_cycles = idle_time_ns // 4


###################
#   QUA Program   #
###################

with program() as cr_time_rabi:
    n = declare(int)
    n_st = declare_stream()
    # I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    t = declare(int)
    s = declare(int)  # QUA variable for the control state
    c = declare(int)  # QUA variable for the projection index in QST

    for i, qp in enumerate(qubit_pairs):
        qc = qp.qubit_control
        qt = qp.qubit_target
        cr = qp.cross_resonance

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(t, idle_time_cycles)):
                with for_(c, 0, c < 3, c + 1):  # bases
                    with for_(s, 0, s < 2, s + 1):  # states
                        with if_(s == 1):
                            qc.xy.play("x180")
                            align(qc.xy.name, cr.name)

                        # Control
                        cr.play("square", duration=t)

                        align(qt.xy.name, cr.name)
                        with switch_(c):
                            with case_(0):  # projection along X
                                qt.xy.play("-y90")
                            with case_(1):  # projection along Y
                                qt.xy.play("x90")
                            with case_(2):  # projection along Z
                                qt.xy.wait(qt.xy.operations["x180"].length * u.ns)

                        align(qt.xy.name, qc.resonator.name, qt.resonator.name)

                        # Measure the state of the resonators
                        readout_state(qc, state_control[i])
                        readout_state(qt, state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])

                        # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                        wait(1 * u.us)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(2).buffer(3).buffer(len(idle_time_cycles)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(3).buffer(len(idle_time_cycles)).average().save(f"state_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cr_time_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    qm = qmm.open_qm(config, close_other_machines=True)
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(cr_time_rabi)

    results = fetching_tool(job, ["n"], mode="live")
    while results.is_processing():
        # Fetch results
        n = results.fetch_all()[0]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        
# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(
        job.result_handles,
        qubit_pairs,
        {"qc_state": ["0", "1"], "qt_component": ["X", "Y", "Z"], "times": idle_time_ns},
    )

    node.results = {"ds": ds}

    # %% {Live_plot}
    # Prepare the figure for live plotting
    for qp in qubit_pairs:
        fig, axss = plt.subplots(3, 2, figsize=(8, 8), sharex=True)
        # Plots
        plt.suptitle("non-echo CR Time Rabi")
        for i, (axs, bss) in enumerate(zip(axss, ["X", "Y", "Z"])):
            for stc in ["0", "1"]:
                ds_sliced = ds.sel(qubit=qp.name, qc_state=stc, qt_component=bss)
                axs[0].plot(ds_sliced.times.data, ds_sliced.state_control.data, label=[f"qc=|{stc}>"])
                axs[1].plot(ds_sliced.times.data, ds_sliced.state_target.data, label=[f"qc=|{stc}>"])
                axs[0].set_ylabel("<Z>")
                axs[1].set_ylabel(f"<{bss}>")
                axs[0].set_title(f"control: {qp.qubit_control.name}") if i == 0 else None
                axs[1].set_title(f"target: {qp.qubit_target.name}") if i == 0 else None
                for ax in axs:
                    ax.set_xlabel("cr durations [ns]") if i == 2 else None
                    # ax.legend(["0", "1"])
        plt.tight_layout()

    qm.close()
    print("Experiment QM is now closed")
    plt.show(block=True)


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()