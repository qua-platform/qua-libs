"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit.wait(qubit.thermalization_time * u.ns) spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e thresholds (threshold & rus_threshold) in the state.
    - Update the confusion matrices in the state.
    - Save the current state
"""


# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.experiments.iq_blobs.fetch_dataset import fetch_dataset
from quam_libs.experiments.iq_blobs.parameters import Parameters
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *


# %% {Node_parameters}
node = QualibrationNode(
    name="07b_IQ_Blobs",
    parameters=Parameters(
        qubits=None,
        multiplexed=True,
        flux_point_joint_or_independent="joint",
        num_runs=2000,
        load_data_id=None,
        simulate=False,
        simulation_duration_ns=1000,
        use_waveform_report=False
    )
)

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()

if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

config = machine.generate_config()

# %% {QUA_program}
n_runs = node.parameters.num_runs
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
operation_name = node.parameters.operation_name

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    for multiplexed_qubits in qubits.batch():
        machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])

        with for_(n, 0, n < n_runs, n + 1):

            # measure ground-state IQ blob for all qubits
            for i, qubit in multiplexed_qubits.items():
                save(n, n_st)
                if reset_type == "active":
                    active_reset(qubit, "readout")
                elif reset_type == "thermal":
                    qubit.wait(4 * qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

            align()
            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.measure(operation_name, qua_vars=(I_g[i], Q_g[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

            # measure excited-state IQ blob for all qubits
            align()
            for i, qubit in multiplexed_qubits.items():
                if reset_type == "active":
                    active_reset(qubit, "readout")
                elif reset_type == "thermal":
                    qubit.wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

            align()
            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play("x180")
                qubit.align()
                qubit.resonator.measure(operation_name, qua_vars=(I_e[i], Q_e[i]))
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, iq_blobs, node.parameters)
    node.results = {"figure": fig}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs)
        for i in range(num_qubits):
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_runs, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # todo: Write docstring
        ds = fetch_dataset(job, qubits, node.parameters)
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    
    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    plot_individual = False
    for q in qubits:
        # Perform two state discrimination
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            ds.I_g.sel(qubit=q.name),
            ds.Q_g.sel(qubit=q.name),
            ds.I_e.sel(qubit=q.name),
            ds.Q_e.sel(qubit=q.name),
            True,
            b_plot=plot_individual,
        )
        # TODO: check the difference between the above and the below
        # Get the rotated 'I' quadrature
        I_rot = ds.I_g.sel(qubit=q.name) * np.cos(angle) - ds.Q_g.sel(qubit=q.name) * np.sin(angle)
        # Get the blobs histogram along the rotated axis
        hist = np.histogram(I_rot, bins=100)
        # Get the discriminating threshold along the rotated axis
        RUS_threshold = hist[1][1:][np.argmax(hist[0])]
        # Save the individual figures if requested
        if plot_individual:
            fig = plt.gcf()
            plt.show()
            node.results["figs"][q.name] = fig
        node.results["results"][q.name] = {}
        node.results["results"][q.name]["angle"] = float(angle)
        node.results["results"][q.name]["threshold"] = float(threshold)
        node.results["results"][q.name]["fidelity"] = float(fidelity)
        node.results["results"][q.name]["confusion_matrix"] = np.array([[gg, ge], [eg, ee]])
        node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        n_avg = n_runs // 2
        qn = qubit["qubit"]
        # TODO: maybe wrap it up in a function plot_IQ_blobs?
        ax.plot(
            1e3
            * (
                ds.I_g.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"])
                - ds.Q_g.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"])
            ),
            1e3
            * (
                ds.I_g.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"])
                + ds.Q_g.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"])
            ),
            ".",
            alpha=0.2,
            label="Ground",
            markersize=1,
        )
        ax.plot(
            1e3
            * (
                ds.I_e.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"])
                - ds.Q_e.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"])
            ),
            1e3
            * (
                ds.I_e.sel(qubit=qn) * np.sin(node.results["results"][qn]["angle"])
                + ds.Q_e.sel(qubit=qn) * np.cos(node.results["results"][qn]["angle"])
            ),
            ".",
            alpha=0.2,
            label="Excited",
            markersize=1,
        )
        ax.axvline(
            1e3 * node.results["results"][qn]["rus_threshold"],
            color="k",
            linestyle="--",
            lw=0.5,
            label="RUS Threshold",
        )
        ax.axvline(
            1e3 * node.results["results"][qn]["threshold"],
            color="r",
            linestyle="--",
            lw=0.5,
            label="Threshold",
        )
        ax.axis("equal")
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        ax.set_title(qubit["qubit"])

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    grid.fig.suptitle("g.s. and e.s. discriminators (rotated)")
    plt.tight_layout()
    node.results["figure_IQ_blobs"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        confusion = node.results["results"][qubit["qubit"]]["confusion_matrix"]
        ax.imshow(confusion)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels=["|g>", "|e>"])
        ax.set_yticklabels(labels=["|g>", "|e>"])
        ax.set_ylabel("Prepared")
        ax.set_xlabel("Measured")
        ax.text(0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", color="k")
        ax.text(1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", color="w")
        ax.text(0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", color="w")
        ax.text(1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", color="k")
        ax.set_title(qubit["qubit"])

    grid.fig.suptitle("g.s. and e.s. fidelity")
    plt.tight_layout()
    plt.show()
    node.results["figure_fidelity"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                qubit.resonator.operations[operation_name].integration_weights_angle -= float(
                    node.results["results"][qubit.name]["angle"]
                )
                # Convert the thresholds back in demod units
                qubit.resonator.operations[operation_name].threshold = (
                    float(node.results["results"][qubit.name]["threshold"])
                    * qubit.resonator.operations[operation_name].length
                    / 2**12
                )
                # todo: add conf matrix to the readout operation rather than the resonator
                qubit.resonator.operations[operation_name].rus_exit_threshold = (
                    float(node.results["results"][qubit.name]["rus_threshold"])
                    * qubit.resonator.operations[operation_name].length
                    / 2**12
                )
                if operation_name == "readout":
                    qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

