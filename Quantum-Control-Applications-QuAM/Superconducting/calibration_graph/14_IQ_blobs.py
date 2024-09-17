# %%
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
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e threshold (ge_threshold) in the state.
    - Save the current state by calling machine.save("quam")
"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_runs: int = 2000
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False

node = QualibrationNode(
    name="07a_IQ_Blobs",
    parameters_class=Parameters
)

node.parameters = Parameters()


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, active_reset

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
import xarray as xr

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]
num_qubits = len(qubits)
# %%
###################
# The QUA program #
###################
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
            
        for qb in qubits:
            wait(1000, qb.z.name)
        
        align()
        
        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            save(n, n_st)
            if reset_type == "active":
                active_reset(machine, qubit.name)
            elif reset_type == "thermal":
                wait(5*machine.thermalization_time * u.ns)
            else:
                raise ValueError(f"Unrecognized reset type {reset_type}.")

            qubit.align()
            qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
            align()
            # save data
            save(I_g[i], I_g_st[i])
            save(Q_g[i], Q_g_st[i])
            
            if reset_type == "active":
                active_reset(machine, qubit.name)
            elif reset_type == "thermal":
                wait(5*machine.thermalization_time * u.ns)
            else:
                raise ValueError(f"Unrecognized reset type {reset_type}.")
            align()
            qubit.xy.play('x180')
            align()
            qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
            align()
            save(I_e[i], I_e_st[i])
            save(Q_e[i], Q_e_st[i])

        align()
            

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    quit()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(iq_blobs, flags=['auto-element-thread'])

    for i in range(num_qubits):
        print(f"Fetching results for qubit {qubits[i].name}")
        data_list = sum([[f"I_g{i + 1}", f"Q_g{i + 1}",f"I_e{i + 1}", f"Q_e{i + 1}"] ], ["n"])
        results = fetching_tool(job, data_list, mode="live")
        while results.is_processing():
            fetched_data = results.fetch_all()
            n = fetched_data[0]
            progress_counter(n, n_runs, start_time=results.start_time)

    # # Fetch data
    # data_list = sum(
    #     [[f"I_g{i+1}", f"Q_g{i+1}", f"I_e{i+1}", f"Q_e{i+1}"] for i in range(num_qubits)],
    #     [],
    # )
    # results = fetching_tool(job, data_list)
    # fetched_data = results.fetch_all()
    # I_g_data = fetched_data[1::2]
    # Q_g_data = fetched_data[2::2]
    # I_e_data = fetched_data[3::2]
    # Q_e_data = fetched_data[4::2]
    # # Prepare for save data
    # data = {}
    # # Plot the results
    # figs = []
    # for i, qubit in enumerate(qubits):
    #     I_g = I_g_data[i]
    #     Q_g = Q_g_data[i]
    #     I_e = I_e_data[i]
    #     Q_e = Q_e_data[i]

    #     hist = np.histogram(I_g, bins=100)
    #     rus_threshold = hist[1][1:][np.argmax(hist[0])]
    #     angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I_g, Q_g, I_e, Q_e, True, b_plot=True)

    #     plt.suptitle(f"{qubit.name} - IQ Blobs")
    #     plt.axvline(rus_threshold, color="k", linestyle="--", label="Threshold")
    #     figs.append(plt.gcf())

    #     data[f"{qubit.name}_I_g"] = I_g
    #     data[f"{qubit.name}_Q_g"] = Q_g
    #     data[f"{qubit.name}_I_e"] = I_e
    #     data[f"{qubit.name}_Q_e"] = Q_e
    #     data[f"{qubit.name}"] = {
    #         "angle": angle,
    #         "threshold": threshold,
    #         "rus_exit_threshold": rus_threshold,
    #         "fidelity": fidelity,
    #         "confusion_matrix": [[gg, ge], [eg, ee]],
    #     }
    #     data[f"{qubit.name}_figure"] = figs[i]

    #     qubit.resonator.operations["readout"].integration_weights_angle -= angle
    #     qubit.resonator.operations["readout"].threshold = threshold
    #     qubit.resonator.operations["readout"].rus_exit_threshold = rus_threshold
    # plt.show()
    qm.close()

    # node_save(machine, "iq_blobs", data, additional_files=True)

# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"N": np.linspace(1, n_runs, n_runs)})


# Fix the structure of ds to avoid tuples
def extract_value(element):
    if isinstance(element, tuple):
        return element[0]
    return element
ds = xr.apply_ufunc(
    extract_value,
    ds,
    vectorize=True,  # This ensures the function is applied element-wise
    dask='parallelized',  # This allows for parallel processing
    output_dtypes=[float]  # Specify the output data type
)

node.results = {}
node.results['ds'] = ds

# %%
node.results["figs"] = {}
node.results["results"] = {}

plot_indvidual = False
for q in qubits:
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(ds.I_g.sel(qubit = q.name), ds.Q_g.sel(qubit = q.name), ds.I_e.sel(qubit = q.name), ds.Q_e.sel(qubit = q.name), True, b_plot=plot_indvidual)
    I_rot = ds.I_g.sel(qubit = q.name) * np.cos(angle) - ds.Q_g.sel(qubit = q.name) * np.sin(angle)
    hist = np.histogram(I_rot, bins=100)
    RUS_threshold = hist[1][1:][np.argmax(hist[0])]
    if plot_indvidual:
        fig = plt.gcf()
        plt.show()
        node.results["figs"][q.name] = fig
    node.results["results"][q.name] = {}
    node.results["results"][q.name]["angle"] = float(angle)
    node.results["results"][q.name]["threshold"] = float(threshold)
    node.results["results"][q.name]["fidelity"] = float(fidelity)
    node.results["results"][q.name]["confusion_matrix"] = np.array([[gg, ge], [eg, ee]])
    node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)

# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    n_avg = n_runs // 2
    qn = qubit['qubit']
    ax.plot(1e3*(ds.I_g.sel(qubit =qn) * np.cos(node.results["results"][qn]["angle"]) - ds.Q_g.sel(qubit =qn) * np.sin(node.results["results"][qn]["angle"])), 1e3*(ds.I_g.sel(qubit =qn) * np.sin(node.results["results"][qn]["angle"]) + ds.Q_g.sel(qubit =qn) * np.cos(node.results["results"][qn]["angle"])), ".", alpha=0.1, label="Ground", markersize=1)
    ax.plot(1e3 * (ds.I_e.sel(qubit =qn) * np.cos(node.results["results"][qn]["angle"]) - ds.Q_e.sel(qubit =qn) * np.sin(node.results["results"][qn]["angle"])), 1e3 * (ds.I_e.sel(qubit =qn) * np.sin(node.results["results"][qn]["angle"]) + ds.Q_e.sel(qubit =qn) * np.cos(node.results["results"][qn]["angle"])), ".", alpha=0.1, label="Excited", markersize=1)
    ax.axvline(1e3 * node.results["results"][qn]["rus_threshold"], color="k", linestyle="--", lw = 0.5, label="RUS Threshold")
    ax.axvline(1e3 * node.results["results"][qn]["threshold"], color="r", linestyle="--", lw = 0.5, label="Threshold")
    ax.axis("equal")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(qubit['qubit'])

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
grid.fig.suptitle('g.s. and e.s. discriminators (rotated)')
plt.tight_layout()
node.results['figure_IQ_blobs'] = grid.fig


grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    confusion = node.results["results"][qubit['qubit']]["confusion_matrix"]
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
    ax.set_title(qubit['qubit'])

grid.fig.suptitle('g.s. and e.s. fidelities')
plt.tight_layout()
plt.show()
node.results['figure_fidelities'] = grid.fig

# %%
with node.record_state_updates():
    for qubit in qubits:
        qubit.resonator.operations["readout"].integration_weights_angle -= float(node.results["results"][qubit.name]["angle"])
        qubit.resonator.operations["readout"].threshold = float(node.results["results"][qubit.name]["threshold"])
        # to add conf matrix  to the readout operation rather than the resonator
        qubit.resonator.operations["readout"].rus_exit_threshold = float(node.results["results"][qubit.name]["rus_threshold"])
        qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()

# %%
node.outcomes = {q.name: "successful" for q in qubits}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%