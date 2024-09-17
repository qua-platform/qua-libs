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
    name="07b_IQ_Blobs_G_E_F",
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

for q in qubits:
    # Check if an optimized GEF frequency exists
    if not hasattr(q, 'GEF_frequency_shift'):
        q.GEF_frequency_shift = 0

for q in qubits:
    # check if an EF_x180 operation exists
    if 'EF_x180' in q.xy.operations:
        GEF_operation = 'EF_x180'
    else:
        GEF_operation = "x180"
        
###################
# The QUA program #
###################
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    I_f, I_f_st, Q_f, Q_f_st, _, _ = qua_declaration(num_qubits=num_qubits)

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
        
        update_frequency(qubit.resonator.name, qubit.resonator.intermediate_frequency+ q.GEF_frequency_shift)

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

            wait(5*machine.thermalization_time * u.ns)
            align()
            qubit.xy.play('x180')
            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency -qubit.anharmonicity)
            qubit.xy.play(GEF_operation)
            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)                
            align()
            qubit.resonator.measure("readout", qua_vars=(I_f[i], Q_f[i]))
            align()
            save(I_f[i], I_f_st[i])
            save(Q_f[i], Q_f_st[i])       

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].save_all(f"I_g{i + 1}")
            Q_g_st[i].save_all(f"Q_g{i + 1}")
            I_e_st[i].save_all(f"I_e{i + 1}")
            Q_e_st[i].save_all(f"Q_e{i + 1}")
            I_f_st[i].save_all(f"I_f{i + 1}")
            Q_f_st[i].save_all(f"Q_f{i + 1}")

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
    job = qm.execute(iq_blobs)

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
node.results["results"] = {}

plot_indvidual = False
for q in qubits:
    I_g_cent, Q_g_cent = ds.I_g.sel(qubit=q.name).mean(dim="N"), ds.Q_g.sel(qubit=q.name).mean(dim="N")
    I_e_cent, Q_e_cent = ds.I_e.sel(qubit=q.name).mean(dim="N"), ds.Q_e.sel(qubit=q.name).mean(dim="N")
    I_f_cent, Q_f_cent = ds.I_f.sel(qubit=q.name).mean(dim="N"), ds.Q_f.sel(qubit=q.name).mean(dim="N")

    node.results["results"][q.name] = {}
    node.results["results"][q.name]["I_g_cent"] = float(I_g_cent)
    node.results["results"][q.name]["Q_g_cent"] = float(Q_g_cent)
    node.results["results"][q.name]["I_e_cent"] = float(I_e_cent)
    node.results["results"][q.name]["Q_e_cent"] = float(Q_e_cent)
    node.results["results"][q.name]["I_f_cent"] = float(I_f_cent)
    node.results["results"][q.name]["Q_f_cent"] = float(Q_f_cent)
    node.results["results"][q.name]["center_matrix"] = np.array([[I_g_cent, Q_g_cent], [I_e_cent, Q_e_cent], [I_f_cent, Q_f_cent]])

    confusion = np.zeros((3,3))
    for p, prep_state in enumerate(["g", "e", "f"]):
        dist_g = np.sqrt((I_g_cent - ds[f"I_{prep_state}"].sel(qubit=q.name))**2 + (Q_g_cent - ds[f"Q_{prep_state}"].sel(qubit=q.name))**2)
        dist_e = np.sqrt((I_e_cent - ds[f"I_{prep_state}"].sel(qubit=q.name))**2 + (Q_e_cent - ds[f"Q_{prep_state}"].sel(qubit=q.name))**2)
        dist_f = np.sqrt((I_f_cent - ds[f"I_{prep_state}"].sel(qubit=q.name))**2 + (Q_f_cent - ds[f"Q_{prep_state}"].sel(qubit=q.name))**2)
        dist = np.stack([dist_g, dist_e, dist_f], axis=0)
        counts = np.argmin(dist,axis= 0)
        confusion[p][0] = np.sum(counts == 0)/len(counts)
        confusion[p][1] = np.sum(counts == 1)/len(counts)
        confusion[p][2] = np.sum(counts == 2)/len(counts)
    node.results["results"][q.name]["confusion_matrix"] = confusion

# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):

    qn = qubit['qubit']
    ax.plot(1e3*ds.I_g.sel(qubit =qn) , 1e3* ds.Q_g.sel(qubit =qn) , ".", alpha=0.1, markersize=1)
    ax.plot(1e3 * ds.I_e.sel(qubit =qn) , 1e3 * ds.Q_e.sel(qubit =qn) , ".", alpha=0.1, markersize=1)
    ax.plot(1e3 * ds.I_f.sel(qubit =qn) , 1e3 * ds.Q_f.sel(qubit =qn) , ".", alpha=0.1, markersize=1)
    ax.plot(1e3 * node.results["results"][qn]["I_g_cent"], 1e3 * node.results["results"][qn]["Q_g_cent"], "o", c='C0', ms=3,  mec='k', label="G")
    ax.plot(1e3 * node.results["results"][qn]["I_e_cent"], 1e3 * node.results["results"][qn]["Q_e_cent"], "o", c='C1', ms=3,  mec='k', label="E")
    ax.plot(1e3 * node.results["results"][qn]["I_f_cent"], 1e3 * node.results["results"][qn]["Q_f_cent"], "o", c='C2', ms=3,  mec='k', label="F")
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
    ax.set_xticks([0, 1, 2], labels=["|g>", "|e>", "|f>"])
    ax.set_yticks([0, 1, 2], labels=["|g>", "|e>", "|f>"])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    for prep in range(3):
        for meas in range(3):
            color = "k" if prep == meas else "w"
            ax.text(
                meas, prep, f"{100 * confusion[prep, meas]:.1f}%", ha="center", va="center", color=color)
    ax.set_title(qubit['qubit'])

grid.fig.suptitle('g.s. and e.s. fidelities')
plt.tight_layout()
plt.show()
node.results['figure_fidelities'] = grid.fig

# %%
# todo: fix list state updating in Qualibrate
for qubit in qubits:
    qubit.resonator.gef_centers = node.results["results"][qubit.name]["center_matrix"].tolist()
    qubit.resonator.gef_confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%
