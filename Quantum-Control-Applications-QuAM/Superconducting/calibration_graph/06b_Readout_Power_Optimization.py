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
# TODO: this script isn't working great, the readout amp found at the end isn't always correct maybe because of SNR...

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.analysis import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_runs: int = 2000
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    start_amp: float = 0.5
    end_amp: float = 1.99
    num_amps: int = 10
    outliers_threshold: float = 0.98

node = QualibrationNode(
    name="06b_Readout_Power_Optimization",
    parameters=Parameters()
)

from sklearn.mixture import GaussianMixture


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
# %%

n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
amps = np.linspace(node.parameters.start_amp, node.parameters.end_amp, node.parameters.num_amps)

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    a = declare(fixed)
    
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
                    # Wait for the flux bias to settle
        # Wait for the flux bias to settle
        for qb in qubits:
            wait(1000, qb.z.name)
        
        align()
        
        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            save(n, n_st)
            with for_(*from_array(a, amps)):
                if reset_type == "active":
                    active_reset(machine, qubit.name)
                elif reset_type == "thermal":
                    wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

                qubit.align()
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]), amplitude_scale=a)
                align()
                # save data
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])
                
                if reset_type == "active":
                    active_reset(machine, qubit.name)
                elif reset_type == "thermal":
                    wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")
                align()
                qubit.xy.play('x180')
                align()
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]), amplitude_scale=a)
                align()
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

        align()
            

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(len(amps)).buffer(n_runs).save(f"I_g{i + 1}")
            Q_g_st[i].buffer(len(amps)).buffer(n_runs).save(f"Q_g{i + 1}")
            I_e_st[i].buffer(len(amps)).buffer(n_runs).save(f"I_e{i + 1}")
            Q_e_st[i].buffer(len(amps)).buffer(n_runs).save(f"Q_e{i + 1}")





if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
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

    # node_save(machine, "iq_blobs", data, additional_files=True)

# %%
if not node.parameters.simulate:
    # %% {Data_fetching_and_dataset_creation}
    
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"amplitude": amps, "N": np.linspace(1, n_runs, n_runs)})

    def abs_amp(q):
        def foo(amp):
            return amp * q.resonator.operations["readout"].amplitude
        return foo

    ds = ds.assign_coords({'readout_amp' : (['qubit','amplitude'],np.array([abs_amp(q)(amps) for q in qubits]))})

    # Rearrange the data to combine I_g and I_e into I, and Q_g and Q_e into Q
    ds_rearranged = xr.Dataset()

    # Combine I_g and I_e into I
    ds_rearranged['I'] = xr.concat([ds.I_g, ds.I_e], dim='state')
    ds_rearranged['I'] = ds_rearranged['I'].assign_coords(state=[0, 1])

    # Combine Q_g and Q_e into Q
    ds_rearranged['Q'] = xr.concat([ds.Q_g, ds.Q_e], dim='state')
    ds_rearranged['Q'] = ds_rearranged['Q'].assign_coords(state=[0, 1])

    # Copy other coordinates and data variables
    for var in ds.coords:
        if var not in ds_rearranged.coords:
            ds_rearranged[var] = ds[var]

    for var in ds.data_vars:
        if var not in ['I_g', 'I_e', 'Q_g', 'Q_e']:
            ds_rearranged[var] = ds[var]

    # Replace the original dataset with the rearranged one
    ds = ds_rearranged


    node.results = {"ds": ds}

    node.results["results"] = {}
    node.results["figs"] = {}


    plot_raw = False

    if plot_raw:
        fig, axes = plt.subplots(ncols=num_qubits, nrows=len(ds.amplitude), sharex=False, sharey=False,
                                squeeze=False, figsize=(5*num_qubits, 5*len(ds.amplitude)))
        for amplitude, ax1 in zip(ds.amplitude, axes):
            for q, ax2 in zip(list(qubits), ax1):
                ds_q = ds.sel(qubit=q.name, amplitude=amplitude)
                ax2.plot(ds_q.I.sel(state=0), ds_q.Q.sel(state=0), ".", alpha=0.2, label="Ground", markersize=2)
                ax2.plot(ds_q.I.sel(state=1), ds_q.Q.sel(state=1), ".", alpha=0.2, label="Excited", markersize=2)
                ax2.set_xlabel('I')
                ax2.set_ylabel('Q')
                ax2.set_title(f'{q.name}, {float(amplitude)}')
                ax2.axis('equal')
        plt.show()
        node.results['figure_raw_data'] = fig
        
# %%
if not node.parameters.simulate:

    def apply_fit_gmm(I, Q):
        I_mean = np.mean(I, axis=1)
        Q_mean = np.mean(Q, axis=1)
        means_init = [[I_mean[0], Q_mean[0]], [I_mean[1], Q_mean[1]]]
        precisions_init = [
            1 / ((np.mean(np.var(I, axis=1)) + np.mean(np.var(Q, axis=1)))/2)] * 2
        clf = GaussianMixture(n_components=2, covariance_type='spherical', means_init=means_init,
                              precisions_init=precisions_init, tol=1e-5, reg_covar=1e-12)
        X = np.array([np.array(I).flatten(), np.array(Q).flatten()]).T
        clf.fit(X)
        meas_fidelity = (np.sum(clf.predict(np.array([I[0], Q[0]]).T) == 0)/len(
            I[0]) + np.sum(clf.predict(np.array([I[1], Q[1]]).T) == 1)/len(I[1]))/2
        loglikelihood = clf.score_samples(X)
        max_ll = np.max(loglikelihood)
        outliers = np.sum(loglikelihood > np.log(0.01) + max_ll) / len(X)
        return np.array([meas_fidelity, outliers])


    fit_res = xr.apply_ufunc(apply_fit_gmm, ds.I, ds.Q,
                             input_core_dims=[['state', 'N'], ['state', 'N']],
                             output_core_dims=[['result']],
                             vectorize=True)

    fit_res = fit_res.assign_coords(result=['meas_fidelity', 'outliers'])

# %%
if not node.parameters.simulate:

    plot_individual = False
    best_data = {}

    best_amp = {}
    for q in qubits:
        fit_res_q = fit_res.sel(qubit=q.name)
        valid_amps = fit_res_q.amplitude[(fit_res_q.sel(result='outliers') >= node.parameters.outliers_threshold)]
        amps_fidelity = fit_res_q.sel(
            amplitude=valid_amps.values, result='meas_fidelity')
        best_amp[q.name] = float(amps_fidelity.readout_amp[amps_fidelity.argmax()])
        print(f'amp for {q.name} is {best_amp[q.name]}')
        node.results["results"][q.name] = {}
        node.results["results"][q.name]["best_amp"] = best_amp[q.name]
           
        # Select data for the best amplitude
        best_amp_data = ds.sel(qubit=q.name, amplitude=float(amps_fidelity.idxmax()))
        best_data[q.name] = best_amp_data
        
        # Extract I and Q data for ground and excited states
        I_g = best_amp_data.I.sel(state=0)
        Q_g = best_amp_data.Q.sel(state=0)
        I_e = best_amp_data.I.sel(state=1)
        Q_e = best_amp_data.Q.sel(state=1)
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I_g, Q_g, I_e, Q_e, True, b_plot=plot_individual)
        I_rot = I_g * np.cos(angle) - Q_g * np.sin(angle)
        hist = np.histogram(I_rot, bins=100)
        RUS_threshold = hist[1][1:][np.argmax(hist[0])]
        if plot_individual:
            fig = plt.gcf()
            plt.show()
            node.results["figs"][q.name] = fig
        node.results["results"][q.name]["angle"] = float(angle)
        node.results["results"][q.name]["threshold"] = float(threshold)
        node.results["results"][q.name]["fidelity"] = float(fidelity)
        node.results["results"][q.name]["confusion_matrix"] = np.array([[gg, ge], [eg, ee]])
        node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)




# %%
if not node.parameters.simulate:

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit_res.loc[qubit].plot(ax = ax,x = 'readout_amp', hue='result', add_legend=False)
        ax.axvline(best_amp[qubit['qubit']], color = 'k', linestyle = 'dashed')
        ax.set_xlabel('Relative power')
        ax.set_ylabel('Fidelity / outliers')
        ax.set_title(qubit['qubit'])
    grid.fig.suptitle('Assignment fidelity and non-outlier probability')

    plt.tight_layout()
    plt.show()
    node.results['figure_assignment_fid'] = grid.fig

# %%
if not node.parameters.simulate:

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds_q = best_data[qubit['qubit']]
        qn = qubit['qubit']
        ax.plot(1e3*(ds_q.I.sel(state=0) * np.cos(node.results["results"][qn]["angle"]) - ds_q.Q.sel(state=0) * np.sin(node.results["results"][qn]["angle"])), 1e3*(ds_q.I.sel(state=0) * np.sin(node.results["results"][qn]["angle"]) + ds_q.Q.sel(state=0) * np.cos(node.results["results"][qn]["angle"])), ".", alpha=0.1, label="Ground", markersize=1)
        ax.plot(1e3 * (ds_q.I.sel(state=1) * np.cos(node.results["results"][qn]["angle"]) - ds_q.Q.sel(state=1) * np.sin(node.results["results"][qn]["angle"])), 1e3 * (ds_q.I.sel(state=1) * np.sin(node.results["results"][qn]["angle"]) + ds_q.Q.sel(state=1) * np.cos(node.results["results"][qn]["angle"])), ".", alpha=0.1, label="Excited", markersize=1)
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


    grid = QubitGrid(ds, [q.grid_location for q in qubits])
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

    grid.fig.suptitle('g.s. and e.s. fidelity')
    plt.tight_layout()
    plt.show()
    node.results['figure_fidelities'] = grid.fig

# %%
if not node.parameters.simulate:

    with node.record_state_updates():
        for qubit in qubits:
            qubit.resonator.operations["readout"].integration_weights_angle -= float(node.results["results"][qubit.name]["angle"])
            qubit.resonator.operations["readout"].threshold = float(node.results["results"][qubit.name]["threshold"])
            qubit.resonator.operations["readout"].rus_exit_threshold = float(node.results["results"][qubit.name]["rus_threshold"])
            qubit.resonator.operations["readout"].amplitude = float(node.results["results"][qubit.name]["best_amp"])
            qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()

# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%