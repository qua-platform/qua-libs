"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_averages: int = 10
    min_flux_offset_in_v: float = -0.5
    max_flux_offset_in_v: float = 0.5
    num_flux_points: int = 201
    frequency_span_in_mhz: float = 20
    frequency_step_in_mhz: float = 0.1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    update_flux_min: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None

node = QualibrationNode(name="02b_Resonator_Spectroscopy_vs_Flux", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Flux bias sweep in V
dcs = np.linspace(
    node.parameters.min_flux_offset_in_v,
    node.parameters.max_flux_offset_in_v,
    node.parameters.num_flux_points,
)
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
update_flux_min = node.parameters.update_flux_min  # Update the min flux point

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I_control, I_control_st, Q_control, Q_control_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    I_target, I_target_st, Q_target, Q_target_st, _ , _ = qua_declaration(num_qubits=num_qubit_pairs)
    dc = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    for i, qp in enumerate(qubit_pairs):
        # resonator of the qubit
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qp)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(dc, dcs)):
                # Flux sweeping by tuning the OPX dc offset associated with the flux_line element
                qp.coupler.set_dc_offset(dc)
                qp.align()
                rr_control = qp.qubit_control.resonator
                rr_target = qp.qubit_target.resonator
                with for_(*from_array(df, dfs)):
                    # Update the resonator frequencies for resonator
                    update_frequency(rr_control.name, df + rr_control.intermediate_frequency)
                    update_frequency(rr_target.name, df + rr_target.intermediate_frequency)
                    # readout the resonator
                    rr_control.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                    rr_target.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                    # wait for the resonator to relax
                    rr_control.wait(machine.depletion_time * u.ns)
                    rr_target.wait(machine.depletion_time * u.ns)
                    # save data
                    save(I_control[i], I_control_st[i])
                    save(Q_control[i], Q_control_st[i])
                    save(I_target[i], I_target_st[i])
                    save(Q_target[i], Q_target_st[i])
        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            I_control_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I_control{i + 1}")
            Q_control_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q_control{i + 1}")
            I_target_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"I_target{i + 1}")
            Q_target_st[i].buffer(len(dfs)).buffer(len(dcs)).average().save(f"Q_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"freq": dfs, "flux": dcs})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubit_pairs)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs_control": np.sqrt(ds["I_control"] ** 2 + ds["Q_control"] ** 2)})
        ds = ds.assign({"IQ_abs_target": np.sqrt(ds["I_target"] ** 2 + ds["Q_target"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq_control = np.array([dfs + qp.qubit_control.resonator.RF_frequency for qp in qubit_pairs])
        RF_freq_target = np.array([dfs + qp.qubit_target.resonator.RF_frequency for qp in qubit_pairs])
        ds = ds.assign_coords({"freq_full_control": (["qubit", "freq"], RF_freq_control)})
        ds.freq_full_control.attrs["long_name"] = "Frequency"
        ds.freq_full_control.attrs["units"] = "GHz"
        ds = ds.assign_coords({"freq_full_target": (["qubit", "freq"], RF_freq_target)})
        ds.freq_full_target.attrs["long_name"] = "Frequency"
        ds.freq_full_target.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}

    # %% {Plotting}
    from quam_libs.lib.plot_utils import QubitPairGrid, grid_pair_names
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        # Plot using the attenuated current x-axis
        ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp['qubit']).IQ_abs_control.plot(
            ax=ax,
            add_colorbar=False,
            x="flux",
            y="freq_GHz",
            robust=True,
        )
        ax.set_title(qp["qubit"])
        ax.set_xlabel("Coupler Flux (V)")

    grid.fig.suptitle("Control Resonator spectroscopy vs Coupler Flux")
    plt.tight_layout()
    plt.show()
    node.results["figure_control"] = grid.fig


    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        # Plot using the attenuated current x-axis
        ds.assign_coords(freq_GHz=ds.freq_full_target / 1e9).sel(qubit=qp['qubit']).IQ_abs_target.plot(
            ax=ax,
            add_colorbar=False,
            x="flux",
            y="freq_GHz",
            robust=True,
        )
        ax.set_title(qp["qubit"])
        ax.set_xlabel("Coupler Flux (V)")

    grid.fig.suptitle("Target Resonator spectroscopy vs Coupler Flux")
    plt.tight_layout()
    plt.show()
    node.results["figure_target"] = grid.fig

    # %% {Update_state}

    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()



# %%
