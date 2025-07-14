"""
        QUBIT SPECTROSCOPY - 0->2 TRANSITION
This sequence involves sending a saturation pulse to the qubit to find the 0->2 / 2 transition frequency,
which allows estimation of the qubit anharmonicity. The sequence searches around f01 - α/2,
where f01 is the qubit frequency and α is the anharmonicity (default guess of 200 MHz).

The data is post-processed to determine the 0->2 transition frequency and calculate the actual anharmonicity.

Prerequisites:
    - Having run the qubit spectroscopy (03a) to find the 0->1 transition frequency
    - Set the flux bias to the desired working point

Before proceeding to the next node:
    - Update the qubit anharmonicity in the state
    - Save the current state
"""


# %% {Imports}
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 1000
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 7  # Higher power to drive 0->2 transition
    operation_len_in_ns: Optional[int] = None
    initial_anharmonicity_mhz: float = 200.0  # Default anharmonicity guess
    frequency_span_in_mhz: float = 50  # Search window around f01 - α/2
    frequency_step_in_mhz: float = 0.25
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    target_peak_width: Optional[float] = 2e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="03e_Qubit_Spectroscopy_02", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
operation = node.parameters.operation  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0

# Calculate expected 0->2 transition frequency (f01 - α/2)
init_anharmonicity = node.parameters.initial_anharmonicity_mhz * u.MHz
# Qubit detuning sweep around expected 0->2 transition
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)

flux_point = node.parameters.flux_point_joint_or_independent
qubit_freqs = {q.name: q.xy.RF_frequency for q in qubits}  # for opx

# Set the qubit frequency for a given flux point
if node.parameters.arbitrary_flux_bias is not None:
    arb_flux_bias_offset = {q.name: node.parameters.arbitrary_flux_bias for q in qubits}
    detunings = {q.name: q.freq_vs_flux_01_quad_term * arb_flux_bias_offset[q.name] ** 2 for q in qubits}
elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
    detunings = {
        q.name: 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name] for q in qubits
    }
    arb_flux_bias_offset = {q.name: np.sqrt(detunings[q.name] / q.freq_vs_flux_01_quad_term) for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}

# Adjust detunings to search around 0->2 transition
for q in qubits:
    detunings[q.name] -= init_anharmonicity / 2

target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = 3e6  # the desired width of the response to the saturation pulse

with program() as qubit_spec_02:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name])
                qubit.align()
                duration = operation_len * u.ns if operation_len is not None else (qubit.xy.operations[operation].length + qubit.z.settle_time) * u.ns
                # Bring the qubit to the desired point during the saturation pulse
                qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude, duration=duration)
                # Play the saturation pulse
                qubit.xy.wait(qubit.z.settle_time * u.ns)
                qubit.xy.play(
                    operation,
                    amplitude_scale=operation_amp,
                    duration=duration,
                )
                qubit.align()

                # readout the resonator
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                # Wait for the qubit to decay to the ground state
                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                # save data
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec_02, simulation_config)
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
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qubit_spec_02)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + qubit_freqs[q.name] + detunings[q.name] for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # search for frequency for which the amplitude the farthest from the mean to indicate the approximate location of the peak
    shifts = np.abs((ds.IQ_abs - ds.IQ_abs.mean(dim="freq"))).idxmax(dim="freq")
    # Find the rotation angle to align the separation along the 'I' axis
    angle = np.arctan2(
        ds.sel(freq=shifts).Q - ds.Q.mean(dim="freq"),
        ds.sel(freq=shifts).I - ds.I.mean(dim="freq"),
    )
    # rotate the data to the new I axis
    ds = ds.assign({"I_rot" : ds.I * np.cos(angle) + ds.Q * np.sin(angle)})
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    result = peaks_dips(ds.I_rot, dim="freq", prominence_factor=5)
    # The resonant RF frequency of the qubits
    abs_freqs = dict(
        [
            (
                q.name,
                ds.freq_full.sel(freq = result.position.sel(qubit=q.name).values).sel(qubit=q.name).values,
            )
            for q in qubits if not np.isnan(result.sel(qubit=q.name).position.values)
        ]
    )

    # Save fitting results and calculate anharmonicities
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(result.sel(qubit=q.name).position.values):
            fit_results[q.name]["fit_successful"] = True
            f02_2 = (result.sel(qubit=q.name).position.values + init_anharmonicity / 2)  # Hz
            measured_anharmonicity = 2*f02_2  # Hz
            f01 = q.xy.RF_frequency
            
            print(f"\nResults for {q.name}:")
            print(f"f01 frequency: {f01/1e9:.6f} GHz")
            print(f"Measured anharmonicity: {measured_anharmonicity/1e6:.2f} MHz")
            print(f"(compared to initial guess of {node.parameters.initial_anharmonicity_mhz:.2f} MHz)")
            
            fit_results[q.name]["f01"] = f01
            fit_results[q.name]["f02_2"] = f02_2
            fit_results[q.name]["anharmonicity"] = measured_anharmonicity
            fit_results[q.name]["peak_width"] = result.sel(qubit=q.name).width.values
            print(f"Found a peak width of {result.sel(qubit=q.name).width.values/1e6:.2f} MHz")
            print(f"readout angle for qubit {q.name}: {angle.sel(qubit=q.name).values:.4}")
        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"\nFailed to find a peak for {q.name}")
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    approx_peak = result.base_line + result.amplitude * (1 / (1 + ((ds.freq - result.position) / result.width) ** 2))
    for ax, qubit in grid_iter(grid):
        # Plot the line
        (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].I_rot * 1e3).plot(ax=ax, x="freq_GHz")
        # Identify the resonance peak
        if not np.isnan(result.sel(qubit=qubit["qubit"]).position.values):
            ax.plot(
                abs_freqs[qubit["qubit"]] / 1e9,
                ds.loc[qubit].sel(freq=result.loc[qubit].position.values, method="nearest").I_rot * 1e3,
                ".r",
            )
            # Identify the width
            (approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit] * 1e3).plot(
                ax=ax, x="freq_GHz", linewidth=0.5, linestyle="--"
            )
        ax.set_xlabel("Qubit freq [GHz]")
        ax.set_ylabel("Trans. amp. [mV]")
        ax.set_title(f"{qubit['qubit']}")
    grid.fig.suptitle(f"Qubit spectroscopy 0->2 transition \n {date_time} GMT+3 #{node_id} \n multiplexed = {node.parameters.multiplexed}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in qubits:
                if not np.isnan(result.sel(qubit=q.name).position.values):
                    # Update the anharmonicity in the qubit state
                    q.anharmonicity = int(fit_results[q.name]["anharmonicity"])

        node.results["ds"] = ds

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        save_node(node)

# %% 