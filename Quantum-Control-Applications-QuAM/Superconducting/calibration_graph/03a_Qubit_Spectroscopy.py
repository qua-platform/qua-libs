# %%
"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly without having to modify the configuration.

The data is post-processed to determine the qubit resonance frequency, which can then be used to adjust
the qubit intermediate frequency in the configuration under "center".

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

This step can be repeated using the "x180" operation instead of "saturation" to adjust the pulse parameters (amplitude,
duration, frequency) before performing the next calibration steps.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.
    - Specification of the expected qubit T1 in the state.

Before proceeding to the next node:
    - Update the qubit frequency, labeled as f_01, in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.01
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 40
    frequency_step_in_mhz: float = 0.25
    target_peak_width: Optional[int] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = 5.845
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="03a_Qubit_Spectroscopy", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
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
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
operation = node.parameters.operation  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
operation_len = (
    node.parameters.operation_len_in_ns
)  # can be None - will just be ignored
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
qubit_freqs = {q.name: q.xy.RF_frequency for q in qubits}  # for opx
# qubit_freqs = {q.name :  q.xy.intermediate_frequency + q.xy.opx_output.upconverter_frequency for q in qubits} # for MW


target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = 3e6  # the desired width of the response to the saturation pulse (including saturation amp), in Hz

with program() as qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency

    for i, qubit in enumerate(qubits):

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                qubit.xy.update_frequency(
                    df + qubit.xy.intermediate_frequency
                )
                wait(250)
                qubit.align()

                # Play the saturation pulse
                qubit.xy.play(
                    operation,
                    amplitude_scale=operation_amp,
                    duration=operation_len,
                )
                # TODO: why?
                qubit.xy.wait(250)
                qubit.align()

                # # QUA macro the readout the state of the active resonators (defined in macros.py)
                # multiplexed_readout(qubits, I, I_st, Q, Q_st, sequential=False)
                # readout the resonator
                qubit.resonator.wait(250)
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                # Wait for the qubit to decay to the ground state
                qubit.resonator.wait(machine.depletion_time * u.ns)
                # save data
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])

        align(*([q.xy.name for q in qubits] + [q.resonator.name for q in qubits]))

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    # qm = qmm.open_qm(config, close_other_machines=True)
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qubit_spec)
        # Get results from QUA program
        data_list = ["n"] + sum(
            [[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], []
        )
        results = fetching_tool(job, data_list, mode="live")

        # Live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        while results.is_processing():
            # Fetch results
            fetched_data = results.fetch_all()
            n = fetched_data[0]
            I = fetched_data[1::2]
            Q = fetched_data[2::2]

            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

            plt.suptitle("Qubit spectroscopy")
            s_data = []
            for i, qubit in enumerate(qubits):
                s = u.demod2volts(
                    I[i] + 1j * Q[i], qubit.resonator.operations["readout"].length
                )
                s_data.append(s)
                plt.subplot(2, num_qubits, i + 1)
                plt.cla()
                plt.plot((qubit.xy.RF_frequency + dfs) / u.MHz, np.abs(s))
                plt.plot(qubit.xy.RF_frequency / u.MHz, max(np.abs(s)), "r*")
                plt.grid(True)
                plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
                plt.title(f"{qubit.name} (f_01: {qubit.xy.RF_frequency / u.MHz} MHz)")
                plt.subplot(2, num_qubits, num_qubits + i + 1)
                plt.cla()
                plt.plot((qubit.xy.RF_frequency + dfs) / u.MHz, np.unwrap(np.angle(s)))
                plt.plot(
                    qubit.xy.RF_frequency / u.MHz, max(np.unwrap(np.angle(s))), "r*"
                )
                plt.grid(True)
                plt.ylabel("Phase [rad]")
                plt.xlabel(f"{qubit.name} detuning [MHz]")

            plt.tight_layout()
            plt.pause(0.1)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) and phase
    ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
    # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
    ds = ds.assign_coords(
        {
            "freq_full": (
                ["qubit", "freq"],
                np.array(
                    [dfs + qubit_freqs[q.name] for q in qubits]
                ),
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
    ds = ds.assign({"I_rot": np.real(ds.IQ_abs * np.exp(1j * (ds.phase - angle)))})
    # Find the peak with minimal prominence as defined, if no such peak found, returns nan
    result = peaks_dips(ds.I_rot, dim="freq", prominence_factor=5)
    # The resonant RF frequency of the qubits
    abs_freqs = dict(
        [
            (
                q.name,
                result.sel(qubit=q.name).position.values
                + qubit_freqs[q.name],
            )
            for q in qubits
        ]
    )

    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(result.sel(qubit=q.name).position.values):
            fit_results[q.name]["fit_successful"] = True
            Pi_length = q.xy.operations["x180"].length
            used_amp = q.xy.operations["saturation"].amplitude * operation_amp
            print(
                f"Drive frequency for {q.name} is "
                f"{(result.sel(qubit = q.name).position.values + q.xy.RF_frequency) / 1e9:.6f} GHz"
            )
            fit_results[q.name]["drive_freq"] = (
                result.sel(qubit=q.name).position.values + q.xy.RF_frequency
            )
            print(
                f"(shift of {result.sel(qubit = q.name).position.values/1e6:.3f} MHz)"
            )
            factor_cw = float(target_peak_width / result.sel(qubit=q.name).width.values)
            factor_pi = np.pi / (
                result.sel(qubit=q.name).width.values * Pi_length * 1e-9
            )
            print(
                f"Found a peak width of {result.sel(qubit = q.name).width.values/1e6:.2f} MHz"
            )
            print(
                f"To obtain a peak width of {target_peak_width/1e6:.1f} MHz the cw amplitude is modified "
                f"by {factor_cw:.2f} to {factor_cw * used_amp / operation_amp * 1e3:.0f} mV"
            )
            print(
                f"To obtain a Pi pulse at {Pi_length} nS the Rabi amplitude is modified by {factor_pi:.2f} "
                f"to {factor_pi*used_amp*1e3:.0f} mV"
            )
            print(
                f"readout angle for qubit {q.name}: {angle.sel(qubit = q.name).values:.4}"
            )
            print()
        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"Failed to find a peak for {q.name}")
            print()
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    grid_names = [f"{q.name}_0" for q in qubits]
    grid = QubitGrid(ds, grid_names)
    approx_peak = result.base_line + result.amplitude * (
        1 / (1 + ((ds.freq - result.position) / result.width) ** 2)
    )
    for ax, qubit in grid_iter(grid):
        # Plot the line
        (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].I_rot * 1e3).plot(
            ax=ax, x="freq_GHz"
        )
        # Identify the resonance peak
        ax.plot(
            abs_freqs[qubit["qubit"]] / 1e9,
            ds.loc[qubit]
            .sel(freq=result.loc[qubit].position.values, method="nearest")
            .I_rot
            * 1e3,
            ".r",
        )
        # Identify the width
        (approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit] * 1e3).plot(
            ax=ax, x="freq_GHz", linewidth=0.5, linestyle="--"
        )
        ax.set_xlabel("Qubit freq [GHz]")
        ax.set_ylabel("Trans. amp. [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Qubit spectroscopy (amplitude)")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    with node.record_state_updates():
        for q in qubits:
            if not np.isnan(result.sel(qubit=q.name).position.values):
                # intermediate frequency
                q.xy.intermediate_frequency += float(result.sel(qubit=q.name).position.values)
                
                # angle for integration weights
                prev_angle = q.resonator.operations["readout"].integration_weights_angle
                if not prev_angle:
                    prev_angle = 0.0
                q.resonator.operations["readout"].integration_weights_angle = (prev_angle + angle.sel(qubit=q.name).values) % (2 * np.pi)

                # pi len
                Pi_length = q.xy.operations["x180"].length

                # amplitude                
                used_amp = q.xy.operations["saturation"].amplitude * operation_amp
                factor_cw = float(target_peak_width / result.sel(qubit=q.name).width.values)
                factor_pi = np.pi / (result.sel(qubit=q.name).width.values * Pi_length * 1e-9)
                
                if factor_cw * used_amp / operation_amp < 1.0:  # TODO: 1 for OPX1000 MW
                    q.xy.operations["saturation"].amplitude = factor_cw * used_amp / operation_amp
                else:
                    q.xy.operations["saturation"].amplitude = 1.0  # TODO: 1 for OPX1000 MW

                if factor_pi * used_amp < 1.0:  # TODO: 1 for OPX1000 MW
                    q.xy.operations["x180"].amplitude = factor_pi * used_amp
                elif factor_pi * used_amp >= 1.0:  # TODO: 1 for OPX1000 MW
                    q.xy.operations["x180"].amplitude = 1.0

    node.results["ds"] = ds

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
