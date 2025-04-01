# %% {Imports}
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np

from qm import SimulationConfig
from qm.qua import *

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate import QualibrationNode, NodeParameters
from quam_config import QuAM
from qualibration_libs.plot_utils import QubitGrid, grid_iter
from qualibration_libs.save_utils import fetch_results_as_xarray
from quam_experiments.analysis.fit import peaks_dips
from quam_experiments.parameters.qubits_experiment import get_qubits


# %% {Description}
description = """
        EF QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed
state, and then measuring the state of the resonator across various qubit drive
intermediate dfs. In order to facilitate the qubit search, the qubit pulse duration and
amplitude can be changed manually in the QUA program directly without having to
modify the configuration.

The data is post-processed to determine the qubit resonance frequency, which can then be
used to adjust the qubit intermediate frequency in the configuration under "center".

Note that it can happen that the qubit is excited by the image sideband or LO leakage
instead of the desired sideband. This is why calibrating the qubit mixer is highly
recommended.

This step can be repeated using the "x180" operation instead of "saturation" to adjust
the pulse parameters (amplitude,duration, frequency) before performing the next
calibration steps.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in
      question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an
      external mixer or an Octave port).
    - Set the flux bias to the minimum frequency point, labeled as
      "max_frequency_point", in the state.
    - Specification of the expected qubit T1 in the state.

Before proceeding to the next node:
    - Update the qubit frequency, labeled as f_01, in the state.
    - Save the current state by calling node.machine.save("quam")
"""


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.5
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 200
    frequency_step_in_mhz: float = 1.0
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    timeout: int = 100
    multiplexed: bool = False
    load_data_id: Optional[int] = None


node = QualibrationNode[Parameters, QuAM](
    name="12a_qubit_spectroscopy_e_to_f.py",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QuAM class from the state file
node.machine = QuAM.load()

# %% {Create_QUA_program}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Generate the OPX and Octave configurations
config = node.machine.generate_config()

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = node.machine.connect()

# Get the active qubits from the node and organize them by batches
node.namespace["qubits"] = qubits = get_qubits(node)
num_qubits = len(qubits)

operation = node.parameters.operation  # The qubit operation to play
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit
    I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
    df = declare(int)  # QUA variable for the qubit frequency

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        node.machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                qubit.align()
                qubit.xy.wait(5 * qubit.thermalization_time * u.ns)
                # Reset the qubit frequency
                update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                # Drive the qubit to the excited state
                qubit.xy.play("x180")
                # Update the qubit frequency to scan around the excepted f_01
                update_frequency(
                    qubit.xy.name,
                    df - qubit.anharmonicity + qubit.xy.intermediate_frequency,
                )
                # Play the saturation pulse
                qubit.xy.play(
                    operation,
                    amplitude_scale=operation_amp,
                    duration=operation_len,
                )
                qubit.align()
                # readout the resonator
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                # Wait for the qubit to decay to the ground state
                qubit.resonator.wait(qubit.thermalization_time * u.ns)
                # save data
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()

# %% {Execute}
if not node.parameters.simulate and node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qubit_spec)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Load_data}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
        # Add the qubit pulse absolute amplitude and phase to the dataset
        ds = ds.assign({"IQ_abs": (np.sqrt((ds.I - ds.I.mean(dim="freq")) ** 2 + (ds.Q - ds.Q.mean(dim="freq")) ** 2))})
        ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + q.xy.RF_frequency - q.anharmonicity for q in qubits]),
                ),
                "detuning": (
                    ["qubit", "freq"],
                    np.array([dfs - q.anharmonicity for q in qubits]),
                ),
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}

# %% {Analyse_data}
# shifts = (np.sqrt((ds.I - ds.I.mean(dim = "freq"))**2 + (ds.Q - ds.Q.mean(dim = "freq"))**2) ).idxmax(dim="freq")

if not node.parameters.simulate:

    # find the peak with minimal prominence as defined, if no such peak found, returns nan
    result = peaks_dips(ds.I, dim="freq", prominence_factor=3, remove_baseline=True)

    # calculate the modified anharmonicity
    anharmonicities = dict(
        [
            (
                q.name,
                ds.detuning.sel(qubit=q.name, freq=result.sel(qubit=q.name).position.values),
            )
            for q in qubits
            if not np.isnan(result.sel(qubit=q.name).position.values)
        ]
    )

    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(result.sel(qubit=q.name).position.values):
            fit_results[q.name]["success"] = True
            print(f"Anharmonicity for {q.name} is {anharmonicities[q.name]/1e6:.3f} MHz")
            fit_results[q.name]["anharmonicity"] = anharmonicities[q.name].values
        else:
            fit_results[q.name]["success"] = False
            print(f"Failed to find a peak for {q.name}")
            print()

    node.results["fit_results"] = fit_results
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }

# %% {Plot_data}
if not node.parameters.simulate:

    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        # Plot the IQ_abs as a function of the full frequency
        (ds.assign_coords(freq_MHz=ds.detuning / 1e6).loc[qubit].I * 1e3).plot(ax=ax, x="freq_MHz")
        # Add a lin where the e->f is supposed to be
        ax.axvline(
            anharmonicities[qubit["qubit"]] / 1e6,
            color="r",
            linestyle="--",
        )
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("Trans. amp. [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Qubit spectroscopy (E-F)")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    # Revert the change done at the beginning of the node
    for qubit in tracked_qubits:
        qubit.revert_changes()

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            q.anharmonicity = int(fit_result["anharmonicity"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()
