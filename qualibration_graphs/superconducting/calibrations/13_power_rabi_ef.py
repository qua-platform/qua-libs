# %% {Imports}
import dataclasses
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.power_rabi import (
    EfParameters,
    fit_raw_data,
    get_number_of_pulses,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs import data
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
        EF POWER RABI CALIBRATION
This node calibrates the pi pulse operation between the |e> and |f> states of a superconducting
qubit by populating the |e> state with a previously calibrated pi pulse and applying a varying
amplitude detuned pulse at the |e> -> |f> transition frequency.
Prerequisites:
        - Having calibrated a pi pulse operation between the |g> and |e> states of the qubit (x180).
            (04_power_rabi.py)
        - Having calibrated the readout resonator dispersive shift (chi).
            (08a_readout_frequency_optimization.py)
        - Having calibrated the qubit anharmonicity.

State update:
        - The qubit pulse amplitude corresponding to the EF_x180 operation
            (qubit.xy.operations["EF_x180"].amplitude).
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[EfParameters, Quam](
    name="13_power_rabi_ef",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=EfParameters(),  # EF-specific parameters set
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[EfParameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[EfParameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    if node.parameters.reset_type == "active":
        raise ValueError("'active' is not supported, use 'thermal' or 'active_gef' instead")
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)  # Get qubits to calibrate
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    # Number of applied Rabi pulses sweep
    N_pi_vec = get_number_of_pulses(node.parameters)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }
    for qubit in qubits:
        # Check if the qubit has the required operations
        if hasattr(qubit.xy.operations, "EF_x180"):
            continue
        else:
            x180 = qubit.xy.operations["x180"]
            qubit.xy.operations["EF_x180"] = (
                dataclasses.replace(x180, alpha=0.0) if hasattr(x180, "alpha") else dataclasses.replace(x180)
            )

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        npi = declare(int)  # QUA variable for the number of qubit pulses
        count = declare(int)  # QUA variable for counting the qubit pulses

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
                qubit.resonator.update_frequency(
                    qubit.resonator.intermediate_frequency
                    + (qubit.resonator.GEF_frequency_shift if node.parameters.use_state_discrimination else qubit.chi)
                )
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():  # Reset the qubit to ground state
                        qubit.reset(reset_type=node.parameters.reset_type, simulate=node.parameters.simulate)
                        # Wait twice the regular thermal time for proper |f> state reset
                        if node.parameters.reset_type == "thermal":
                            qubit.wait(qubit.thermalization_time * u.ns)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        # Set the XY channel to the |g> -> |e> transition (GE) intermediate frequency
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        # Apply previously calibrated pi pulse to populate |e>
                        qubit.xy.play("x180")
                        # Shift drive to the |e> -> |f> (EF) transition by subtracting the anharmonicity
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency - qubit.anharmonicity)
                        # Apply EF pi pulse with swept amplitude scaling factor 'a'
                        qubit.xy.play("EF_x180", amplitude_scale=a)
                    align()

                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state_gef(state[i])  # Need to calibrate gef readout first
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amps)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[EfParameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[EfParameters, Quam]):
    """Connect, execute QUA program, fetch raw data and store as dataset 'ds_raw'."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[EfParameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[EfParameters, Quam]):
    """Analyse raw data -> store fitted dataset 'ds_fit' and dict 'fit_results'."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[EfParameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[EfParameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            else:
                operation = q.xy.operations["EF_x180"]
                operation.amplitude = node.results["fit_results"][q.name]["opt_amp"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[EfParameters, Quam]):
    node.save()


# %%
