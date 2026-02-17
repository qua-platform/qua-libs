# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey import (
    RamseyDetuningParameters,
    fit_raw_data,
    log_fitted_results,
    plot_ramsey_detuning,
)
from calibration_utils.common_utils.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        RAMSEY DETUNING PARITY DIFFERENCE
This sequence performs a Ramsey measurement with parity difference to characterize the qubit detuning.
The measurement involves sweeping the detuning frequency of the qubit and performing a sequence of
two π/2 rotations with a fixed idle time in between. PSB is used to measure the parity of the resulting state.

The sequence uses voltage sequences to navigate through a triangle in voltage space (empty -
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each pulse duration,
the parity is measured before (P1) and after (P2) the qubit pulse, and the parity difference (P_diff) is
calculated. When P1 == P2, P_diff = 0; otherwise P_diff = 1.

The parity difference signal reveals oscillations as a function of pulse detuning frequency, which can be used
to extract the qubit frequency.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X90 pulse amplitude and frequency).

State update:
    - The qubit Larmor frequency.
"""


node = QualibrationNode[RamseyDetuningParameters, Quam](
    name="10b_ramsey_detuning_parity_diff", description=description, parameters=RamseyDetuningParameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit = ["q1"]
    # node.parameters.num_shots = 10
    # node.parameters.detuning_span_in_mhz = 5.0
    # node.parameters.detuning_step_in_mhz = 0.1
    # node.parameters.idle_time_ns = 100
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    idle_time = node.parameters.idle_time_ns // 4  # Convert to clock cycles
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_step_in_mhz * u.MHz,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(detuning_values, attrs={"long_name": "frequency detuning", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        df = declare(int)
        n = declare(int)

        p1 = declare(int, size=num_qubits)
        p2 = declare(int, size=num_qubits)

        p1_st = {qubit.name: declare_stream() for qubit in qubits}
        p2_st = {qubit.name: declare_stream() for qubit in qubits}
        pdiff_st = {qubit.name: declare_stream() for qubit in qubits}
        n_st = declare_stream()

        for batched_qubits in qubits.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, detuning_values)):
                    for i, qubit in batched_qubits.items():
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency + df)

                    # ---------------------------------------------------------
                    # Step 1: Empty - step to empty point (fixed duration)
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.empty()

                    align()
                    for i, qubit in batched_qubits.items():
                        assign(p1[i], Cast.to_int(qubit.measure()))

                    # ---------------------------------------------------------
                    # Step 2: Initialize - load electron into dot (fixed duration)
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        op_length = qubit.macros["x90"].duration
                        qubit.initialize(duration=node.parameters.gap_wait_time_in_ns + op_length * 2 + 4 * idle_time)
                    # ---------------------------------------------------------
                    # Step 3: X90 pulse, idle, X90 pulse
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        qubit.x90()
                        qubit.xy.wait(idle_time)
                        qubit.x90()
                    # ---------------------------------------------------------
                    # Step 4: Measure - move to PSB and measure
                    # ---------------------------------------------------------
                    align()

                    for i, qubit in batched_qubits.items():
                        assign(p2[i], Cast.to_int(qubit.measure()))

                    # ---------------------------------------------------------
                    # Step 5: Apply compensation pulse to reset DC bias
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.voltage_sequence.apply_compensation_pulse()

                    # ---------------------------------------------------------
                    # Save results
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        save(p1[i], p1_st[qubit.name])
                        save(p2[i], p2_st[qubit.name])

                        with if_(p1[i] == p2[i]):
                            save(0, pdiff_st[qubit.name])
                        with else_():
                            save(1, pdiff_st[qubit.name])

        with stream_processing():
            n_st.save("n")

            n_detuning = len(detuning_values)
            for qubit in qubits:
                p1_st[qubit.name].buffer(n_detuning).average().save(f"p1_{qubit.name}")
                p2_st[qubit.name].buffer(n_detuning).average().save(f"p2_{qubit.name}")
                pdiff_st[qubit.name].buffer(n_detuning).average().save(f"pdiff_{qubit.name}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[RamseyDetuningParameters, Quam]):
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
def execute_qua_program(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["sensors"] = get_sensors(node)
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Plot the raw and fitted data."""
    fig = plot_ramsey_detuning(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results.get("fit_results"),
    )
    plt.show()
    node.results["figures"] = {"ramsey_detuning": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Update the relevant parameters if the qubit pair data analysis was successful."""

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.xy.RF_frequency -= fit_result["freq_offset"]
            qubit.larmor_frequency -= fit_result["freq_offset"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    node.save()
