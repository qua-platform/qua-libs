# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey_detuning_parity_diff import (
    Parameters as RamseyDetuningParameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.common_utils.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
RAMSEY DETUNING PARITY DIFFERENCE (TWO-τ)

Sweeps the drive-frequency detuning at two fixed idle times (τ_short
and τ_long) and measures the parity difference after a π/2 – idle – π/2
sequence at each.

The two traces act as a Vernier: wide fringes (short τ) localise the
resonance coarsely, narrow fringes (long τ) sharpen the estimate.  The
amplitude ratio between traces gives the exponential decay rate (T₂*).
A joint differential-evolution fit with shared (bg, A₀, δ₀, γ) across
both traces extracts the resonance detuning and T₂*.

Prerequisites:
    - Calibrated resonators and voltage points (empty - init - measure).
    - Calibrated X90 pulse amplitude and frequency.

State update:
    - qubit.xy.intermediate_frequency
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
    """Create the sweep axes and generate the QUA program.

    Sweeps drive-frequency detuning at two fixed idle times (τ_short
    and τ_long), performing a π/2 – idle – π/2 Ramsey sequence with
    parity-difference readout at each.  Produces a 2-D dataset
    (tau × detuning).
    """
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    # Two idle times in clock cycles (4 ns each)
    idle_times_cc = np.array([
        node.parameters.idle_time_ns // 4,
        node.parameters.idle_time_long_ns // 4,
    ])
    idle_times_ns = np.array([
        node.parameters.idle_time_ns,
        node.parameters.idle_time_long_ns,
    ], dtype=float)
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_step_in_mhz * u.MHz,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(
            idle_times_ns, attrs={"long_name": "idle time", "units": "ns"}
        ),
        "detuning": xr.DataArray(
            detuning_values, attrs={"long_name": "frequency detuning", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        t = declare(int)
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

                with for_(*from_array(t, idle_times_cc)):
                    with for_(*from_array(df, detuning_values)):
                        for i, qubit in batched_qubits.items():
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency + df)

                        # Step 1: Empty
                        align()
                        for i, qubit in batched_qubits.items():
                            qubit.empty()

                        align()
                        for i, qubit in batched_qubits.items():
                            assign(p1[i], Cast.to_int(qubit.measure()))

                        # Step 2: Initialize
                        align()
                        for i, qubit in batched_qubits.items():
                            op_length = qubit.macros["x90"].duration
                            qubit.initialize(
                                duration=node.parameters.gap_wait_time_in_ns + op_length * 2 + 4 * t
                            )

                        # Step 3: X90 – idle – X90
                        for i, qubit in batched_qubits.items():
                            qubit.x90()
                            qubit.xy.wait(t)
                            qubit.x90()

                        # Step 4: Measure
                        align()
                        for i, qubit in batched_qubits.items():
                            assign(p2[i], Cast.to_int(qubit.measure()))

                        # Step 5: Compensation
                        align()
                        for i, qubit in batched_qubits.items():
                            qubit.voltage_sequence.apply_compensation_pulse()

                        # Save results
                        for i, qubit in batched_qubits.items():
                            save(p1[i], p1_st[qubit.name])
                            save(p2[i], p2_st[qubit.name])

                            with if_(p1[i] == p2[i]):
                                save(0, pdiff_st[qubit.name])
                            with else_():
                                save(1, pdiff_st[qubit.name])

        with stream_processing():
            n_st.save("n")

            n_tau = len(idle_times_cc)
            n_detuning = len(detuning_values)
            for qubit in qubits:
                p1_st[qubit.name].buffer(n_tau, n_detuning).average().save(f"p1_{qubit.name}")
                p2_st[qubit.name].buffer(n_tau, n_detuning).average().save(f"p2_{qubit.name}")
                pdiff_st[qubit.name].buffer(n_tau, n_detuning).average().save(f"pdiff_{qubit.name}")


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
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Fit resonance detuning via joint two-τ differential evolution."""
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "_diag"}
        for k, v in fit_results.items()
    }
    # Keep diagnostics in namespace for plotting
    node.namespace["_fit_results_full"] = fit_results

    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Plot parity-difference vs detuning for both τ traces with joint fit."""
    fit_with_diag = node.namespace.get("_fit_results_full", node.results.get("fit_results", {}))
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubits"],
        fit_with_diag,
    )
    plt.show()
    node.results["figure"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    """Correct the qubit intermediate frequency by the fitted resonance offset."""

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.xy.intermediate_frequency -= fit_result["freq_offset"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[RamseyDetuningParameters, Quam]):
    node.save()
