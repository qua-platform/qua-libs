# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_qubit_pairs, enable_dual_drive_mw_pairs
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.crot_spectroscopy_parity_diff import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)


# %% {Node initialisation}
description = """
        CROT (CONTROLLED-ROTATION) SPECTROSCOPY - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to measure the exchange coupling J between two spin qubits and identify
the conditional resonance frequencies required for implementing a CROT (controlled-rotation) gate.

In exchange-coupled spin qubits, the resonance frequency of one qubit depends on the spin state
of its partner. When the spectator qubit is in |↓⟩ vs |↑⟩, the driven qubit's resonance frequency
shifts by the exchange coupling strength J. This state-dependent frequency shift enables conditional
quantum operations - the foundation of two-qubit gates in the Loss-DiVincenzo architecture.

This measurement performs a 2D sweep of drive frequency vs virtual barrier gate (or virtual exchange voltage)
to map out the exchange coupling as a function of the inter-dot tunnel coupling. It runs two symmetric
experiments per pair (selected by the ``measured_qubit`` axis):
    - Drive + read out the TARGET qubit, conditioning on the CONTROL (spectator) state.
    - Drive + read out the CONTROL qubit, conditioning on the TARGET (spectator) state.
In both cases the spectator is the qubit flipped by the conditional x180 (the ``spectator_x180`` axis).

The QUA program sequence (per measured/driven qubit):
    1) Start at the initialization point.
    2) Optionally apply an x180 to the spectator qubit to prepare its |↑⟩ state.
    3) Step to the two-qubit exchange point and play the RF drive pulse on the driven qubit
       while sweeping the drive frequency.
    4) Step to the measurement point and read out the driven qubit.

The CROT spectroscopy works by:
    - At each virtual barrier/exchange voltage, the exchange coupling J varies.
    - The driven qubit resonance splits into two frequencies (f_↓ and f_↑) separated by J.
    - Sweeping frequency vs barrier voltage produces a chevron-like pattern showing J(V_barrier).
    - The optimal exchange point and CROT drive frequencies can be extracted from this 2D map.

The CROT gate is equivalent to a CNOT gate up to single-qubit rotations. For high-fidelity CROT gates,
the Zeeman energy difference between qubits must be much larger than the exchange coupling J, ensuring
addressability and avoiding off-resonant rotations.

Prerequisites:
    - Having calibrated single-qubit gates (π and π/2 pulses) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).
    - Having set the appropriate flux/gate voltages to enable exchange coupling between the qubits.

Before proceeding to the next node:
    - Extract the exchange coupling J from the frequency shift between the two resonance peaks.
    - Identify the conditional resonance frequencies f_↓ and f_↑ for CROT gate implementation.
    - Verify that J is sufficiently large for the desired gate speed but small enough for addressability.

State update:
    - exchange_coupling_J
    - crot_frequency_down
    - crot_frequency_up
"""


node = QualibrationNode[Parameters, Quam](
    name="15_crot_spectroscopy",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubit_pairs = ["q1q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)

    for gate_set_id in {qp.voltage_sequence.gate_set.id for qp in qubit_pairs}:
        node.machine.reset_voltage_sequence(gate_set_id)

    exchange_min = node.parameters.exchange_min
    exchange_max = node.parameters.exchange_max
    exchange_points = node.parameters.exchange_points
    exchange_step = (exchange_max - exchange_min) / exchange_points

    exchange_array = np.arange(exchange_min, exchange_max, exchange_step)


    u = unit(coerce_to_integer=True)

    esr_frequency_min = int(node.parameters.esr_frequency_min) * u.MHz
    esr_frequency_max = int(node.parameters.esr_frequency_max) * u.MHz
    esr_frequency_points = node.parameters.esr_frequency_points
    esr_frequency_step = int(
        (esr_frequency_max - esr_frequency_min) / esr_frequency_points
    )
    esr_frequency_array = np.arange(
        esr_frequency_min, esr_frequency_max, esr_frequency_step
    )

    spectator_x180_values = [True, False]
    measured_qubit_values = ["target", "control"]

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "measured_qubit": xr.DataArray(
            measured_qubit_values,
            attrs={"long_name": "measured qubit", "units": ""},
        ),
        "spectator_x180": xr.DataArray(
            spectator_x180_values,
            attrs={"long_name": "x180 on spectator qubit", "units": "bool"},
        ),
        "exchange": xr.DataArray(
            exchange_array, attrs={"long_name": "voltage", "units": "V"}
        ),
        "esr_frequency": xr.DataArray(
            esr_frequency_array, attrs={"long_name": "frequency", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw_pairs(node)

        n = declare(int)
        n_st = declare_output_stream()

        exchange = declare(fixed)
        esr_frequency = declare(int)

        p2, p1, parity_streams = declare_parity_streams(node, qubit_pairs)

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:

                # measured_qubit == "target": drive the target qubit, flip the
                #   control (spectator) with x180, and read out the target.
                # measured_qubit == "control": the symmetric experiment — drive
                #   the control qubit, flip the target (spectator), read out the
                #   control.
                for measure_target in [True, False]:
                    if measure_target:
                        drive_qubit = qubit_pair.qubit_target
                        spectator_qubit = qubit_pair.qubit_control
                        measured_qubit = qubit_pair.qubit_target
                    else:
                        drive_qubit = qubit_pair.qubit_control
                        spectator_qubit = qubit_pair.qubit_target
                        measured_qubit = qubit_pair.qubit_control

                    intermediate_frequency = drive_qubit.xy.intermediate_frequency
                    spectator_if = spectator_qubit.xy.intermediate_frequency

                    for apply_x180 in spectator_x180_values:
                        with for_(*from_array(exchange, exchange_array)):
                            with for_(*from_array(esr_frequency, esr_frequency_array)):

                                drive_qubit.xy.update_frequency(intermediate_frequency)
                                reset_frame(drive_qubit.xy.name)

                                qubit_pair.initialize()

                                align()

                                if apply_x180:
                                    spectator_qubit.xy.update_frequency(spectator_if)
                                    spectator_qubit.x180()

                                    align()

                                drive_qubit.xy.update_frequency(esr_frequency + intermediate_frequency)
                                reset_frame(drive_qubit.xy.name)

                                align()

                                qubit_pair.crot(
                                    point={
                                        qubit_pair.quantum_dot_pair.barrier_gate.id: exchange
                                    },
                                    duration=node.parameters.duration,
                                    ramp_duration = node.parameters.ramp_duration,
                                    drive_target=measure_target,
                                )

                                a2 = measured_qubit.measure()

                                align()

                                qubit_pair.crot.balance()

                                align()

                                qubit_pair.voltage_sequence.ramp_to_zero()
                                align()

                                assign(p2, Cast.to_int(a2))
                                save_parity_measurement(node, qubit_pair.name, p1, p2, parity_streams)

                    drive_qubit.xy.update_frequency(intermediate_frequency)


        with stream_processing():
            n_st.save("n")

            n_measured_qubit = len(measured_qubit_values)
            n_spectator_x180 = len(spectator_x180_values)
            n_exchange = len(exchange_array)
            n_esr_frequency = len(esr_frequency_array)
            for qubit_pair in qubit_pairs:
                buffer_parity_streams(node, qubit_pair.name, parity_streams, n_measured_qubit, n_spectator_x180, n_exchange, n_esr_frequency)


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect(timeout=node.parameters.timeout)
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [qp.name for qp in node.namespace["qubit_pairs"]],
        parity_pre_measurement=False,
        item_dim="qubit_pair",
        sweep_dims=("measured_qubit", "spectator_x180", "exchange", "esr_frequency"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data: fit Lorentzians to extract CROT frequencies and exchange coupling J."""
    ds_fit, fit_results = fit_raw_data(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        name: ("successful" if r["success"] else "failed")
        for name, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted CROT spectroscopy data."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubit_pairs"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["figure"] = fig
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    with node.record_state_updates():
        for qubit_pair in node.namespace["qubit_pairs"]:
            if not node.results["fit_results"][qubit_pair.name]["success"]:
                continue
            fit_result = node.results["fit_results"][qubit_pair.name]
            qubit_pair.exchange_coupling_J = fit_result["exchange_coupling_J"]
            qubit_pair.crot_frequency_down = fit_result["crot_frequency_down"]
            qubit_pair.crot_frequency_up = fit_result["crot_frequency_up"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
