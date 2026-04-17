# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from quam_builder.architecture.quantum_dots.operations.names import VoltagePointName

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

In exchange-coupled spin qubits, the resonance frequency of a target qubit depends on the spin state
of a control qubit. When the control qubit is in |↓⟩ vs |↑⟩, the target qubit's resonance frequency
shifts by the exchange coupling strength J. This state-dependent frequency shift enables conditional
quantum operations - the foundation of two-qubit gates in the Loss-DiVincenzo architecture.

This measurement performs a 2D sweep of drive frequency vs virtual barrier gate (or virtual exchange voltage)
to map out the exchange coupling as a function of the inter-dot tunnel coupling.

The QUA program sequence:
    1) Start at the initialization point.
    2) Step to the two-qubit exchange point using sticky elements (virtual barrier/exchange voltage).
    3) Apply the RF drive pulse to the target qubit while sweeping the drive frequency.
    4) Step to the initialization (operation) point.
    5) Step to the measurement point and read out the two-qubit state via parity readout.

The CROT spectroscopy works by:
    - At each virtual barrier/exchange voltage, the exchange coupling J varies.
    - The target qubit resonance splits into two frequencies (f_↓ and f_↑) separated by J.
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
    name="14a_crot_spectroscopy_parity_diff", description=description, parameters=Parameters()
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

    esr_frequency_min = int(node.parameters.esr_frequency_min)
    esr_frequency_max = int(node.parameters.esr_frequency_max)
    esr_frequency_points = node.parameters.esr_frequency_points
    esr_frequency_step = int((esr_frequency_max - esr_frequency_min) / esr_frequency_points)
    esr_frequency_array = np.arange(esr_frequency_min, esr_frequency_max, esr_frequency_step)

    control_x180_values = [False, True]

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair.name for pair in qubit_pairs]),
        "control_x180": xr.DataArray(
            control_x180_values,
            attrs={"long_name": "x180 on control qubit", "units": "bool"},
        ),
        "exchange": xr.DataArray(exchange_array, attrs={"long_name": "voltage", "units": "V"}),
        "esr_frequency": xr.DataArray(esr_frequency_array, attrs={"long_name": "frequency", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()

        exchange = declare(fixed)
        esr_frequency = declare(int)

        p1 = declare(int)
        p2 = declare(int)

        p1_st = {qubit_pair.name: declare_stream() for qubit_pair in qubit_pairs}
        p2_st = {qubit_pair.name: declare_stream() for qubit_pair in qubit_pairs}
        pdiff_st = {qubit_pair.name: declare_stream() for qubit_pair in qubit_pairs}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for qubit_pair in qubit_pairs:
                for apply_x180 in control_x180_values:
                    with for_(*from_array(exchange, exchange_array)):
                        with for_(*from_array(esr_frequency, esr_frequency_array)):
                            # ---------------------------------------------------------
                            # Step 1a: Empty - step to empty point (fixed duration)
                            # ---------------------------------------------------------
                            qubit_pair.empty()
                            align()

                            assign(p1, Cast.to_int(qubit_pair.measure()))
                            align()

                            # ---------------------------------------------------------
                            # Step 2: Initialize - load electron into dots (fixed duration)
                            # ---------------------------------------------------------
                            qubit_pair.initialize()
                            align()

                            # ---------------------------------------------------------
                            # Step 3: Optionally flip control qubit before CROT drive
                            # ---------------------------------------------------------
                            if apply_x180:
                                qubit_pair.qubit_control.x180()
                                align()

                            # ---------------------------------------------------------
                            # Step 4: CROT spectroscopy drive on target qubit
                            # ---------------------------------------------------------
                            qubit_pair.crot(
                                voltage_point={qubit_pair.quantum_dot_pair.barrier_gate.id: exchange},
                                esr_frequency=esr_frequency,
                                duration=node.parameters.duration,
                                hold_time=node.parameters.hold_duration,
                            )
                            align()

                            qubit_pair.step_to_point(VoltagePointName.INITIALIZE, duration=16)

                            # ---------------------------------------------------------
                            # Step 5: Measure
                            # ---------------------------------------------------------
                            assign(p2, Cast.to_int(qubit_pair.measure()))
                            align()

                            qubit_pair.voltage_sequence.apply_compensation_pulse()

                            save(p1, p1_st[qubit_pair.name])
                            save(p2, p2_st[qubit_pair.name])

                            with if_(p1 == p2):
                                save(0, pdiff_st[qubit_pair.name])
                            with else_():
                                save(1, pdiff_st[qubit_pair.name])

        with stream_processing():
            n_st.save("n")

            n_control_x180 = len(control_x180_values)
            n_exchange = len(exchange_array)
            n_esr_frequency = len(esr_frequency_array)
            for qubit_pair in qubit_pairs:
                p1_st[qubit_pair.name].buffer(n_control_x180, n_exchange, n_esr_frequency).average().save(
                    f"p1_{qubit_pair.name}"
                )
                p2_st[qubit_pair.name].buffer(n_control_x180, n_exchange, n_esr_frequency).average().save(
                    f"p2_{qubit_pair.name}"
                )
                pdiff_st[qubit_pair.name].buffer(n_control_x180, n_exchange, n_esr_frequency).average().save(
                    f"pdiff_{qubit_pair.name}"
                )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
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
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data: fit Lorentzians to extract CROT frequencies and exchange coupling J."""
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node.namespace["qubit_pairs"])
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {name: ("successful" if r["success"] else "failed") for name, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted CROT spectroscopy data."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubit_pairs"],
        node.results.get("fit_results", {}),
    )
    plt.show()
    node.results["figures"] = {"crot_spectroscopy": fig}


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
