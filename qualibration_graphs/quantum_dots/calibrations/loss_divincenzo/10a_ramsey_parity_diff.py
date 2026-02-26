# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey import RamseyParameters
from calibration_utils.ramsey_parity_diff import (
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.common_utils.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.sweep import get_idle_times_in_clock_cycles

# %% {Node initialisation}
description = """
        RAMSEY PARITY DIFFERENCE (±δ triangulation)
This sequence performs a Ramsey measurement at two symmetric detunings ±δ from the qubit
intermediate frequency.  At each detuning the idle time between two π/2 pulses is swept,
producing a damped-cosine oscillation whose frequency equals the true detuning from resonance.

By fitting both traces independently, the analysis triangulates the residual frequency offset:
    Δ = (f₋ − f₊) / 2
This resolves the sign ambiguity inherent in a single-detuning measurement and provides a
robust correction for the qubit drive frequency.

The sequence uses voltage sequences to navigate through voltage space (empty - initialization -
measurement) using OPX channels on the fast lines of the bias-tees.  At each idle time the
parity is measured before (P1) and after (P2) the qubit pulse, and the parity difference
(P_diff) is calculated.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X90 pulse amplitude and frequency).

State update:
    - The qubit intermediate frequency (Larmor frequency correction).
"""


node = QualibrationNode[RamseyParameters, Quam](
    name="10a_ramsey_parity_diff",
    description=description,
    parameters=RamseyParameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[RamseyParameters, Quam]):
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[RamseyParameters, Quam]):
    """Create the sweep axes and generate the QUA program.

    Sweeps idle time at two symmetric detunings [+δ, −δ] from the qubit
    intermediate frequency, producing a 2-D dataset (detuning × tau).
    """
    u = unit(coerce_to_integer=True)

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    detuning = node.parameters.frequency_detuning_in_mhz * u.MHz
    detuning_values = np.array([detuning, -detuning])
    # Idle time sweep (in clock cycles of 4ns)
    tau_values = get_idle_times_in_clock_cycles(node.parameters)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            detuning_values,
            attrs={"long_name": "frequency detuning", "units": "Hz"},
        ),
        "tau": xr.DataArray(
            tau_values * 4, attrs={"long_name": "idle time", "units": "ns"}
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

                with for_(*from_array(df, detuning_values)):
                    for i, qubit in batched_qubits.items():
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency + df)

                    with for_(*from_array(t, tau_values)):
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
                            qubit.initialize(duration=node.parameters.gap_wait_time_in_ns + op_length * 2 + 4 * t)

                        # Step 3: X90 - idle - X90
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

            n_detuning = len(detuning_values)
            n_tau = len(tau_values)
            for qubit in qubits:
                p1_st[qubit.name].buffer(n_detuning, n_tau).average().save(f"p1_{qubit.name}")
                p2_st[qubit.name].buffer(n_detuning, n_tau).average().save(f"p2_{qubit.name}")
                pdiff_st[qubit.name].buffer(n_detuning, n_tau).average().save(f"pdiff_{qubit.name}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[RamseyParameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[RamseyParameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[RamseyParameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[RamseyParameters, Quam]):
    """Analyse the raw data to extract Ramsey frequency and T2*."""
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {qname: ("successful" if r["success"] else "failed") for qname, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[RamseyParameters, Quam]):
    """Plot the raw and fitted data."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
    )
    node.results["figure"] = fig


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[RamseyParameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.xy.intermediate_frequency -= fit_result["freq_offset"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[RamseyParameters, Quam]):
    node.save()
