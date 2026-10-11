# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode

from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from quam.components.pulses import SquareReadoutPulse
from quam_config import Quam

from calibration_utils.resonator_linewidth import (
    CHI_MISMATCH_WARNING_FRACTION,
    SKIP_IN_LIFETIMES,
    FitParameters,
    Parameters,
    choose_probe_timing,
    fit_raw_data,
    log_fitted_results,
    plot_circle_fit,
    plot_dispersive_shift,
    plot_magnitude_with_fit,
    process_raw_dataset,
)

# %% {Node initialisation}
description = """
        RESONATOR LINEWIDTH AND DISPERSIVE SHIFT (NOTCH-PORT CIRCLE FIT)
Sweeps a low-power probe across the resonator, with the qubit in |0> and optionally |1>, and fits a
notch-port circle to each complex trace. This gives kappa split into kappa_ext (to the amplifier) and
kappa_int (lost), and chi as half the |0>-|1> splitting. kappa_ext / kappa_tot bounds the collection
efficiency; 2|chi| / kappa = 1 is the readout SNR optimum.

The probe is a temporary pulse that skips the resonator fill-up (default 8 / kappa) and integrates the
next window (default min(2 us, T1 / 10)); integrating the fill-up biases kappa high. A linear tilt of the
background |S21| and the cable delay are fitted from the data.

Prerequisites:
    - Readout frequency (02a_resonator_spectroscopy) and time of flight (01a_time_of_flight).
    - For |1>: the x180 pulse (03a_qubit_spectroscopy, 04b_power_rabi).

State update (all in Hz):
    - qubit.resonator.extras: kappa_ext_hz, kappa_int_hz, kappa_ext_excited_hz, kappa_int_excited_hz,
      kappa_ground_hz, kappa_excited_hz, f_r_ground_hz, f_r_excited_hz (used by node 25).
    - qubit.chi (also written by node 08a; this fit is the more precise one), with qubit.extras["chi_source"]
      and qubit.resonator.extras["kappa_source"] naming the node that wrote them.
The fitted resonance frequency is only reported; node 02a owns the readout frequency.
"""

node = QualibrationNode[Parameters, Quam](
    name="24_resonator_kappa_chi_circle_fit",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)

PROBE_OPERATION = "linewidth_probe"
"""Temporary low-amplitude probe pulse built per qubit in create_qua_program (zero integration weights
during the fill-up skip) and removed again in save_results, so it never reaches the saved state."""


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]) -> None:
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)
    probe_scale = node.parameters.probe_amplitude_scale

    # Dedicated probe per qubit: the readout amplitude scaled down, long enough to skip the fill-up and
    # then integrate a steady-state window. Integration weights are zero during the skip.
    probe_timing = {}
    for qubit in qubits:
        skip_ns, window_ns, source = choose_probe_timing(qubit, node.parameters)
        readout = qubit.resonator.operations["readout"]
        qubit.resonator.operations[PROBE_OPERATION] = SquareReadoutPulse(
            length=skip_ns + window_ns,
            amplitude=readout.amplitude * probe_scale,
            integration_weights=[(0.0, skip_ns), (1.0, window_ns)] if skip_ns > 0 else [(1.0, window_ns)],
            integration_weights_angle=readout.integration_weights_angle,
        )
        probe_timing[qubit.name] = (skip_ns, window_ns)
        node.log(f"{qubit.name} probe: {source}.")
    node.namespace["probe_timing"] = probe_timing

    # The prepared states, innermost in the sweep. Measuring |0> and |1> back to back at each
    # frequency means slow drift subtracts out of the splitting, which is what chi is made of.
    states = [0, 1] if node.parameters.measure_excited_state else [0]

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
        "state": xr.DataArray(states, attrs={"long_name": "prepared qubit state", "units": ""}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for prepared_state in states:
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            rr = qubit.resonator
                            rr.update_frequency(df + rr.intermediate_frequency)
                            if prepared_state == 1:
                                qubit.xy.play("x180")
                                qubit.align()
                            rr.measure(PROBE_OPERATION, qua_vars=(I[i], Q[i]))
                            rr.wait(rr.depletion_time * u.ns)
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(states)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(states)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
    """Connect to the QOP, execute the QUA program and store the raw data in "ds_raw"."""
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
    # Kept on the dataset so the volts conversion and the skip check also work for a reloaded run.
    timing = node.namespace["probe_timing"]
    names = list(dataset.qubit.values)
    node.results["ds_raw"] = dataset.assign_coords(
        probe_skip_ns=("qubit", [timing[name][0] for name in names]),
        probe_window_ns=("qubit", [timing[name][1] for name in names]),
    )


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Fit the circle to each resonance and report which qubits failed and why."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node)
    fit_results: dict[str, FitParameters]

    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)

    ds_raw = node.results["ds_raw"]
    for name, result in node.results["fit_results"].items():
        if not result["success"]:
            continue
        # The skip came from the stored kappa; if the fitted kappa needs a longer one, kappa is biased high.
        needed_ns = SKIP_IN_LIFETIMES * result["photon_lifetime_ns"]
        if "probe_skip_ns" in ds_raw.coords and float(ds_raw.probe_skip_ns.sel(qubit=name)) < 0.75 * needed_ns:
            node.log(
                f"WARNING {name}: the probe skip is shorter than {SKIP_IN_LIFETIMES}/κ = {needed_ns:.0f} ns for "
                f"the fitted κ, so κ may be biased high. Re-run; the next run derives the skip from this κ."
            )
        mismatch = result["chi_mismatch_fraction"]
        if np.isfinite(mismatch) and mismatch > CHI_MISMATCH_WARNING_FRACTION:
            node.log(
                f"WARNING {name}: fitted χ {1e-6 * result['chi_hz']:+.4f} MHz differs from the stored "
                f"{1e-6 * result['stored_chi_hz']:+.4f} MHz (e.g. from node 08a) by {100 * mismatch:.0f}%."
            )

    node.outcomes = {
        name: ("successful" if result["success"] else "failed") for name, result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Plot the circle fit over the measured complex data, and the magnitude against detuning."""
    qubits = node.namespace["qubits"]
    figures = {
        "circle_fit": plot_circle_fit(node.results["ds_raw"], qubits, node.results["ds_fit"]),
        "magnitude": plot_magnitude_with_fit(node.results["ds_raw"], qubits, node.results["ds_fit"]),
    }
    if node.parameters.measure_excited_state:
        figures["dispersive_shift"] = plot_dispersive_shift(node.results["ds_raw"], qubits, node.results["ds_fit"])
    plt.show()
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]) -> None:
    """Write the linewidths into the resonator extras, and chi onto the qubit."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if node.outcomes[qubit.name] == "failed":
                continue
            result = node.results["fit_results"][qubit.name]
            qubit.resonator.extras["kappa_ext_hz"] = float(result["kappa_ext_hz"])
            qubit.resonator.extras["kappa_int_hz"] = float(result["kappa_int_hz"])
            qubit.resonator.extras["kappa_source"] = node.name
            # Total linewidth per qubit state, the form a state-aware pulse such as DRACHMA takes.
            qubit.resonator.extras["kappa_ground_hz"] = float(result["kappa_tot_hz"])
            if not result["excited_state_measured"]:
                continue
            qubit.resonator.extras["kappa_excited_hz"] = float(result["kappa_tot_excited_hz"])
            # chi goes on the qubit, in the field node 08a also writes.
            qubit.chi = float(result["chi_hz"])
            qubit.extras["chi_source"] = node.name
            # The |1> linewidths go next to the |0> ones, so that a resonator whose decay depends on
            # the qubit state shows it rather than hiding behind a single number.
            qubit.resonator.extras["kappa_ext_excited_hz"] = float(result["kappa_ext_excited_hz"])
            qubit.resonator.extras["kappa_int_excited_hz"] = float(result["kappa_int_excited_hz"])
            # The two resonances themselves, so that a node holding a tone on this resonator can say
            # where that tone sits relative to them. Measuring a detuning against the readout
            # frequency, which is where such a tone usually sits, only ever returns chi back again.
            qubit.resonator.extras["f_r_ground_hz"] = float(result["resonance_frequency"])
            qubit.resonator.extras["f_r_excited_hz"] = float(result["resonance_frequency_excited"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]) -> None:
    # The probe operation is temporary; it must not reach the saved state.
    for qubit in node.namespace.get("qubits", []):
        qubit.resonator.operations.pop(PROBE_OPERATION, None)
    node.save()


# %%
