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

from quam_config import Quam

from calibration_utils.resonator_linewidth import (
    FitParameters,
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_circle_fit,
    plot_dispersive_shift,
    plot_magnitude_with_fit,
    process_raw_dataset,
)

# %% {Node initialisation}
description = """
        RESONATOR LINEWIDTH (NOTCH-PORT CIRCLE FIT)
Sweeps the readout frequency at a single low probe power and captures both quadratures, then fits a
notch-port circle to the complex trace. The fit separates the resonator linewidth into the part that
leaves through the feedline towards the amplifier, kappa_ext, and the part that is lost, kappa_int.

kappa_ext / kappa_tot is the hard ceiling on the collection efficiency of the resonator, so it is the
number every later efficiency measurement is compared against. The cable delay is fitted from the
data rather than assumed, because an assumed delay makes Q_c silently wrong.

The probe power is a scale factor on each qubit's own readout amplitude, so the absolute power
differs per qubit. The node records that power alongside Q_i, which matters because two-level-system
loss saturates with power and Q_i is only interpretable together with the power it was measured at.

Prerequisites:
    - Having calibrated the readout frequency (node 02a_resonator_spectroscopy.py).
    - Having calibrated the time of flight (node 01a_time_of_flight.py), used as the starting guess
      for the cable delay.

With measure_excited_state on, the sweep is repeated with the qubit prepared in |1>, the two states
measured back to back at each frequency so that slow drift subtracts out of the splitting. Half that
splitting is chi, and |chi|/kappa says whether the weak-dispersive formulas behind node 23b hold at
all. Both resonances are fitted separately, so a resonator whose decay depends on the qubit state
shows it instead of hiding behind one number.

Prerequisites (only with measure_excited_state):
    - Having calibrated the x180 pulse (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).

State update:
    - The external linewidth: qubit.resonator.extras["kappa_ext_hz"]
    - The internal linewidth: qubit.resonator.extras["kappa_int_hz"]
    - The same two with the qubit in |1>: qubit.resonator.extras["kappa_ext_excited_hz"] and
      ["kappa_int_excited_hz"]
    - The two dressed resonance frequencies: qubit.resonator.extras["f_r_ground_hz"] and
      ["f_r_excited_hz"], which node 23b needs to measure its tone detuning against something that is
      not the tone itself
    - The dispersive shift: qubit.chi
All in Hz. The fitted resonance frequency is reported but never written, so node 02a
stays the only owner of the readout frequency. chi is the one field here that another node also
writes, node 08a; this node takes it from a full complex fit of both resonances rather than from the
spacing of two magnitude minima on a discrete grid, so it is the better estimate.
"""

node = QualibrationNode[Parameters, Quam](
    name="23a_resonator_linewidth",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]) -> None:
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.qubits = ["q1", "q2"]
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
    # The probe power is a real-time amplitude scale, so nothing in the config is modified and there
    # is nothing to revert when the node finishes.
    probe_scale = node.parameters.probe_amplitude_scale

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
                            rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=probe_scale)
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
    node.results["ds_raw"] = dataset


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

    warn_fraction = node.parameters.kappa_mismatch_warning_fraction
    for name, result in node.results["fit_results"].items():
        mismatch = result["kappa_mismatch_fraction"]
        if result["success"] and np.isfinite(mismatch) and mismatch > warn_fraction:
            node.log(
                f"WARNING {name}: the fitted kappa_tot of {1e-6 * result['kappa_tot_hz']:.3f} MHz disagrees "
                f"with the stored {1e-6 * result['stored_kappa_hz']:.3f} MHz by {100 * mismatch:.0f}%, "
                f"which is more than the {100 * warn_fraction:.0f}% allowed. The resonance may not be a "
                f"single Lorentzian."
            )
        if not result["success"] or not result["excited_state_measured"]:
            continue
        chi_over_kappa = result["chi_over_kappa"]
        if np.isfinite(chi_over_kappa) and chi_over_kappa > node.parameters.max_chi_over_kappa:
            node.log(
                f"WARNING {name}: |χ|/κ is {chi_over_kappa:.3f}, above {node.parameters.max_chi_over_kappa:.2f}. "
                f"The weak-dispersive formulas behind node 23b assume χ ≪ κ, so the photon number it "
                f"reports will be wrong; the general steady-state expressions are needed instead."
            )
        chi_mismatch = result["chi_mismatch_fraction"]
        if np.isfinite(chi_mismatch) and chi_mismatch > node.parameters.chi_mismatch_warning_fraction:
            node.log(
                f"WARNING {name}: the fitted χ of {1e-6 * result['chi_hz']:+.4f} MHz disagrees with the "
                f"stored {1e-6 * result['stored_chi_hz']:+.4f} MHz by {100 * chi_mismatch:.0f}%. Everything "
                f"node 23b reports scales with χ, so the old photon numbers were wrong by that factor."
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
            if not result["excited_state_measured"]:
                continue
            # chi goes on the qubit, in the field node 08a also writes.
            qubit.chi = float(result["chi_hz"])
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
    node.save()
