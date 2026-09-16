"""Readout quantum efficiency node, following arXiv:1711.05336."""

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
from quam_config import Quam
from calibration_utils.readout_quantum_efficiency import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_efficiency_vs_frequency,
    plot_efficiency_map,
    plot_snr_and_dephasing,
    plot_raw_fringes,
    plot_stark_phase,
    plot_fringe_amplitude_vs_power,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        READOUT QUANTUM EFFICIENCY
Measures the quantum efficiency eta of the readout chain over a 2D sweep of the measurement
pulse frequency and amplitude, following the method of C.C. Bultink et al.,
"General method for extracting the quantum efficiency of dispersive qubit readout in circuit
QED", arXiv:1711.05336 (Fig. 1b and 1c).

Set `sweep_frequency = False` to drop the frequency axis altogether: the measurement pulse then
stays at the calibrated readout frequency and only the amplitude is swept, which is the fast way
to get eta at the current setpoint (the detuning axis is kept, with a single point at 0 Hz, so the
analysis is unchanged).

The frequency and amplitude loops are the outer ones. Within each (frequency, amplitude) cell the
two sequences below run back to back, the first averaged in hardware and the second kept shot by
shot, so both halves see the same conditions for the cell they belong to:
    b) Dephasing: x90 - [measurement pulse at (f, eps)] - depletion wait - x90 with a swept
       azimuthal angle - strong readout. The fringe amplitude versus that angle gives the qubit
       coherence |rho01(eps)|, while its phase offset is the deterministic AC-Stark shift; the
       sweep exists to tell those two apart.
    c) SNR: prepare |0> or |1> and integrate the SAME measurement pulse single-shot, giving
       SNR = |mu_1 - mu_0| / sigma along the axis separating the two blobs.

The measurement-induced dephasing is exp(-beta) = |rho01(eps)| / |rho01(0)|, and the efficiency
is eta = SNR^2 / (4 beta). Since eta is amplitude independent in the linear regime, the reported
number per frequency comes from the paper's global fits over the amplitude axis,
SNR = a*eps and beta = eps^2/(2 sigma_m^2), giving eta = a^2 sigma_m^2 / 2. The point-by-point
ratio is also produced, as a check that the sweep stayed in the linear regime.

Optimal integration weights and active photon depletion, the other two steps of the paper's
method, are NOT implemented here: this node uses the existing readout weights and a passive
depletion wait. eta measured this way is therefore a lower bound on what the chain can deliver.

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 08a).
    - Having calibrated the qubit x90/x180 pulses (nodes 04b, 06a).
    - Having calibrated the readout discrimination threshold AND the confusion matrix
      (node 08b_readout_power_optimization or 07_iq_blobs) - the confusion matrix is used to
      undo readout assignment errors on the fringes.

State update:
    - None. eta characterises the amplification chain; it is not a knob, and the amplitude that
      maximises eta is generally not the one that maximises readout fidelity.
"""


node = QualibrationNode[Parameters, Quam](
    name="08c_readout_quantum_efficiency",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Override parameters for local debugging runs."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1"]
    node.parameters.reset_type = "active"
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots_ramsey
    n_shots_snr = node.parameters.num_shots_snr
    # The frequency sweep around the current readout frequency. With sweep_frequency = False the
    # axis collapses to the calibrated frequency itself: the dimension is kept so that the
    # analysis, which indexes by detuning throughout, needs no special case.
    if node.parameters.sweep_frequency:
        span = node.parameters.frequency_span_in_mhz * u.MHz
        step = node.parameters.frequency_step_in_mhz * u.MHz
        dfs = np.arange(-span / 2, +span / 2 + 0.5 * step, step)
    else:
        dfs = np.array([0])
    # eps = 0 is the reference the dephasing is measured against, so it is always in the grid.
    amps = np.linspace(0, node.parameters.max_amp_prefactor, node.parameters.num_amps + 1)
    # The frame rotation is expressed in turns; one full turn covers exactly one fringe.
    phases_in_turns = np.linspace(0, 2, node.parameters.num_phases, endpoint=False)

    # The frequency and amplitude loops are the outer ones: each (f, eps) cell is measured to
    # completion, first the whole dephasing experiment and then the whole SNR experiment, before
    # moving on. The two experiments therefore keep their own, independent shot counts. Note that
    # elapsed time now maps onto the frequency axis, so slow drift shows up as structure in
    # eta(f); the eps = 0 fringe, measured first in every cell, is the monitor for that.
    node.namespace["num_cells"] = len(dfs) * len(amps)

    # Only the Ramsey half is stacked by the XarrayDataFetcher; the single-shot half has a
    # different shape and is fetched separately below.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout detuning", "units": "Hz"}),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "measurement pulse amplitude prefactor"}),
        "phase": xr.DataArray(2 * np.pi * phases_in_turns, attrs={"long_name": "second pi/2 phase", "units": "rad"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        # The shot loops are innermost now, so `n` no longer tracks the run: the progress bar
        # follows completed (frequency, amplitude) cells instead.
        cell = declare(int)
        cell_st = declare_stream()
        df = declare(int)
        a = declare(fixed)
        phi = declare(fixed)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        Ig = [declare(fixed) for _ in range(num_qubits)]
        Qg = [declare(fixed) for _ in range(num_qubits)]
        Ie = [declare(fixed) for _ in range(num_qubits)]
        Qe = [declare(fixed) for _ in range(num_qubits)]
        Ig_st = [declare_stream() for _ in range(num_qubits)]
        Qg_st = [declare_stream() for _ in range(num_qubits)]
        Ie_st = [declare_stream() for _ in range(num_qubits)]
        Qe_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            assign(cell, 0)
            # A single-frequency run keeps this loop, with one iteration at df = 0: `from_array`
            # supports a one-element array, so nothing downstream needs a special case.
            with for_(*from_array(df, dfs)):
                with for_(*from_array(a, amps)):
                    # ---- Fig. 1b: dephasing under a variable-strength measurement ----
                    # The averaging loop is outside the phase loop, so every pass samples the
                    # whole fringe: drift within the cell then scales all phases alike and
                    # cancels in |rho01(eps)|/|rho01(0)|, instead of distorting the fringe shape.
                    with for_(n, 0, n < n_avg, n + 1):
                        with for_(*from_array(phi, phases_in_turns)):
                            for i, qubit in multiplexed_qubits.items():
                                reset_frame(qubit.xy.name)
                                qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                # Detune the measurement pulse; the resonator is idle here.
                                update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                                qubit.xy.play("x90")
                                qubit.align()
                                # The weak measurement itself: played, not demodulated - the
                                # information it carries is measured in the second half. It stays
                                # at the detuning set above, which is the whole point of the
                                # frequency axis: beta and the SNR must be measured at the SAME
                                # measurement-pulse frequency for their ratio to mean anything.
                                qubit.resonator.play("readout", amplitude_scale=a)
                                qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                                qubit.align()
                                # Sweeping this phase is what separates dephasing (fringe
                                # amplitude) from the AC-Stark shift (fringe phase).
                                qubit.xy.frame_rotation_2pi(phi)
                                qubit.xy.play("x90")
                                qubit.align()
                                # The strong readout must sit at the calibrated frequency,
                                # otherwise its threshold and confusion matrix do not apply.
                                update_frequency(qubit.resonator.name, qubit.resonator.intermediate_frequency)
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            align()

                    # ---- Fig. 1c: SNR of the same variable-strength measurement ----
                    # Single shots, kept one by one: nothing here is averaged in hardware.
                    with for_(n, 0, n < n_shots_snr, n + 1):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                            qubit.resonator.measure("readout", qua_vars=(Ig[i], Qg[i]), amplitude_scale=a)
                            save(Ig[i], Ig_st[i])
                            save(Qg[i], Qg_st[i])
                            # Restore the calibrated frequency: the reset macro may itself
                            # read the resonator out.
                            update_frequency(qubit.resonator.name, qubit.resonator.intermediate_frequency)
                        align()

                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x180")
                            qubit.align()
                            update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                            qubit.resonator.measure("readout", qua_vars=(Ie[i], Qe[i]), amplitude_scale=a)
                            save(Ie[i], Ie_st[i])
                            save(Qe[i], Qe_st[i])
                            update_frequency(qubit.resonator.name, qubit.resonator.intermediate_frequency)
                        align()

                    assign(cell, cell + 1)
                    save(cell, cell_st)

        with stream_processing():
            cell_st.save("cell")
            for i in range(num_qubits):
                # Average over the Ramsey shot axis only; the sweep axes stay resolved.
                state_st[i].buffer(len(phases_in_turns)).buffer(n_avg).map(FUNCTIONS.average()).buffer(
                    len(amps)
                ).buffer(len(dfs)).save(f"state{i + 1}")
                for stream, label in (
                    (Ig_st[i], "Ig"),
                    (Qg_st[i], "Qg"),
                    (Ie_st[i], "Ie"),
                    (Qe_st[i], "Qe"),
                ):
                    stream.buffer(n_shots_snr).buffer(len(amps)).buffer(len(dfs)).save(f"{label}{i + 1}")


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
# Defined before the action: `run_action` executes at decoration time, so a helper declared
# further down the cell would not exist yet.
def fetch_single_shots(job, qubits, sweep_axes, n_shots_snr: int) -> xr.Dataset:
    """Fetch the single-shot half into its own dataset.

    Its shape differs from the averaged Ramsey half, and XarrayDataFetcher requires every
    handle it stacks to share one shape, so these handles are fetched on their own and the two
    halves merged afterwards.
    """
    n_detunings = sweep_axes["detuning"].size
    n_amps = sweep_axes["amp_prefactor"].size
    data_vars = {}
    for label in ("Ig", "Qg", "Ie", "Qe"):
        values = np.empty((len(qubits), n_shots_snr, n_detunings, n_amps))
        for i, _qubit in enumerate(qubits):
            handle = job.result_handles.get(f"{label}{i + 1}")
            handle.wait_for_all_values()
            fetched = handle.fetch_all()
            if getattr(fetched, "dtype", None) is not None and fetched.dtype.names:
                fetched = fetched["value"]
            # The stream buffers the shot axis innermost; the dataset wants it leading.
            values[i] = np.asarray(fetched).reshape(n_detunings, n_amps, n_shots_snr).transpose(2, 0, 1)
        data_vars[label] = (("qubit", "shot", "detuning", "amp_prefactor"), values)

    # `.values`, not the DataArrays themselves: the sweep axes are built without explicit dim
    # names, so they still carry xarray's default `dim_0`, which would collide here. The
    # XarrayDataFetcher renames those on its own copy only.
    return xr.Dataset(
        data_vars,
        coords={
            "qubit": sweep_axes["qubit"].values,
            "shot": np.arange(n_shots_snr),
            "detuning": sweep_axes["detuning"].values,
            "amp_prefactor": sweep_axes["amp_prefactor"].values,
        },
    )


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and store raw data in `ds_raw`."""
    qubits = node.namespace["qubits"]
    sweep_axes = node.namespace["sweep_axes"]
    num_cells = node.namespace["num_cells"]

    single_shot_handles = [f"{label}{i + 1}" for i in range(len(qubits)) for label in ("Ig", "Qg", "Ie", "Qe")]
    fetcher_cls = type(
        "QuantumEfficiencyFetcher",
        (XarrayDataFetcher,),
        {"ignore_handles": XarrayDataFetcher.ignore_handles + single_shot_handles},
    )

    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = fetcher_cls(job, sweep_axes)
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("cell", 0),
                num_cells,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
        ds_single_shots = fetch_single_shots(job, qubits, sweep_axes, node.parameters.num_shots_snr)
    # Register the raw dataset
    node.results["ds_raw"] = xr.merge([dataset, ds_single_shots])


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    # load_from_id rebuilds node.parameters from the SAVED run, which would revert any
    # re-fit knob the user changed to re-analyse loaded data. Snapshot the user's current
    # values for those knobs (+ load_data_id) and restore them after load.
    _refit_keep = {k: getattr(node.parameters, k) for k in ("load_data_id", "max_amp_for_fit", "linearity_rtol")}
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    for _k, _v in _refit_keep.items():
        setattr(node.parameters, _k, _v)
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse raw data and store fits in `ds_fit` and `fit_results`."""
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
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    qubits = node.namespace["qubits"]
    ds_raw, ds_fit = node.results["ds_raw"], node.results["ds_fit"]
    figures = {
        "efficiency_map": plot_efficiency_map(ds_fit, qubits),
        "snr_and_dephasing": plot_snr_and_dephasing(ds_fit, qubits),
        "fringes": plot_raw_fringes(ds_raw, ds_fit, qubits),
        "fringe_amplitude_vs_power": plot_fringe_amplitude_vs_power(ds_fit, qubits),
        "stark_phase": plot_stark_phase(ds_fit, qubits),
    }
    # A single-frequency run has nothing to show against frequency; eta is then reported by
    # log_fitted_results and visible in the two amplitude figures above.
    if ds_fit.sizes["detuning"] > 1:
        figures["quantum_efficiency"] = plot_efficiency_vs_frequency(ds_fit, qubits)
    plt.show()
    # Store the generated figures
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No state update.

    eta characterises the amplification chain rather than a setting of it, and the (frequency,
    amplitude) pair that maximises eta is generally not the one that maximises readout fidelity
    - that is what 08a_readout_frequency_optimization and 08b_readout_power_optimization are for.
    """
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results."""
    node.save()


# %%
