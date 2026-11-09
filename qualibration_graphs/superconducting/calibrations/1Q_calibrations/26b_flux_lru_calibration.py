"""Flux-activated f-LRU calibration: |f0> -> |e1> via a flux tone at half the f0-e1 frequency."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.common_utils import ensure_gef_readout_pulse, reset_for_gef
from calibration_utils.lru_calibration import (
    FLUX_LRU_OPERATION,
    FluxParameters,
    add_flux_lru_elements,
    assign_flux_lru_coords,
    fit_raw_data,
    flux_lru_element,
    flux_tone_headroom_v,
    get_amp_factors,
    get_detunings,
    log_fitted_results,
    plot_lru_maps,
    plot_pf_along_line,
    process_raw_dataset,
)
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
from quam_config.lru_components import f0e1_frequency

# %% {Description}
description = """
        FLUX-ACTIVATED f-LRU CALIBRATION (|f0> -> |e1>)
Leakage removal: |f0> -> |e1> (transmon level, resonator photon number), after which the photon leaks
out of the resonator and leaves the qubit in |e0>. |f0> and |e1> have the same parity, so a microwave
drive on the XY line cannot couple them; a flux drive can. Near the sweet spot the qubit frequency is
quadratic in flux, so a flux tone at f_m modulates the transmon at 2 * f_m, and the transition is driven
when 2 * f_m = f_r - f_12 (about 1.5 GHz, so f_m is about 750 MHz).

The tone is played by a flux-LRU element added to this run's config on the qubit's flux port, which
is switched to 2 GS/s for the run. The qubit's own flux pulses are left out of the run (they are
sampled at 1 GS/s); only its static flux offset is used. The QUAM state is not modified.

Sequence, for each duration, amplitude and frequency:
    1. Prepare |f> (x180, then EF_x180 at the e-f frequency).
    2. Play the flux tone at the swept amplitude, frequency f_m and duration.
    3. Wait for the resonator to empty (wait_after_lru_ns, by default its depletion time).
    4. Measure g/e/f with the readout_GEF pulse.

The resonance appears as a dip in P(f) and a ridge in P(e) at 2 * f_m = f0e1 + c2 * A^2, where A is the
tone amplitude (the quadratic flux dependence also shifts the mean qubit frequency). The node fits that
line through the P(f) minima, reads P(f) along it, and selects the duration and amplitude reaching the
lowest P(f).

Prerequisites:
    - Calibrated x180 and EF_x180 (nodes 04b and 13) and anharmonicity (node 12).
    - Calibrated qubit frequency (nodes 03a and 06a) and resonator frequency (node 02a), since both
      enter the f0-e1 prediction.
    - Calibrated g/e/f readout: readout_GEF, GEF_frequency_shift and gef_centers (nodes 14a/14b/15).
    - The qubit near its flux sweet spot (quadratic flux dependence).

State update:
    - qubit.extras: flux_lru_frequency (Hz, the f0-e1 drive frequency; the flux tone is at half of it),
      flux_lru_amplitude_v (V at the port) and flux_lru_duration_ns.
"""


# Be sure to include [FluxParameters, Quam] so the node has proper type hinting
node = QualibrationNode[FluxParameters, Quam](
    name="26b_flux_lru_calibration",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=FluxParameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[FluxParameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


def _flux_lru_config(node: QualibrationNode[FluxParameters, Quam]):
    """Config of this run: the QUAM config plus a flux-LRU element per qubit, flux ports at 2 GS/s."""
    return add_flux_lru_elements(
        node.machine.generate_config(), node.namespace["qubits"], node.parameters.flux_amplitude_v
    )


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[FluxParameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    if node.parameters.reset_type == "active":
        raise ValueError("'active' reset is not GEF-aware; use 'thermal' or 'active_gef'")
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    readout_operation = node.parameters.readout_operation
    amp_factors = get_amp_factors(node.parameters)
    detunings = get_detunings(node.parameters)
    durations_ns = np.array(node.parameters.pulse_durations_ns, dtype=int)
    # f0e1 drive frequency at the sweep centre, per qubit; the flux tone runs at half of it
    node.namespace["centres"] = centres = np.array([int(f0e1_frequency(q)) for q in qubits])
    peak_v = abs(node.parameters.flux_amplitude_v) * np.max(np.abs(amp_factors))
    for qubit, centre in zip(qubits, centres):
        if qubit.resonator.gef_centers is None:
            raise ValueError(f"{qubit.name} has no gef_centers: calibrate the g/e/f readout with node 15 first.")
        headroom_v = flux_tone_headroom_v(qubit)
        if peak_v > headroom_v:
            raise ValueError(
                f"{qubit.name}: the flux tone reaches {peak_v:.3f} V, but only {headroom_v:.3f} V fits on top of the "
                f"{qubit.z.joint_offset:.3f} V flux offset ({qubit.z.opx_output.output_mode} output mode); "
                "lower flux_amplitude_v or max_amp_factor."
            )
        top = (centre + detunings.max()) / 2
        if top >= 0.95e9:
            raise ValueError(
                f"{qubit.name}: the flux tone reaches {top / 1e6:.0f} MHz, too close to the 1 GHz Nyquist limit."
            )
    ensure_gef_readout_pulse(qubits, log_callable=node.log)

    # Register the sweep axes to be added to the dataset when fetching data (outer to inner loop)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "duration": xr.DataArray(durations_ns, attrs={"long_name": "flux-tone duration", "units": "ns"}),
        "amp_prefactor": xr.DataArray(amp_factors, attrs={"long_name": "flux-tone amplitude pre-factor"}),
        "detuning": xr.DataArray(detunings, attrs={"long_name": "f0e1 drive detuning", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        state = [declare(int) for _ in range(num_qubits)]
        is_e = [declare(bool) for _ in range(num_qubits)]
        is_f = [declare(bool) for _ in range(num_qubits)]
        fm = [declare(int) for _ in range(num_qubits)]
        e_st = [declare_stream() for _ in range(num_qubits)]
        f_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # flux-tone duration in clock cycles
        a = declare(fixed)  # flux-tone amplitude pre-factor
        df = declare(int)  # f0e1 drive detuning in Hz

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, (durations_ns // 4).tolist()):
                    with for_(*from_array(a, amp_factors)):
                        with for_(*from_array(df, detunings)):
                            # Qubit initialization, leaving the resonator at the GEF readout frequency
                            for i, qubit in multiplexed_qubits.items():
                                reset_for_gef(qubit, node.parameters.reset_type, node.parameters.simulate, u)
                                # Flux tone at half the f0e1 drive frequency
                                assign(fm[i], (int(centres[i]) + df) >> 1)
                            align()

                            # Prepare |f> as node 15 does (no frequency update right before x180), then the flux tone
                            for i, qubit in multiplexed_qubits.items():
                                element = flux_lru_element(qubit)
                                qubit.xy.play("x180")
                                qubit.xy.update_frequency(
                                    qubit.xy.intermediate_frequency - abs(qubit.anharmonicity), keep_phase=True
                                )
                                qubit.xy.play("EF_x180")
                                qubit.xy.update_frequency(qubit.xy.intermediate_frequency, keep_phase=True)
                                align(qubit.xy.name, element)
                                update_frequency(element, fm[i])
                                play(FLUX_LRU_OPERATION * amp(a), element, duration=t)
                                wait_ns = node.parameters.wait_after_lru_ns
                                if wait_ns is None:
                                    wait_ns = qubit.resonator.depletion_time
                                if wait_ns >= 16:
                                    wait(int(wait_ns) // 4, element)
                                align(qubit.xy.name, qubit.resonator.name, element)

                            # g/e/f readout
                            for i, qubit in multiplexed_qubits.items():
                                qubit.readout_state_gef(state[i], pulse_name=readout_operation)
                                # Booleans, since QUA can only save variables; averaging them gives P(e) and P(f)
                                assign(is_e[i], state[i] == 1)
                                assign(is_f[i], state[i] == 2)
                                save(is_e[i], e_st[i])
                                save(is_f[i], f_st[i])
                            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                for stream, name in ((e_st[i], "pe"), (f_st[i], "pf")):
                    stream.boolean_to_int().buffer(len(detunings)).buffer(len(amp_factors)).buffer(
                        len(durations_ns)
                    ).average().save(f"{name}{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[FluxParameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, _flux_lru_config(node), node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[FluxParameters, Quam]):
    """Connect, execute QUA program, fetch raw data and store as dataset 'ds_raw'."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, _flux_lru_config(node), timeout=node.parameters.timeout) as qm:
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
    # Register the raw dataset, with the flux-tone amplitudes and f0e1 frequencies needed to reload it
    node.results["ds_raw"] = assign_flux_lru_coords(
        dataset, node.namespace["qubits"], node.parameters.flux_amplitude_v, node.namespace["centres"]
    )


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[FluxParameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[FluxParameters, Quam]):
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
def plot_data(node: QualibrationNode[FluxParameters, Quam]):
    """Plot the P(e) maps with the fitted resonance line, and P(f) along that line."""
    ds, ds_fit, qubits = node.results["ds_raw"], node.results["ds_fit"], node.namespace["qubits"]
    figures = {f"map_{duration:.0f}ns": plot_lru_maps(ds, qubits, ds_fit, duration) for duration in ds.duration.values}
    figures["pf_along_line"] = plot_pf_along_line(ds, qubits, ds_fit, node.results["fit_results"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[FluxParameters, Quam]):
    """Record the calibrated flux-LRU frequency, amplitude and duration of every successful qubit."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit_result = node.results["fit_results"][q.name]
            q.extras["flux_lru_frequency"] = float(fit_result["frequency"])
            q.extras["flux_lru_amplitude_v"] = float(fit_result["amplitude"])
            q.extras["flux_lru_duration_ns"] = int(fit_result["duration_ns"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[FluxParameters, Quam]):
    """Save all node results and state updates."""
    node.save()


# %%
