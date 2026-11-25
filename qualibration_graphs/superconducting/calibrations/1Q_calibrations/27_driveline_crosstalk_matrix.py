# %% {Imports}
from dataclasses import asdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from quam.components.pulses import WaveformPulse

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.driveline_crosstalk_matrix import (
    FRAMES,
    OPERATION,
    PHASE_MODE,
    Parameters,
    add_pair_info,
    fit_raw_data,
    log_fitted_results,
    matrix_names,
    plan_pairs,
    plot_pair_fits,
    plot_crosstalk_matrix,
    state_matrix,
    unmeasurable_pairs,
    upgrade_names,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        DRIVE-LINE CROSSTALK MATRIX AT THE SOURCE FREQUENCY
Measures, for every directed pair "source-victim", how much of the source qubit's gate leaks onto the victim through
the lines: the complex coefficient c, such that the victim feels c times the source pulse, at the source frequency.
|c| is given in % of amplitude; the phase of c is relative to the victim line. To cancel the leak, the victim line
plays -c times the source pulse, at the source frequency, whenever the source qubit plays a gate.

Sequence, for one tone setting z (amplitude |z|, phase arg z):
    victim: x90 - [copy of the source's x180, times z, at the source frequency] x N - frame - x90 - readout
    source:       [x180 at its own frequency]                                  x N
The victim feels the leak plus the tone, a field proportional to (c + z):
    - pairs detuned by more than min_stark_detuning_mhz: the field shifts the victim's frequency (AC Stark shift);
      the Ramsey phase it picks up is a |c + z|^2 + const, read from the populations at four final frames
      (0, 1/4, 1/2, 3/4 turn).
    - closer pairs: the field rotates the victim; without the x90 pulses, its excited population is
      a |c + z|^2 + const (small angles). The four frames are then repeats.
How c is obtained. The tone is played at 17 settings: no tone, |z| = r_max / 2 at 8 phases and |z| = r_max at 8
other phases (circles around z = 0). The response is a bowl over the complex plane of z, a |c + z|^2 + e0, whose
bottom is where the tone cancels the leak, z = -c. Without crosstalk the bowl is centred on z = 0 and the 8 points of
a circle give the same response; with crosstalk, the points nearer to -c respond less than those on the far side.
A linear least-squares fit of the 17 points to a|z|^2 + b Re z + d Im z + e gives the bottom, as for a parabola:
Re(-c) = -b / 2a and Im(-c) = -d / 2a, i.e. c = (b + i d) / (2a). |c| is the distance of the bottom from z = 0 and
arg c the direction from the bottom to z = 0. The error of c follows from the shot noise of each point. Constant
offsets (ZZ, detuning, readout) only change e. N is even, so that the source qubit ends in |0>; r_max and N are
adapted per pair so that the largest signal is about 1 rad (the Ramsey phase does not wrap over the grid).

The pairs given by pairs ("source-victim"; if None, all pairs among the selected qubits) are measured one after the
other in a single QUA program. Pairs with |f_source - victim LO| > 400 MHz, that share an output port, or whose
f_source is within min_stark_detuning_mhz of the victim's 1-2 transition are skipped. The tone is a temporary
operation on the victim's xy channel, played after switching it to the source frequency (and back,
keep_phase=False, for the Ramsey pulses); it is removed at the end.

Figures:
    - pair_fits: per pair, the response over the measured region of the complex plane of z (Ramsey:
      S = (1 + cos(Ramsey phase - phase at -c)) / 2, as in Nuerbolati et al., arXiv:2204.02946, Fig. 2(b);
      population: P_e). Circles: measured points; background and white lines: the fit, whose lines of equal response
      are circles around -c (red x, +/- 1 sigma). Title: |c|, arg c and the angle of
      the correction -c (if |c| > 2 sigma).
    - crosstalk_matrix (and crosstalk_matrix_state): |c| and the phase of c as stored in the state, with this run's
      pairs outlined (green: updated, red: failed).

Prerequisites:
    - Having calibrated the readout (nodes 02a, 02b and/or 02c) and the IQ blobs (node 07).
    - Having calibrated the qubit frequency and the x180 / x90 pulses (nodes 03a, 04b) and T2 (node 06a).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - For every pair whose fit passed, on the source qubit:
      qubit.extras["drive_crosstalk_complex"][victim] = {"c_re", "c_im", "c_err", "frequency_hz", "victim_lo_hz",
      "updated"}, with frequency_hz the source frequency. To compensate, the victim line plays -c times the source
      pulse at the source frequency, with the same victim LO. The entry is stale when the victim LO changes or the
      source frequency moves.
    - Entries of pairs not measured, or whose fit failed, are kept. The crosstalk matrix figures show the matrix
      stored in the state over the qubits of this run and over every qubit of the state, with this run's pairs
      outlined.
"""

STATE_KEY = "drive_crosstalk_complex"

node = QualibrationNode[Parameters, Quam](
    name="27_driveline_crosstalk_matrix", description=description, parameters=Parameters(), machine=Quam.load()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Plan the pairs, add the temporary tone operations, create the sweep axes and generate the QUA program."""
    p = node.parameters
    node.namespace["qubits"] = qubits = get_qubits(node)
    if len(qubits) < 2:
        raise ValueError("Drive-line crosstalk requires at least two qubits.")
    node.namespace["pairs"] = pairs = plan_pairs(qubits, p, STATE_KEY, log_callable=node.log)
    if not pairs:
        raise ValueError("No pair among the selected qubits can be measured (see the log).")

    # Temporary tone operations: a copy of the source pulse on the victim line (removed in revert_config_changes)
    node.namespace["tone_ops"] = sorted({(pair["victim"].name, pair["tone_op"]) for pair in pairs})
    for pair in pairs:
        tone = pair["tone"]
        pulse = WaveformPulse(waveform_I=list(tone.real), waveform_Q=list(tone.imag))
        pair["victim"].xy.operations[pair["tone_op"]] = pulse

    num_points = len(pairs[0]["z"])
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray([pair["key"] for pair in pairs]),
        "point": xr.DataArray(np.arange(num_points), attrs={"long_name": "grid point of the tone z"}),
        "frame": xr.DataArray(list(FRAMES), attrs={"long_name": "final frame", "units": "turns"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_total = declare(int)
        n_st = declare_stream()
        t = declare(int)
        source_amp = declare(fixed)
        amp = declare(fixed)
        phase = declare(fixed)
        frame = declare(fixed)
        state = declare(int)
        state_st = [declare_stream() for _ in pairs]

        for i, pair in enumerate(pairs):
            victim, source = pair["victim"], pair["source"]
            phase_mode = pair["mode"] == PHASE_MODE
            source_on = [float(s) for s in pair["source_on"]]
            amps = [float(abs(z)) for z in pair["z"]]
            phases = [float((np.angle(z) / (2 * np.pi)) % 1) for z in pair["z"]]
            tone_if = int(round(pair["tone_if"]))
            own_if = int(round(victim.xy.intermediate_frequency))

            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in (victim, source):
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < p.num_shots, n + 1):
                # Progress over all the pairs
                assign(n_total, n + i * p.num_shots)
                save(n_total, n_st)
                with for_each_((source_amp, amp, phase), (source_on, amps, phases)):
                    with for_each_(frame, list(FRAMES)):
                        # Reset the qubits to the ground state
                        victim.reset(p.reset_type, p.simulate, log_callable=node.log)
                        source.reset(p.reset_type, p.simulate, log_callable=node.log)
                        align()
                        if phase_mode:
                            victim.xy.play("x90")
                        align(source.xy.name, victim.xy.name)
                        # Tone on the victim line at f_source, with the phase arg z
                        victim.xy.update_frequency(tone_if)
                        victim.xy.frame_rotation_2pi(phase)
                        with for_(t, 0, t < pair["train_pulses"], t + 1):
                            source.xy.play(OPERATION, amplitude_scale=source_amp)
                            victim.xy.play(pair["tone_op"], amplitude_scale=amp)
                        # Back to the victim frequency: keep_phase=False restores its phase reference
                        reset_frame(victim.xy.name)
                        victim.xy.update_frequency(own_if)
                        align(source.xy.name, victim.xy.name)
                        if phase_mode:
                            victim.xy.frame_rotation_2pi(frame)
                            victim.xy.play("x90")
                            reset_frame(victim.xy.name)
                        align()
                        victim.readout_state(state)
                        save(state, state_st[i])
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(len(pairs)):
                state_st[i].buffer(len(FRAMES)).buffer(num_points).average().save(f"pe{i + 1}")


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
    total_shots = node.parameters.num_shots * len(node.namespace["pairs"])
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), total_shots, start_time=data_fetcher.t_start)
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset, with the plan of every pair (grid, mode, frequencies, train length)
    node.results["ds_raw"] = add_pair_info(dataset, node.namespace["pairs"], node.parameters)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit the bowl of every measured pair and store c in ds_fit and fit_results."""
    node.results["ds_fit"], fits = fit_raw_data(node.results["ds_raw"], node)
    node.namespace["fits"] = fits
    node.results["fit_results"] = {k: asdict(v) for k, v in fits.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    # A source qubit fails if any of its measured pairs failed. A failed curvature means the measurement itself
    # failed (the curvature comes from the tone, not from the crosstalk), so a weak crosstalk does not fail.
    node.outcomes = {
        qubit.name: (
            "failed" if any(not f.success for f in fits.values() if f.source_qubit == qubit.name) else "successful"
        )
        for qubit in node.namespace["qubits"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the fit of every pair (amplitude and angle of c) and the crosstalk matrix as the state holds it, with
    this run's pairs outlined: over the qubits of this run, and over every qubit of the state when that is larger."""
    qubits, fits = node.machine.qubits, node.namespace["fits"]
    measured = {name for fit in fits.values() for name in (fit.source_qubit, fit.victim_qubit)}
    figures = {"pair_fits": plot_pair_fits(node.results["ds_fit"])}
    run_names = [name for name in qubits.keys() if name in measured]
    state_names = matrix_names(qubits, run_names, STATE_KEY)
    for label, names in (("crosstalk_matrix", run_names), ("crosstalk_matrix_state", state_names)):
        if label == "crosstalk_matrix_state" and names == run_names:
            continue
        matrix = state_matrix(qubits, names, STATE_KEY, fits)
        unmeasurable = unmeasurable_pairs([qubits[name] for name in names], node.parameters)
        figures[label] = plot_crosstalk_matrix(matrix, names, unmeasurable)
    node.results["figures"] = figures
    plt.show()


# %% {Revert_config_changes}
@node.run_action()
def revert_config_changes(node: QualibrationNode[Parameters, Quam]):
    """Remove the temporary tone operations, whether or not the node simulated, analysed or failed to fit."""
    for victim, tone_op in node.namespace.get("tone_ops", []):
        operations = node.machine.qubits[victim].xy.operations
        if tone_op in operations:
            del operations[tone_op]


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write c of every pair whose fit passed on its source qubit; the other stored entries are kept."""
    ds_raw = upgrade_names(node.results["ds_raw"])
    updated = datetime.now().isoformat(timespec="seconds")
    with node.record_state_updates():
        for pair, fit in node.results["fit_results"].items():
            if not fit["success"]:
                continue
            source = node.machine.qubits[fit["source_qubit"]]
            if STATE_KEY not in source.extras:
                source.extras[STATE_KEY] = {}
            source.extras[STATE_KEY][fit["victim_qubit"]] = {
                "c_re": fit["c_re"],
                "c_im": fit["c_im"],
                "c_err": fit["c_err"],
                "frequency_hz": fit["frequency_hz"],
                "victim_lo_hz": float(ds_raw.victim_lo.sel(qubit_pair=pair)),
                "updated": updated,
            }


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
