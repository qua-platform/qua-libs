# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode

from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from quam_config import Quam

from calibration_utils.ac_stark_photon_calibration import (
    CLOCK_CYCLE_NS,
    DEPLETION_IN_LIFETIMES,
    N_BAR_OVER_CRITICAL_WARNING_FRACTION,
    STEADY_STATE_PAD_IN_LIFETIMES,
    FitParameters,
    Parameters,
    assign_separate_core,
    fit_raw_data,
    free_evolution_times_ns,
    kappa_tot_hz_from_extras,
    log_fitted_results,
    photon_lifetime_ns,
    plot_dephasing_vs_photon_number,
    plot_fringes,
    plot_phase_vs_time,
    plot_photon_calibration,
    process_raw_dataset,
    round_up_to_clock_cycle,
    steady_state_fraction,
    stark_tone_frequency_hz,
    steady_state_pad_ns,
    tone_intermediate_frequency,
    warn_on_sequence_feasibility,
)

# %% {Node initialisation}
description = """
        AC STARK PHOTON NUMBER CALIBRATION
Converts a readout drive power into the mean photon number in the resonator at the readout frequency.

A Stark tone is held on the resonator while a Ramsey runs on the qubit; the phase of the second pi/2 is
swept to form a fringe. Against the zero-amplitude reference, the fringe phase gives the Stark phase and
the contrast gives the measurement-induced dephasing. Both are fitted against the free evolution time tau
with a free intercept: the slopes are the Stark shift Delta_omega and the dephasing Gamma_d, and
n_bar = Delta_omega / (2 chi F), with F the exact Stark shift per photon in units of 2 chi.

The tone sits tone_detuning_in_linewidths * max(|chi|, kappa) above the midpoint of the two dressed
resonances (node 24), not at the readout frequency. There the |0> and |1> fields differ strongly when
chi is comparable to kappa, so the fringe dies before the phase becomes linear and F is near zero. Far
from both resonances the fields nearly coincide: dephasing per photon is small and F is close to 1.
n_bar is then carried over to the readout frequency with the Lorentzian response of the two dressed
resonances, assuming the drive line delivers the same power at both frequencies.

After the first pi/2 the field takes a few photon lifetimes to settle, so only tau >= 4 / kappa is
fitted. The tone is held for a steady-state pad before the first pi/2, and the resonator gets its own
pulse-processor core so the tone can overlap the Ramsey; both are reverted when the node finishes.
The amplitude sweep stays below the operating amplitude, so n_bar there comes from the fitted line.

Prerequisites:
    - Node 24_resonator_kappa_chi_circle_fit: kappa, chi and the dressed resonances. The node stops
      without them.
    - The pi/2 pulse (node 04b_power_rabi) and a `const` operation on the resonator.

State update (qubit.resonator.extras), at the readout frequency:
    - photons_per_mw and single_photon_power_dbm (the same constant as a power level)
    - n_bar_at_operating_readout_amplitude
    - critical_photon_number (from chi, qubit frequency and anharmonicity; written even if the fit fails)
      and critical_power_dbm. Treat both as an order of magnitude.
"""

node = QualibrationNode[Parameters, Quam](
    name="25_photon_number_stark_ramsey",
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
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    free_evolution_ns = free_evolution_times_ns(node)
    node.parameters.free_evolution_times_in_ns = free_evolution_ns
    if not (0 < node.parameters.max_amplitude_scale < 2):
        raise ValueError(
            "max_amplitude_scale must lie strictly between 0 and 2, the hardware limit on a real-time scale"
        )
    # Amplitude scale 1.0 is the operating readout amplitude, and 0 is the drift reference.
    amps = np.linspace(0.0, node.parameters.max_amplitude_scale, node.parameters.num_amplitude_points)
    # One full turn of the second pi/2 pulse, without repeating the endpoint.
    phases = np.linspace(0.0, 1.0, node.parameters.num_phase_points, endpoint=False)

    # Size the pad and the depletion per qubit, and fail before taking any fridge time if a resonator
    # has no measured linewidth or no const operation to play the tone with.
    node.namespace["sequence_info"] = sequence_info = {}
    tone_frequencies = {}
    for qubit in qubits:
        if "const" not in qubit.resonator.operations:
            raise KeyError(
                f"{qubit.name}: the resonator has no 'const' operation to play the Stark tone with. Add a "
                f"SquarePulse named 'const' to the resonator in your populate script."
            )
        kappa_tot_hz = kappa_tot_hz_from_extras(qubit)
        tone_frequencies[qubit.name] = stark_tone_frequency_hz(qubit, node.parameters.tone_detuning_in_linewidths)
        lifetime_ns = photon_lifetime_ns(qubit)
        pad_ns = steady_state_pad_ns(qubit, STEADY_STATE_PAD_IN_LIFETIMES)
        depletion_ns = max(
            round_up_to_clock_cycle(DEPLETION_IN_LIFETIMES * lifetime_ns),
            round_up_to_clock_cycle(qubit.resonator.depletion_time),
        )
        x90_length_ns = int(qubit.xy.operations["x90"].length)
        sequence_info[qubit.name] = {
            "kappa_tot_hz": kappa_tot_hz,
            "lifetime_ns": lifetime_ns,
            "pad_ns": pad_ns,
            "steady_state_fraction": steady_state_fraction(STEADY_STATE_PAD_IN_LIFETIMES),
            "depletion_ns": depletion_ns,
            "depletion_over_lifetime": depletion_ns / lifetime_ns,
            "x90_length_ns": x90_length_ns,
            "tone_length_ns": round_up_to_clock_cycle(pad_ns + 2 * x90_length_ns + max(free_evolution_ns)),
        }
        info = sequence_info[qubit.name]
        node.log(
            f"{qubit.name}: kappa_tot {1e-6 * kappa_tot_hz:.3f} MHz, photon lifetime {lifetime_ns:.0f} ns. "
            f"Pad {pad_ns} ns reaches {100 * info['steady_state_fraction']:.0f}% of the steady-state photon "
            f"number; depletion {depletion_ns} ns is {info['depletion_over_lifetime']:.1f} lifetimes. Tone at "
            f"{1e-9 * tone_frequencies[qubit.name]:.6f} GHz."
        )
        warn_on_sequence_feasibility(node, qubit, amps, free_evolution_ns, tone_frequencies[qubit.name])

    # Lengthen the Stark tone to cover the pad, both pi/2 pulses and the longest free evolution, set
    # its amplitude to the qubit's operating readout amplitude so that scale 1.0 means that
    # amplitude, and move the resonator to its own core so the tone can overlap the Ramsey at all.
    # Every change is reverted when the node finishes.
    node.namespace["tracked_resonators"] = []
    for qubit in qubits:
        with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            assign_separate_core(resonator, qubit, node)
            resonator.operations["const"].length = sequence_info[qubit.name]["tone_length_ns"]
            resonator.operations["const"].amplitude = qubit.resonator.operations["readout"].amplitude
            node.namespace["tracked_resonators"].append(resonator)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(free_evolution_ns, attrs={"long_name": "free evolution time", "units": "ns"}),
        "phase": xr.DataArray(phases, attrs={"long_name": "phase of the second π/2", "units": "turns"}),
        "amp_scale": xr.DataArray(amps, attrs={"long_name": "Stark tone amplitude scale", "units": ""}),
    }

    tone_intermediate_frequencies = {q.name: tone_intermediate_frequency(q, tone_frequencies[q.name]) for q in qubits}

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        a = declare(fixed)
        phase = declare(fixed)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # The free evolution axis is unrolled in Python, so each time has its own tone length.
                for tau_ns in free_evolution_ns:
                    with for_(*from_array(phase, phases)):
                        # The amplitude axis is innermost, so the zero-amplitude reference is taken
                        # next to every tone point and slow qubit frequency drift subtracts out.
                        with for_each_(a, amps):
                            for i, qubit in multiplexed_qubits.items():
                                reset_frame(qubit.xy.name)
                                qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                            # Put the resonator on the Stark tone frequency.
                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.update_frequency(tone_intermediate_frequencies[qubit.name])
                            align()
                            # Fill the resonator, then run the Ramsey inside the filled resonator.
                            for i, qubit in multiplexed_qubits.items():
                                info = node.namespace["sequence_info"][qubit.name]
                                tone_ns = info["pad_ns"] + 2 * info["x90_length_ns"] + tau_ns
                                qubit.resonator.play("const", amplitude_scale=a, duration=tone_ns // CLOCK_CYCLE_NS)
                                if info["pad_ns"]:
                                    qubit.xy.wait(info["pad_ns"] // CLOCK_CYCLE_NS)
                                qubit.xy.play("x90")
                                qubit.xy.wait(tau_ns // CLOCK_CYCLE_NS)
                                qubit.xy.frame_rotation_2pi(phase)
                                qubit.xy.play("x90")
                            align()
                            # Back to the readout frequency, then let the resonator empty before the
                            # readout. The qubit state is in populations by now, so this costs nothing.
                            for i, qubit in multiplexed_qubits.items():
                                rr = qubit.resonator
                                rr.update_frequency(rr.intermediate_frequency)
                                rr.wait(node.namespace["sequence_info"][qubit.name]["depletion_ns"] // CLOCK_CYCLE_NS)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                if node.parameters.use_state_discrimination:
                                    qubit.readout_state(state[i])
                                    save(state[i], state_st[i])
                                else:
                                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                            align()

        with stream_processing():
            n_st.save("n")
            num_taus = len(free_evolution_ns)
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    stream, label = state_st[i], f"state{i + 1}"
                    stream.buffer(len(amps)).buffer(len(phases)).buffer(num_taus).average().save(label)
                else:
                    I_st[i].buffer(len(amps)).buffer(len(phases)).buffer(num_taus).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).buffer(len(phases)).buffer(num_taus).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
    """Connect to the QOP and simulate the QUA program.

    Worth running before every change to the sequence: the waveform report shows whether the Stark
    tone and the two pi/2 pulses actually overlap, which is the one thing the analysis cannot check
    for itself.
    """
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
    node.namespace.setdefault("sequence_info", {})


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Fit the slopes, convert the Stark shift into a photon number and run the independent checks."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node)
    fit_results: dict[str, FitParameters]

    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)

    for name, result in node.results["fit_results"].items():
        headroom = result["n_bar_over_critical"]
        if np.isfinite(headroom) and headroom > N_BAR_OVER_CRITICAL_WARNING_FRACTION:
            node.log(
                f"WARNING {name}: the readout runs at {100 * headroom:.0f}% of the critical photon number "
                f"({result['critical_photon_number']:.0f}); the dispersive approximation is close to breaking down."
            )

    node.outcomes = {
        name: ("successful" if result["success"] else "failed") for name, result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]) -> None:
    """Plot the fringes, the phase against time, the photon calibration and the dephasing."""
    qubits = node.namespace["qubits"]
    ds_raw = node.results["ds_raw"]
    ds_fit = node.results["ds_fit"]
    node.results["figures"] = {
        "fringes": plot_fringes(ds_raw, qubits, ds_fit),
        "phase_vs_time": plot_phase_vs_time(ds_raw, qubits, ds_fit),
        "photon_calibration": plot_photon_calibration(ds_raw, qubits, ds_fit),
        "dephasing": plot_dephasing_vs_photon_number(ds_raw, qubits, ds_fit),
    }
    plt.show()


# %% {Revert_config_changes}
@node.run_action()
def revert_config_changes(node: QualibrationNode[Parameters, Quam]) -> None:
    """Undo the temporary changes made to the Stark tone and to the resonator's core assignment.

    This runs whether or not the node simulated, analysed or failed to fit, so the machine is left as
    it was found.
    """
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]) -> None:
    """Write the photon calibration into the resonator extras."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            result = node.results["fit_results"][qubit.name]

            # The critical photon number is derived from chi, the detuning and the anharmonicity, not
            # from this node's data, so it is written even when the fringe fit failed. It goes stale
            # when any of those three changes.
            n_crit = result["critical_photon_number"]
            if np.isfinite(n_crit):
                qubit.resonator.extras["critical_photon_number"] = float(n_crit)

            if node.outcomes[qubit.name] == "failed":
                continue
            # The two constants are one fit expressed two ways, so they are always written together.
            qubit.resonator.extras["photons_per_mw"] = float(result["photons_per_mw"])
            qubit.resonator.extras["single_photon_power_dbm"] = float(result["single_photon_power_dbm"])
            qubit.resonator.extras["n_bar_at_operating_readout_amplitude"] = float(
                result["n_bar_at_operating_amplitude"]
            )
            # Name the node that wrote them, so another node's calibration is distinguishable.
            qubit.resonator.extras["photons_per_mw_source"] = node.name

            # The power at which n̄ reaches the critical photon number. It combines the calculated
            # critical number with the measured calibration, so it is stale if either moves, and it
            # is written only alongside the calibration it was derived from.
            critical_power = result["critical_power_dbm"]
            if np.isfinite(critical_power):
                qubit.resonator.extras["critical_power_dbm"] = float(critical_power)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]) -> None:
    node.save()
