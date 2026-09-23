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
    FitParameters,
    Parameters,
    critical_photon_number,
    fit_raw_data,
    gamma_per_photon,
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
    steady_state_pad_ns,
    tone_intermediate_frequency,
)

# %% {Node initialisation}
description = """
        AC STARK PHOTON NUMBER CALIBRATION
Converts a readout drive power into the mean number of photons inside the resonator.

A Ramsey sequence is run with a Stark tone held on throughout, and the phase of the second pi/2 pulse
is swept to produce a fringe. The fringe phase relative to the zero-amplitude reference is the Stark
phase, and the fringe contrast relative to the same reference is the measurement-induced dephasing.

The free evolution time is swept, and both quantities are fitted against it with a
FREE INTERCEPT. The slope of the phase is the Stark shift Delta_omega, which divided by 2 chi gives
the photon number n_bar; the slope of the log contrast ratio is the dephasing Gamma_d. The intercept
is free on purpose: everything the tone does to the qubit outside the free evolution, meaning the
phase picked up while the two pi/2 pulses play in a Stark-shifted qubit and the tilt of their
rotation axis, is the same at every free evolution time and so lands there instead of in the slope.

A phase that is really Delta_omega times tau grows in proportion to tau. The R-squared of that growth
is reported and the qubit fails without it, which is what makes the number falsifiable.

The tone is held for a steady-state pad before the first pi/2, so the resonator is full
before any phase is accumulated. The resonator is given its own pulse-processor core for the run,
because an element sharing a core with its qubit's drive cannot hold a tone across the Ramsey at all:
the tone is played first and the Ramsey follows in the ring-down, producing a phase that does not
grow with the free evolution time. Both changes are reverted when the node finishes.

Two independent photon numbers are reported. Gamma_d divided by the dephasing rate per photon, which
comes from chi, kappa and the tone detuning with no expansion in chi/kappa, gives a photon number
that shares no fitted quantity with the one from the phase. They should agree.

Amplitude scale 1.0 is the qubit's operating readout amplitude, and the zero-amplitude reference sits
innermost in the sweep so that slow qubit frequency drift is subtracted rather than absorbed into the
result. The sweep does not reach scale 1.0, because the fringe collapses well before it, so n_bar
there comes from extending the fitted line.

Prerequisites:
    - Having measured the resonator linewidth and the dispersive shift chi
      (node 23a_resonator_linewidth.py). This node stops with an error when kappa is missing, rather
      than assuming a linewidth, because the sequence itself needs it.
    - Having calibrated the pi/2 pulse (node 04b_power_rabi.py).
    - The resonator needs a `const` operation; the node sets its length and amplitude and reverts them.

State update:
    - The photon calibration constant: qubit.resonator.extras["photons_per_mw"]
    - The same constant as a reference level: qubit.resonator.extras["single_photon_power_dbm"]
    - The photon number at the operating readout amplitude:
      qubit.resonator.extras["n_bar_at_operating_readout_amplitude"]
    - The dispersive-breakdown scale: qubit.resonator.extras["critical_photon_number"]
    - The power at which it is reached: qubit.resonator.extras["critical_power_dbm"]

The first three are the measured calibration. Gamma_d, the second photon number and the
full n_bar table stay in the node results. chi is read and never written, so the Gamma_d check stays
independent.

The last two are the readout power ceiling. The critical photon number is calculated
from the stored chi, qubit frequency and anharmonicity, so it needs no measurement and is written
even for a qubit whose fringe fit failed. The anharmonicity is taken as negative whatever sign state
holds it in. Treat the pair as an order of magnitude: they come from the approximation
whose failure they predict, and measurement-induced state transitions generally arrive lower.
"""

node = QualibrationNode[Parameters, Quam](
    name="23b_ac_stark_photon_calibration",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]) -> None:
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Program_helpers}
MIN_WAIT_CYCLES = 4
"""The shortest wait the OPX accepts, in clock cycles."""


def _element_core(channel) -> str:
    """The pulse-processor core a channel is pinned to, whichever field carries it.

    QUAM accepts the name under `core` and, deprecated since qm-qua 1.2.2, under `thread`. They are
    mutually exclusive, so the field already in use is the one to write back to.
    """
    core = getattr(channel, "core", None)
    return core if core is not None else getattr(channel, "thread", None)


def _assign_separate_core(tracked_resonator, qubit, node: QualibrationNode[Parameters, Quam]) -> None:
    """Move the resonator to its own core so the tone can overlap the Ramsey.

    Elements sharing a core cannot play overlapping pulses: the resonator tone is issued first and
    the pi/2 that follows is pushed until it finishes, so the qubit only ever sees the ring-down.
    The change is tracked and reverted with everything else.
    """
    xy_core = _element_core(qubit.xy)
    resonator_core = _element_core(qubit.resonator)
    if xy_core is None or xy_core != resonator_core:
        return
    if not node.parameters.separate_resonator_core:
        node.log(
            f"WARNING {qubit.name}: its drive and resonator share core '{xy_core}', so the tone cannot "
            f"overlap the Ramsey sequence and the measured phase will not grow with the free evolution "
            f"time. separate_resonator_core is off, so this is left as it is."
        )
        return
    new_core = f"{qubit.name}_tone"
    if getattr(qubit.resonator, "core", None) is not None:
        tracked_resonator.core = new_core
    else:
        tracked_resonator.thread = new_core
    node.log(f"{qubit.name}: resonator moved from core '{xy_core}' to '{new_core}' for the duration of the run.")


def _free_evolution_times_ns(node: QualibrationNode[Parameters, Quam]) -> list:
    """The free evolution times, rounded to the clock cycle and long enough for the OPX to wait."""
    minimum_ns = MIN_WAIT_CYCLES * CLOCK_CYCLE_NS
    times = sorted({max(round_up_to_clock_cycle(t), minimum_ns) for t in node.parameters.free_evolution_times_in_ns})
    if len(times) < 3:
        raise ValueError(
            f"at least 3 distinct free evolution times are needed to fit a slope with a free intercept, "
            f"got {times}. The Stark shift is that slope."
        )
    return times


def _warn_on_sequence_feasibility(node, qubit, amps: np.ndarray, taus_ns: list, chi_hz: float, kappa_hz: float) -> None:
    """Warn before the run when the fringe will be dead, or the phase will alias, at the top of the sweep.

    Both predictions need a photon calibration from an earlier run. Without one there is nothing to
    predict from, because the line attenuation is exactly what this node measures, so the node says
    so and relies on the checks made after the fit instead.
    """
    extras = getattr(qubit.resonator, "extras", None) or {}
    photons_per_mw = extras.get("photons_per_mw")
    if not photons_per_mw or not chi_hz:
        node.log(f"{qubit.name}: no earlier photon calibration, so the sweep is only checked after the fit.")
        return
    chi_rad, kappa_rad = 2 * np.pi * chi_hz, 2 * np.pi * kappa_hz
    max_power_mw = 10 ** ((qubit.resonator.get_output_power("readout") + 20 * np.log10(max(amps))) / 10)
    n_bar_max = float(photons_per_mw) * max_power_mw
    longest = max(taus_ns) * 1e-9

    surviving = np.exp(-gamma_per_photon(chi_rad, kappa_rad) * n_bar_max * longest)
    if surviving < node.parameters.min_contrast_fraction:
        node.log(
            f"WARNING {qubit.name}: at amplitude scale {max(amps):.2f} the earlier calibration predicts "
            f"n̄ = {n_bar_max:.2f}, whose dephasing leaves {100 * surviving:.1f}% of the fringe after "
            f"{max(taus_ns)} ns. Those points will be dropped. Shorten the longest free evolution time or "
            f"lower max_amplitude_scale."
        )
    steps = np.diff([0.0] + list(taus_ns)) * 1e-9
    largest_step = abs(2 * chi_rad * n_bar_max) * float(np.max(steps)) / np.pi
    if largest_step >= node.parameters.phase_step_warning_fraction_of_pi:
        node.log(
            f"WARNING {qubit.name}: the fringe phase is expected to grow by {largest_step:.2f} π between "
            f"adjacent free evolution times. The phase is unwrapped along that axis, which only holds below "
            f"π, so add closer-spaced times or lower max_amplitude_scale."
        )


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]) -> None:
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    free_evolution_ns = _free_evolution_times_ns(node)
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
    for qubit in qubits:
        if "const" not in qubit.resonator.operations:
            raise KeyError(
                f"{qubit.name}: the resonator has no 'const' operation to play the Stark tone with. Add a "
                f"SquarePulse named 'const' to the resonator in your populate script."
            )
        kappa_tot_hz = kappa_tot_hz_from_extras(qubit)
        lifetime_ns = photon_lifetime_ns(qubit)
        pad_ns = steady_state_pad_ns(qubit, node.parameters.steady_state_pad_in_lifetimes)
        depletion_ns = max(
            round_up_to_clock_cycle(node.parameters.depletion_in_lifetimes * lifetime_ns),
            round_up_to_clock_cycle(qubit.resonator.depletion_time),
        )
        x90_length_ns = int(qubit.xy.operations["x90"].length)
        sequence_info[qubit.name] = {
            "kappa_tot_hz": kappa_tot_hz,
            "lifetime_ns": lifetime_ns,
            "pad_ns": pad_ns,
            "steady_state_fraction": steady_state_fraction(node.parameters.steady_state_pad_in_lifetimes),
            "depletion_ns": depletion_ns,
            "depletion_over_lifetime": depletion_ns / lifetime_ns,
            "x90_length_ns": x90_length_ns,
            "tone_length_ns": round_up_to_clock_cycle(pad_ns + 2 * x90_length_ns + max(free_evolution_ns)),
        }
        info = sequence_info[qubit.name]
        node.log(
            f"{qubit.name}: kappa_tot {1e-6 * kappa_tot_hz:.3f} MHz, photon lifetime {lifetime_ns:.0f} ns. "
            f"Pad {pad_ns} ns reaches {100 * info['steady_state_fraction']:.0f}% of the steady-state photon "
            f"number; depletion {depletion_ns} ns is {info['depletion_over_lifetime']:.1f} lifetimes."
        )
        _warn_on_sequence_feasibility(node, qubit, amps, free_evolution_ns, getattr(qubit, "chi", None), kappa_tot_hz)

    # Lengthen the Stark tone to cover the pad, both pi/2 pulses and the longest free evolution, set
    # its amplitude to the qubit's operating readout amplitude so that scale 1.0 means that
    # amplitude, and move the resonator to its own core so the tone can overlap the Ramsey at all.
    # Every change is reverted when the node finishes.
    node.namespace["tracked_resonators"] = []
    for qubit in qubits:
        with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            _assign_separate_core(resonator, qubit, node)
            resonator.operations["const"].length = sequence_info[qubit.name]["tone_length_ns"]
            resonator.operations["const"].amplitude = qubit.resonator.operations["readout"].amplitude
            node.namespace["tracked_resonators"].append(resonator)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(free_evolution_ns, attrs={"long_name": "free evolution time", "units": "ns"}),
        "phase": xr.DataArray(phases, attrs={"long_name": "phase of the second π/2", "units": "turns"}),
        "amp_scale": xr.DataArray(amps, attrs={"long_name": "Stark tone amplitude scale", "units": ""}),
    }

    tone_intermediate_frequencies = {
        q.name: tone_intermediate_frequency(q, node.parameters.tone_frequency_in_ghz) for q in qubits
    }

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

    params = node.parameters
    for name, result in node.results["fit_results"].items():
        # The weak-dispersive warning does not need a successful fit; it is a property of the chip.
        ratio = result["chi_over_kappa"]
        if np.isfinite(ratio) and ratio > params.weak_dispersive_warning_ratio:
            node.log(
                f"WARNING {name}: |χ|/κ = {ratio:.2f} is above {params.weak_dispersive_warning_ratio}, so the "
                f"two dressed resonances are separated by less than a linewidth and the weak-dispersive "
                f"formulas behind this node are only approximate. The exact expression is used for the "
                f"Γ_d check, but n̄ = Δω/2χ still assumes the dispersive picture holds."
            )
        if not result["success"]:
            continue

        within = result["num_times_within_linear_regime"]
        limit_ns = result["number_splitting_phase_limit_ns"]
        if np.isfinite(limit_ns) and within < 4:
            node.log(
                f"WARNING {name}: only {within:.0f} of the swept free evolution times sit below the "
                f"photon-number-splitting limit of {limit_ns:.0f} ns, where 2χτ reaches "
                f"{params.max_number_splitting_phase_rad} rad. Past it the apparent phase follows "
                f"n̄ sin(2χτ) rather than growing, so the slope has few points to stand on. Add times "
                f"below {limit_ns:.0f} ns."
            )

        leverage = result["max_tau_over_photon_lifetime"]
        if np.isfinite(leverage) and leverage < params.min_tau_over_photon_lifetime:
            node.log(
                f"WARNING {name}: the longest free evolution time that kept a usable fringe is only "
                f"{leverage:.1f} photon lifetimes. Below about one lifetime a resonator transient grows "
                f"almost linearly with time as well, so the phase linearity of "
                f"{result['phase_linearity_r_squared']:.3f} does not prove the phase is a Stark phase. Add "
                f"longer free evolution times; they survive at the bottom of the amplitude sweep. On this "
                f"qubit the number-splitting limit of {limit_ns:.0f} ns sits below two photon lifetimes "
                f"({2e9 / (2 * np.pi * result['kappa_tot_hz']):.0f} ns), so no single time satisfies both "
                f"and the overlap has to be confirmed from the simulated waveform report instead."
            )

        step = result["max_phase_step_rad"] / np.pi
        if np.isfinite(step) and step >= params.phase_step_warning_fraction_of_pi:
            node.log(
                f"WARNING {name}: the fringe phase grew by {step:.2f} π between adjacent free evolution "
                f"times. The phase is unwrapped along that axis, which only holds below π, so the Stark "
                f"shift at the top of the sweep may be wrong by a multiple of 2π over the step."
            )

        measured = result["gamma_d_over_delta_omega"]
        if np.isfinite(measured) and measured > 1:
            node.log(
                f"WARNING {name}: Γ_d/Δω came out {measured:.2f}. For a tone anywhere in the resonator that "
                f"ratio cannot exceed 1, whatever χ and κ are, so the excess is not measurement backaction. "
                f"Drive-induced qubit transitions are the usual cause."
            )
        disagreement = result["gamma_ratio_disagreement"]
        if np.isfinite(disagreement) and disagreement > params.gamma_ratio_warning_fraction:
            node.log(
                f"WARNING {name}: Γ_d/Δω of {measured:.3f} disagrees with the predicted "
                f"{result['expected_gamma_ratio']:.3f} by {100 * disagreement:.0f}%. Either the tone is not "
                f"where it is assumed to be, or something other than the dispersive interaction is "
                f"decohering the qubit."
            )

        n_ratio = result["n_bar_gamma_over_phase"]
        if np.isfinite(n_ratio) and abs(n_ratio - 1) > params.n_bar_gamma_warning_fraction:
            node.log(
                f"WARNING {name}: the dephasing implies {result['n_bar_from_gamma_at_operating']:.2f} photons "
                f"at the operating amplitude where the phase implies "
                f"{result['n_bar_at_operating_amplitude']:.2f}, a factor of {n_ratio:.2f}. The two come from "
                f"the same fringes by different routes, so they should agree."
            )
        if result["detuning_source"] == "unavailable":
            node.log(
                f"{name}: no measured resonances and no bare resonator frequency in state, so the tone "
                f"detuning is unknown and the predicted Γ_d ratio assumes the tone sits at the midpoint of "
                f"the dressed pair. Node 23a can store f_r_ground_hz and f_r_excited_hz to fix this."
            )

    # The critical photon number comes from stored state, not from this node's data, so these checks
    # run for every qubit including the ones whose fringe fit failed.
    for qubit in node.namespace["qubits"]:
        result = node.results["fit_results"][qubit.name]
        g_hz = result["coupling_g_hz"]
        if not np.isfinite(g_hz):
            continue
        if result["anharmonicity_sign_flipped"]:
            node.log(
                f"{qubit.name}: the stored anharmonicity is positive and was taken as negative. "
                f"A transmon is negatively anharmonic; the populate script should store it signed, because "
                f"anything else reading it, including inferred_f_12, is wrong until it does."
            )
        if result["chi_sign_flipped"]:
            node.log(
                f"WARNING {qubit.name}: with the anharmonicity sign already settled, the stored χ still "
                f"implies a negative g². Its sign was flipped to get a positive critical photon number, but "
                f"check which convention χ was stored in."
            )
        if not (params.min_plausible_coupling_g_in_hz <= g_hz <= params.max_plausible_coupling_g_in_hz):
            node.log(
                f"WARNING {qubit.name}: the stored χ implies g = {1e-6 * g_hz:.1f} MHz, outside the "
                f"{1e-6 * params.min_plausible_coupling_g_in_hz:.0f} to "
                f"{1e-6 * params.max_plausible_coupling_g_in_hz:.0f} MHz a planar chip is built for. "
                f"The stored χ, anharmonicity or qubit frequency is probably wrong, and the critical "
                f"photon number derived from them with it."
            )
        headroom = result["n_bar_over_critical"]
        if np.isfinite(headroom) and headroom > params.n_bar_over_critical_warning_fraction:
            node.log(
                f"WARNING {qubit.name}: the readout runs at {100 * headroom:.0f}% of the critical photon "
                f"number of {result['critical_photon_number']:.1f}, so the dispersive approximation behind "
                f"every formula in this node is close to breaking down at the operating power."
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
