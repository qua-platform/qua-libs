"""Measurement-induced dephasing matrix for characterizing readout crosstalk."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.measurement_induced_dephasing_matrix import (
    Parameters,
    build_phases,
    build_xi_values,
    probe_length_in_ns,
    validate_readout_len,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_contrast_with_fit,
    plot_dephasing_matrix,
    plot_phase_oscillations,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        MEASUREMENT-INDUCED DEPHASING MATRIX
Implements the protocol of Fig. 6 of Phys. Rev. Applied 23, 054089 (arXiv:2412.14853).

A Hahn echo (x90 - tau - x180 - tau - phase - -x90 - measurement) is played on qubit Qi while a
readout pulse of relative amplitude xi is inserted into the FIRST half of the echo on resonator Rj.
The interval tau is fixed; the phase of the final pi/2 pulse is swept over one full turn to reveal a
coherent oscillation. The decay of that oscillation's contrast with the readout amplitude,

    c(xi) = c0 * exp(-Gamma * tau_p * xi**2),

yields the measurement-induced dephasing rate Gamma of the (Qi, Rj) pair, with tau_p the probe pulse
duration. Repeating over every pair builds the dephasing matrix: the diagonal holds the
self-dephasing rates (MHz scale) and the off-diagonal the readout crosstalk (Hz scale).

Because the two scales differ by orders of magnitude, the diagonal is swept logarithmically over a
much smaller amplitude range than the off-diagonal. The amplitude axis therefore differs per pair
and is stored as a two-dimensional 'xi' coordinate rather than as a dimension.

The first half of the echo is filled exactly by the probe pulse followed by the resonator depletion
time, so that no readout photon survives into the second half. Qubits are measured sequentially.

By default the probe is the calibrated readout pulse at its native length. Setting readout_len_in_ns
stretches it to that duration at unchanged amplitude, which is the cheapest way to resolve small
crosstalk: the uncertainty on Gamma falls as 1/tau_p and no extra shots are needed. The final
measurement is never stretched, so the discrimination threshold stays valid. The stretch is limited
by the echo, since the idle time has to grow with the probe and the contrast decays as
exp(-2*idle_time/T2echo); the T2 echo check below refuses idle times beyond half of T2echo.

Prerequisites:
    - Having calibrated the qubit x90 and x180 pulses (nodes 04b and 10b).
    - Having calibrated the readout: frequency, power and discrimination threshold (nodes 07, 08a and 08b).
    - Having measured T2 echo, used to check that the fixed idle time is not lossy (node 06b).

Next steps before going to the next node:
    - This node does not update the QUAM state; it reports the dephasing matrix. If the off-diagonal
      rates are comparable to the qubit decoherence rates, revisit the readout frequencies and
      amplitudes to reduce the spectral overlap between readout tones.
"""

node = QualibrationNode[Parameters, Quam](
    name="23_measurement_induced_dephasing_matrix",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node. The driven resonators are the resonators of those same
    # qubits, so the dephasing matrix is square.
    node.namespace["qubits"] = qubits = get_qubits(node)
    qubit_list = list(qubits)
    num_qubits = len(qubit_list)

    n_avg = node.parameters.num_shots
    # Relative readout amplitudes, one array per (measured qubit, driven resonator) pair
    xi_values = build_xi_values([q.name for q in qubit_list], node.parameters)
    num_xi = xi_values.shape[-1]
    # Phases of the final pi/2 pulse, in turns (frame_rotation_2pi takes units of 2*pi)
    phases = build_phases(node.parameters)

    # Duration of the probe pulse on each driven resonator. Either the native readout length or, when
    # readout_len_in_ns is set, the stretched duration requested by the user.
    validate_readout_len(node.parameters)
    probe_lengths_ns = {
        q.name: probe_length_in_ns(q.resonator.operations["readout"].length, node.parameters) for q in qubit_list
    }
    node.namespace["probe_lengths_in_ns"] = probe_lengths_ns
    # Left as None when no stretching is requested, so that the played pulse keeps the length it has
    # in the configuration instead of being re-specified at its own value.
    probe_duration_cycles = (
        None if node.parameters.readout_len_in_ns is None else node.parameters.readout_len_in_ns // 4
    )

    # The first half of the echo must hold the probe pulse and the subsequent resonator ring-down.
    # A single idle time is used for every qubit and every driven resonator so that the matrix
    # elements are directly comparable.
    idle_time_ns = node.parameters.idle_time_in_ns
    if idle_time_ns is None:
        idle_time_ns = max(probe_lengths_ns[q.name] + q.resonator.depletion_time for q in qubit_list)
        idle_time_ns = int(np.ceil(idle_time_ns / 4) * 4)
    node.namespace["idle_time_in_ns"] = idle_time_ns

    for q in qubit_list:
        probe_and_depletion = probe_lengths_ns[q.name] + q.resonator.depletion_time
        if idle_time_ns < probe_and_depletion:
            raise ValueError(
                f"The idle time ({idle_time_ns} ns) is shorter than the probe pulse plus depletion "
                f"time of {q.name} ({probe_and_depletion} ns): the readout photons would leak into "
                f"the second half of the echo."
            )
        t2_echo = getattr(q, "T2echo", None)
        if t2_echo and idle_time_ns > 0.5 * t2_echo * 1e9:
            raise ValueError(
                f"The idle time ({idle_time_ns} ns) exceeds half of the T2 echo of {q.name} "
                f"({t2_echo * 1e9:.0f} ns): the echo contrast would be dominated by intrinsic "
                f"decoherence rather than by measurement-induced dephasing."
            )

    idle_time_cycles = idle_time_ns // 4

    # Register the sweep axes to be added to the dataset when fetching data. 'xi' itself is not a
    # dimension because its values differ between the diagonal and the off-diagonal pairs; it is
    # attached as a two-dimensional coordinate during the analysis.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "driven_resonator": xr.DataArray(qubits.get_names()),
        "xi_idx": xr.DataArray(np.arange(num_xi), attrs={"long_name": "readout amplitude index"}),
        "phase": xr.DataArray(2 * np.pi * phases, attrs={"long_name": "final pulse phase", "units": "rad"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        shot = declare(int)
        xi = declare(fixed)
        phi = declare(fixed)

        # The qubits are measured sequentially: the outer loop is unrolled in Python so that each
        # measured qubit gets its own flux initialization.
        for i, qubit in enumerate(qubit_list):
            node.machine.initialize_qpu(target=qubit)
            align()

            x90_cycles = qubit.xy.operations["x90"].length // 4

            with for_(shot, 0, shot < n_avg, shot + 1):
                save(shot, n_st)

                # The driven resonator loop sits inside the averaging loop so that the stream
                # arrives in the order expected by the buffering below.
                for j, driven_qubit in enumerate(qubit_list):
                    depletion_cycles = driven_qubit.resonator.depletion_time // 4

                    with for_each_(xi, xi_values[i, j].tolist()):
                        with for_each_(phi, phases.tolist()):
                            # Qubit initialization
                            reset_frame(qubit.xy.name)
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            # Echo sequence on the measured qubit. The elements share a common
                            # starting point thanks to the align() above, so the probe pulse is
                            # placed on the resonator timeline by waiting out the x90 instead of
                            # realigning, which would stall the pipeline.
                            qubit.xy.play("x90")
                            qubit.xy.wait(idle_time_cycles)

                            # Probe pulse inserted in the first half of the echo, followed by the
                            # resonator ring-down.
                            driven_qubit.resonator.wait(x90_cycles)
                            driven_qubit.resonator.play("readout", amplitude_scale=xi, duration=probe_duration_cycles)
                            driven_qubit.resonator.wait(depletion_cycles)

                            qubit.xy.play("x180")
                            qubit.xy.wait(idle_time_cycles)
                            # Sweeping the phase of the final pi/2 pulse turns the echo into a
                            # coherent oscillation whose contrast measures the residual coherence.
                            qubit.xy.frame_rotation_2pi(phi)
                            qubit.xy.play("x90")
                            align()

                            # Qubit readout
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    (
                        state_st[i]
                        .buffer(len(phases))
                        .buffer(num_xi)
                        .buffer(num_qubits)
                        .average()
                        .save(f"state{i + 1}")
                    )
                else:
                    for stream, name in ((I_st[i], "I"), (Q_st[i], "Q")):
                        (
                            stream.buffer(len(phases))
                            .buffer(num_xi)
                            .buffer(num_qubits)
                            .average()
                            .save(f"{name}{i + 1}")
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
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in
    a xarray dataset called "ds_raw"."""
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
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary."""
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
    """Plot the contrast decays, the dephasing matrix and, optionally, the raw phase oscillations."""
    figures = {
        "contrast_vs_amplitude": plot_contrast_with_fit(node.results["ds_fit"], node.namespace["qubits"]),
        "dephasing_matrix": plot_dephasing_matrix(node.results["ds_fit"]),
    }
    if node.parameters.plot_phase_oscillations:
        figures["phase_oscillations"] = plot_phase_oscillations(node.results["ds_fit"], node)
    plt.show()
    # Store the generated figures
    node.results["figures"] = figures


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results. This node is diagnostic and does not update the QUAM state."""
    node.save()
