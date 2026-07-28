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

from calibration_utils.xy8 import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)
from calibration_utils.measurement_utils.measurement_streams import (
    declare_streams,
    save_measurement,
    buffer_streams,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        XY8 DYNAMICAL DECOUPLING T2 MEASUREMENT
This node measures qubit coherence under XY8 dynamical decoupling. Eight refocusing pi
pulses with alternating X/Y axes and CPMG timing filter higher-frequency noise.

Pulse sequence (CPMG timing; X = x180, Y = y180):

    pi/2 - tau - X - 2*tau - Y - 2*tau - X - 2*tau - Y - 2*tau - Y - 2*tau - X - 2*tau - Y - 2*tau - X - tau - pi/2

The swept parameter tau is the CPMG half-spacing: the two bookend delays are tau, and
the seven intervals between refocusing pulses are 2*tau. Total free evolution per point
is 16*tau. The signal decays as exp(-16*tau/T2_xy8).

Prerequisites:
    - Hahn echo node (12) and its prerequisites.
    - Calibrated pi, pi/2, and y180 pulses from Rabi measurements.

Datasets:
    - ``ds_raw``: raw parity streams from the OPX (``p_{qubit}`` or joint-outcome streams).
      Never modified after acquisition.
    - ``ds_fit``: processed conditional expectations, fitted decay curves, and per-qubit
      summary scalars on the ``qubit`` coordinate. Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict (``FitParameters`` serialized with
      ``asdict``). Used by logging and ``node.outcomes``.

Results (``node.results["fit_results"][<qubit>]``):
    - ``success``: whether the exponential fit converged to a physical result.
    - ``T2_xy8`` [ns]: XY8 coherence time.
    - ``amplitude``: dynamical-decoupling contrast.
    - ``offset``: baseline level.
    - ``decay_rate`` [1/ns]: effective rate 16 / T2_xy8.

Figures (``node.results["figures"]``):
    - ``"decay"``: horizontal subplots of conditional readout vs CPMG half-spacing tau
      (16 tau total idle) with exponential fit overlay for each qubit.

State update:
    - None (diagnostic measurement; inspect ``fit_results`` and ``ds_fit``).
"""


node = QualibrationNode[Parameters, Quam](name="13_xy8", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local parameter overrides for debugging (ignored in the GUI / graph)."""
    # node.parameters.qubits = ["q1"]
    # node.parameters.num_shots = 10
    # node.parameters.use_simulated_data = True
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for the XY8 sequence.

    Sweeps half inter-pulse spacing τ. Pulse sequence per sweep point:

        empty → measure(p1) → initialise → x90 → [τ-X-2τ-Y-2τ-X-2τ-Y-2τ-Y-2τ-X-2τ-Y-2τ-X-τ] → x90 → measure(p2)
    """
    node.namespace["qubits"] = qubits = get_qubits(node)

    n_avg = node.parameters.num_shots
    tau_values = np.arange(
        node.parameters.tau_min,
        node.parameters.tau_max,
        node.parameters.tau_step,
    )
    tau_clock_cycles = tau_values // 4  # QUA wait() uses 4 ns clock cycles

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(
            tau_values,
            attrs={
                "long_name": "XY8 CPMG half-spacing τ (bookend τ, inter-pulse 2τ)",
                "units": "ns",
            },
        ),
    }

    with program() as node.namespace["qua_program"]:
        t = declare(int)  # half inter-pulse spacing tau in clock cycles
        n = declare(int)  # shot counter

        p2, p1, parity_streams = declare_streams(node, qubits)
        n_st = declare_output_stream()

        for qubit in qubits:
            # ── OUTER LOOP: average n_avg shots per tau point ────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── INNER LOOP: sweep half inter-pulse spacing tau ───────────
                with for_(*from_array(t, tau_clock_cycles)):
                    reset_frame(qubit.xy.name)

                    if node.parameters.parity_measurement:
                        qubit.empty()
                        a1 = qubit.measure()

                    qubit.initialize(
                        target_state=node.parameters.target_state,
                        max_loops=node.parameters.max_loops,
                        conditional_drive=True,
                    )

                    align()

                    with strict_timing_():
                        # Opening pi/2
                        qubit.x90()

                        # First half-interval (τ)
                        wait(t, qubit.xy.name)

                        # Pulse 1: X
                        qubit.x180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 2: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 3: X
                        qubit.x180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 4: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 5: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 6: X
                        qubit.x180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 7: Y
                        qubit.y180()
                        # Full interval (2τ)
                        wait(2 * t, qubit.xy.name)

                        # Pulse 8: X
                        qubit.x180()

                        # Last half-interval (τ)
                        wait(t, qubit.xy.name)

                        # Closing pi/2
                        qubit.x90()

                    align()

                    a2 = qubit.measure()

                    qubit.voltage_sequence.ramp_to_zero()

                    align()

                    assign(p2, Cast.to_int(a2))

                    if node.parameters.parity_measurement:
                        assign(p1, Cast.to_int(a1))

                    save_measurement(node, qubit.name, p1, p2, parity_streams)

        with stream_processing():
            n_st.save("n")
            n_tau = len(tau_values)
            for qubit in qubits:
                # Buffer tau sweep; average over shots -> 1D trace per qubit
                buffer_streams(node, qubit.name, parity_streams, n_tau)


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate synthetic XY8 data so the analysis pipeline runs without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit an exponential decay to the XY8 data for each qubit.

    Processes raw streams, fits each qubit, and stores ``ds_fit`` (with fitted
    curves and summary scalars) and ``fit_results``.
    """
    ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {qname: ("successful" if r["success"] else "failed") for qname, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot decay traces and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.namespace["qubits"],
        analysis_signal=node.parameters.analysis_signal,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No state update — XY8 is a diagnostic measurement.

    Results are available in node.results["fit_results"] for inspection.
    """
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node results and parameters."""
    node.save()
