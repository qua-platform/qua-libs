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
from calibration_utils.gate_virtualization.cross_capacitance_1d_parameters import (
    CrossCapacitance1DParameters,
)
from calibration_utils.gate_virtualization.analysis import process_raw_dataset
from calibration_utils.gate_virtualization.cross_capacitance_1d_analysis import (
    extract_cross_capacitance_coefficient,
)
from calibration_utils.gate_virtualization.plotting import (
    plot_cross_capacitance_1d_diagnostic,
)
from calibration_utils.common_utils.experiment import get_sensors

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


description = """
        1D CROSS-CAPACITANCE MEASUREMENT (OPX)
Measures cross-capacitance matrix entries via paired 1D plunger sweeps
using a single QUA program that covers all (target, perturbing) pairs
in the mapping.

For each target plunger, one reference sweep (no perturbation) and one
shifted sweep per perturbing gate (+step_voltage) are performed
sequentially.  The shift in charge transition position divided by the
step voltage gives the cross-capacitance coefficient alpha_ij.

This is the fast 1D method described in Volk et al., npj Quantum
Information (2019) 5:29, Supplementary Fig. S1.

For QDAC-driven sweeps or perturbations, use the companion node
``04b_qdac_1d_cross_capacitance``.

Prerequisites:
    - Calibrated IQ mixer / Octave on the readout line.
    - Calibrated time of flight, offsets and gains.
    - Calibrated resonators coupled to SensorDot components.
    - Registered QuantumDot and SensorDot elements in QUAM.
    - Configured VirtualGateSet with initial compensation matrix.
"""


node = QualibrationNode[CrossCapacitance1DParameters, Quam](
    name="04_1d_cross_capacitance",
    description=description,
    parameters=CrossCapacitance1DParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    # node.parameters.cross_capacitance_mapping = {
    #     "virtual_dot_1": ["virtual_dot_2", "barrier_12"],
    # }
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
def create_qua_program(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Create a single QUA program covering all target/perturbing gate pairs.

    The program performs, for each target gate in the mapping:
    1. A reference sweep (no perturbation)
    2. A shifted sweep for each perturbing gate (+step_voltage)

    The sweep schedule is stored in ``node.namespace["sweep_schedule"]``
    for unpacking after execution.
    """
    node.namespace["sensors"] = sensors = get_sensors(node)
    p = node.parameters

    mapping = p.cross_capacitance_mapping
    if mapping is None:
        raise ValueError(
            "cross_capacitance_mapping must be provided. "
            "Automatic generation from the machine is not yet implemented."
        )

    sweep_schedule = []
    for target_gate, perturb_gates in mapping.items():
        sweep_schedule.append((target_gate, None))
        for pg in perturb_gates:
            sweep_schedule.append((target_gate, pg))

    first_target = next(iter(mapping))
    target_obj = node.machine.get_component(first_target)
    vgs_id = target_obj.voltage_sequence.gate_set.id

    v_sweep = np.linspace(
        -p.sweep_span / 2,
        p.sweep_span / 2,
        p.sweep_points,
    )

    num_sensors = len(sensors)
    num_sweeps = len(sweep_schedule)

    qua_prog = _build_opx_program(
        node,
        sensors,
        num_sensors,
        vgs_id,
        v_sweep,
        sweep_schedule,
        p,
    )

    sweep_axes = {
        "sensors": xr.DataArray(sensors.get_names()),
        "sweep_idx": xr.DataArray(
            np.arange(num_sweeps),
            attrs={"long_name": "sweep index"},
        ),
        "x_volts": xr.DataArray(
            v_sweep,
            attrs={"long_name": "plunger voltage", "units": "V"},
        ),
    }

    node.namespace["qua_program"] = qua_prog
    node.namespace["sweep_axes"] = sweep_axes
    node.namespace["sweep_schedule"] = sweep_schedule
    node.namespace["v_sweep"] = v_sweep


def _build_opx_program(node, sensors, num_sensors, vgs_id, v_sweep, sweep_schedule, p):
    """Build a single OPX program covering all sweeps in the schedule.

    The schedule is a list of ``(target_gate_name, perturb_gate_name_or_None)``
    tuples.  Each entry becomes one 1D sweep of ``len(v_sweep)`` points.
    """
    with program() as qua_prog:
        seq = node.machine.voltage_sequences[vgs_id]
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        x = declare(fixed)

        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < p.num_shots, n + 1):
                save(n, n_st)

                for target_gate, perturb_gate in sweep_schedule:
                    target_obj = node.machine.get_component(target_gate)

                    if perturb_gate is not None:
                        perturb_obj = node.machine.get_component(perturb_gate)
                        seq.ramp_to_voltages(
                            {perturb_obj.name: p.step_voltage},
                            duration=p.hold_duration,
                            ramp_duration=p.ramp_duration,
                        )

                    with for_(*from_array(x, v_sweep)):
                        ramp_dict = {target_obj.name: x}
                        if perturb_gate is not None:
                            ramp_dict[perturb_obj.name] = p.step_voltage
                        seq.ramp_to_voltages(
                            ramp_dict,
                            duration=p.hold_duration,
                            ramp_duration=p.ramp_duration,
                        )
                        if p.pre_measurement_delay > 0:
                            seq.step_to_voltages({}, duration=p.pre_measurement_delay)
                        align()
                        for i, sensor in multiplexed_sensors.items():
                            rr = sensor.readout_resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            rr.wait(500)
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                    seq.apply_compensation_pulse()

        num_sweeps = len(sweep_schedule)
        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(v_sweep)).buffer(num_sweeps).average().save(f"I{i}")
                Q_st[i].buffer(len(v_sweep)).buffer(num_sweeps).average().save(f"Q{i}")

    return qua_prog


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Simulate the QUA program for sanity-checking."""
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
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.run_in_video_mode
)
def execute_qua_program(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Execute the single QUA program and unpack into per-pair datasets."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        print(f"[cross_capacitance_1d] {job.execution_report()}")

    node.results["ds_raw_all"] = _unpack_sweep_dataset(
        dataset,
        node.namespace["sweep_schedule"],
        node.parameters.cross_capacitance_mapping,
    )


def _unpack_sweep_dataset(ds, sweep_schedule, mapping):
    """Slice a multi-sweep dataset into per-pair ref/shifted 1D datasets.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions ``(sensors, sweep_idx, x_volts)``.
    sweep_schedule : list[tuple[str, str | None]]
        The sweep schedule used during acquisition.
    mapping : dict[str, list[str]]
        The cross_capacitance_mapping from parameters.

    Returns
    -------
    dict[str, xr.Dataset]
        Keyed by ``"{target}_vs_{perturb}_ref"`` and
        ``"{target}_vs_{perturb}_shifted"``, each containing a 1D
        dataset with dimensions ``(sensors, x_volts)``.
    """
    datasets = {}

    ref_indices = {}
    for idx, (target_gate, perturb_gate) in enumerate(sweep_schedule):
        if perturb_gate is None:
            ref_indices[target_gate] = idx
        else:
            pair_key = f"{target_gate}_vs_{perturb_gate}"
            ds_slice = ds.isel(sweep_idx=idx).drop_vars("sweep_idx", errors="ignore")
            datasets[f"{pair_key}_shifted"] = ds_slice

    for target_gate, perturb_gates in mapping.items():
        ds_ref = ds.isel(sweep_idx=ref_indices[target_gate]).drop_vars("sweep_idx", errors="ignore")
        for pg in perturb_gates:
            pair_key = f"{target_gate}_vs_{pg}"
            datasets[f"{pair_key}_ref"] = ds_ref

    return datasets


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Load a previously acquired dataset."""
    # TODO: implement historical data loading
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def analyse_data(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Analyse paired sweeps to extract cross-capacitance coefficients."""
    fit_results = {}
    ds_raw_all = node.results["ds_raw_all"]

    pair_keys = set()
    for prog_key in ds_raw_all:
        pair_key = prog_key.rsplit("_", maxsplit=1)[0]
        pair_keys.add(pair_key)

    for pair_key in pair_keys:
        ref_key = f"{pair_key}_ref"
        shifted_key = f"{pair_key}_shifted"

        if ref_key not in ds_raw_all or shifted_key not in ds_raw_all:
            continue

        ds_ref = process_raw_dataset(ds_raw_all[ref_key], node)
        ds_shifted = process_raw_dataset(ds_raw_all[shifted_key], node)

        target_gate, perturb_gate = pair_key.split("_vs_", maxsplit=1)

        fit_results[pair_key] = extract_cross_capacitance_coefficient(
            ds_ref,
            ds_shifted,
            step_voltage=node.parameters.step_voltage,
            target_gate=target_gate,
            perturb_gate=perturb_gate,
        )

    node.results["fit_results"] = fit_results


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def plot_data(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Plot paired 1D traces with marked transition positions."""
    ds_raw_all = node.results["ds_raw_all"]
    fit_results = node.results.get("fit_results", {})
    figures = {}

    for pair_key, fit_res in fit_results.items():
        ref_key = f"{pair_key}_ref"
        shifted_key = f"{pair_key}_shifted"

        if ref_key not in ds_raw_all or shifted_key not in ds_raw_all:
            continue

        ds_ref = process_raw_dataset(ds_raw_all[ref_key], node)
        ds_shifted = process_raw_dataset(ds_raw_all[shifted_key], node)

        figures[pair_key] = plot_cross_capacitance_1d_diagnostic(
            ds_ref,
            ds_shifted,
            fit_res,
            pair_key,
            step_voltage=node.parameters.step_voltage,
        )

    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def update_state(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Update the compensation matrix with measured cross-capacitance coefficients.

    Supports two modes:
    - ``"additive"``: adds the measured residual to the existing matrix entry
      (for iterative refinement / screening correction).
    - ``"overwrite"``: replaces the entry with the measured value
      (for initial matrix population).
    """
    if "fit_results" not in node.results:
        return

    for pair_key, fit_res in node.results["fit_results"].items():
        if not fit_res.get("fit_params", {}).get("success", False):
            continue

        target_gate = fit_res["target_gate"]
        perturb_gate = fit_res["perturb_gate"]
        alpha = fit_res["coefficient"]

        vgs = None
        for candidate in node.machine.virtual_gate_sets.values():
            source_gates = candidate.layers[0].source_gates
            if target_gate in source_gates and perturb_gate in source_gates:
                vgs = candidate
                break
        if vgs is None:
            raise ValueError(
                f"Could not find a VirtualGateSet containing both " f"'{target_gate}' and '{perturb_gate}'."
            )

        source_gates = vgs.layers[0].source_gates
        target_row = source_gates.index(target_gate)
        perturb_col = source_gates.index(perturb_gate)
        layer = vgs.layers[0]

        if node.parameters.update_mode == "additive":
            current = layer.matrix[target_row][perturb_col]
            new_value = current + alpha
        else:
            new_value = alpha

        target_physical = layer.target_gates[target_row]
        target_ch = vgs.channels[target_physical]

        target_dc = "both" if vgs.id in node.machine.virtual_dc_sets else "opx"
        node.machine.update_cross_compensation_submatrix(
            virtual_names=[perturb_gate],
            channels=[target_ch],
            matrix=[[new_value]],
            target=target_dc,
        )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[CrossCapacitance1DParameters, Quam]):
    """Save the node results and state."""
    # TODO: uncomment when complete
    # node.save()
    pass
