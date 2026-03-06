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
from calibration_utils.gate_virtualization.cross_capacitance_1d_qdac_parameters import (
    CrossCapacitance1DQdacParameters,
)
from calibration_utils.gate_virtualization.scan_utils import _read_qdac_voltage
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
        1D CROSS-CAPACITANCE MEASUREMENT (QDAC)
Measures cross-capacitance matrix entries via paired 1D plunger sweeps.
For each (target_plunger, perturbing_gate) pair, two 1D sweeps of the
target plunger are performed: one at baseline (reference) and one with
the perturbing gate stepped by +step_voltage.  The shift in charge
transition position divided by the step voltage gives the cross-capacitance
coefficient alpha_ij.

This variant supports QDAC-driven sweeps and/or perturbations, using one
QUA program per pair.  For OPX-only operation (single program, faster),
use the companion node ``04_1d_cross_capacitance``.

This is the fast 1D method described in Volk et al., npj Quantum Information
(2019) 5:29, Supplementary Fig. S1.

Prerequisites:
    - Calibrated IQ mixer / Octave on the readout line.
    - Calibrated time of flight, offsets and gains.
    - Calibrated resonators coupled to SensorDot components.
    - Registered QuantumDot and SensorDot elements in QUAM.
    - Configured VirtualGateSet with initial compensation matrix.
    - Configured QdacSpec on each VoltageGate and VirtualDCSet.
"""


node = QualibrationNode[CrossCapacitance1DQdacParameters, Quam](
    name="04b_qdac_1d_cross_capacitance",
    description=description,
    parameters=CrossCapacitance1DQdacParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
def create_qua_program(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
    """Create paired 1D sweep QUA programs for each (target, perturbing) pair.

    For each pair, two programs are created:
    - ``ref``: sweep the target plunger at baseline
    - ``shifted``: set the perturbing gate offset by +step_voltage, then sweep
    """
    node.namespace["sensors"] = sensors = get_sensors(node)
    p = node.parameters

    mapping = p.cross_capacitance_mapping
    if mapping is None:
        raise ValueError(
            "cross_capacitance_mapping must be provided. "
            "Automatic generation from the machine is not yet implemented."
        )

    programs = {}
    sweep_axes_all = {}

    for target_gate, perturb_gates in mapping.items():
        target_obj = node.machine.get_component(target_gate)
        vgs_id = target_obj.voltage_sequence.gate_set.id

        sweep_center = 0.0
        if p.sweep_from_qdac:
            sweep_center = _read_qdac_voltage(node, target_obj)

        v_sweep = np.linspace(
            sweep_center - p.sweep_span / 2,
            sweep_center + p.sweep_span / 2,
            p.sweep_points,
        )

        num_sensors = len(sensors)
        sweep_axes = {
            "sensors": xr.DataArray(sensors.get_names()),
            "x_volts": xr.DataArray(
                v_sweep,
                attrs={"long_name": f"{target_gate} voltage", "units": "V"},
            ),
        }

        for perturb_gate in perturb_gates:
            pair_key = f"{target_gate}_vs_{perturb_gate}"
            perturb_obj = node.machine.get_component(perturb_gate)

            for label, offset in [("ref", 0.0), ("shifted", p.step_voltage)]:
                prog_key = f"{pair_key}_{label}"

                if not p.sweep_from_qdac:
                    qua_prog = _build_1d_opx_program(
                        node,
                        sensors,
                        num_sensors,
                        target_obj,
                        perturb_obj,
                        vgs_id,
                        v_sweep,
                        offset,
                        p,
                    )
                else:
                    qua_prog = _build_1d_qdac_program(
                        node,
                        sensors,
                        num_sensors,
                        target_obj,
                        perturb_obj,
                        vgs_id,
                        v_sweep,
                        offset,
                        p,
                    )

                programs[prog_key] = qua_prog
                sweep_axes_all[prog_key] = sweep_axes

    node.namespace["programs"] = programs
    node.namespace["sweep_axes_all"] = sweep_axes_all


def _build_1d_opx_program(node, sensors, num_sensors, target_obj, perturb_obj, vgs_id, v_sweep, perturb_offset, p):
    """Build a 1D OPX sweep program with optional perturbing gate offset."""
    with program() as qua_prog:
        seq = node.machine.voltage_sequences[vgs_id]
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        x = declare(fixed)

        for multiplexed_sensors in sensors.batch():
            align()
            if perturb_offset != 0.0:
                seq.ramp_to_voltages(
                    {perturb_obj.name: perturb_offset},
                    duration=p.hold_duration,
                    ramp_duration=p.ramp_duration,
                )
            with for_(n, 0, n < p.num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(x, v_sweep)):
                    ramp_voltages = {target_obj.name: x}
                    if perturb_offset != 0.0:
                        ramp_voltages[perturb_obj.name] = perturb_offset
                    seq.ramp_to_voltages(
                        ramp_voltages,
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

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(v_sweep)).average().save(f"I{i}")
                Q_st[i].buffer(len(v_sweep)).average().save(f"Q{i}")

    return qua_prog


def _build_1d_qdac_program(node, sensors, num_sensors, target_obj, perturb_obj, vgs_id, v_sweep, perturb_offset, p):
    """Build a 1D QDAC sweep program with optional perturbing gate offset."""
    from calibration_utils.charge_stability.scan_modes import ScanMode

    scan_mode = ScanMode.from_name(p.scan_pattern)
    node.machine.connect_to_external_source(external_qdac=True)

    dc_list = node.machine.qdac.channel(target_obj.physical_channel.qdac_spec.qdac_output_port).dc_list(
        voltages=scan_mode.get_outer_loop(v_sweep),
        dwell_s=10e-6,
        stepped=True,
    )
    dc_list.start_on_external(trigger=1)

    if perturb_offset != 0.0:
        perturb_port = perturb_obj.physical_channel.qdac_spec.qdac_output_port
        current_v = float(node.machine.qdac.channel(perturb_port).read_dc())
        node.machine.qdac.channel(perturb_port).dc_constant_V(current_v + perturb_offset)

    with program() as qua_prog:
        seq = node.machine.voltage_sequences[vgs_id]
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        trig_counter = declare(int)

        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < p.num_shots, n + 1):
                save(n, n_st)
                with for_(
                    trig_counter,
                    0,
                    trig_counter < int(len(v_sweep)),
                    trig_counter + 1,
                ):
                    target_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")
                    wait(p.post_trigger_wait_ns // 4)
                    align()
                    for i, sensor in multiplexed_sensors.items():
                        rr = sensor.readout_resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        rr.wait(500)
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(v_sweep)).average().save(f"I{i}")
                Q_st[i].buffer(len(v_sweep)).average().save(f"Q{i}")

    return qua_prog


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
    """Simulate the first QUA program for sanity-checking."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    first_key = next(iter(node.namespace["programs"]))
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["programs"][first_key], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.run_in_video_mode
)
def execute_qua_program(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
    """Execute all paired sweeps and store raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    datasets = {}

    for prog_key, qua_prog in node.namespace["programs"].items():
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes_all"][prog_key])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            print(f"[{prog_key}] {job.execution_report()}")
        datasets[prog_key] = dataset

    node.results["ds_raw_all"] = datasets


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
    """Load a previously acquired dataset."""
    # TODO: implement historical data loading
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def analyse_data(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
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
def plot_data(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
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
def update_state(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
    """Update the compensation matrix with measured cross-capacitance coefficients."""
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
def save_results(node: QualibrationNode[CrossCapacitance1DQdacParameters, Quam]):
    """Save the node results and state."""
    # TODO: uncomment when complete
    # node.save()
    pass
