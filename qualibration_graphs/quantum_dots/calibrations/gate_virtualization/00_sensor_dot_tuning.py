# %% {Imports}
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.gate_virtualization.sensor_dot_tuning_parameters import (
    SensorDotTuningParameters,
)
from calibration_utils.gate_virtualization.base_parameters import get_voltage_arrays
from calibration_utils.gate_virtualization.scan_utils import _read_qdac_voltage
from calibration_utils.gate_virtualization.analysis import process_raw_dataset
from calibration_utils.gate_virtualization.sensor_dot_analysis import (
    fit_lorentzian,
    lorentzian,
)
from calibration_utils.common_utils.experiment import get_sensors

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


description = """
           SENSOR DOT TUNING — 1D COULOMB PEAK SWEEP
Sweeps a sensor gate voltage across its Coulomb peak, fits a Lorentzian to
the measured response, and sets the operating point at the inflection point
x0 ± γ/(2√3) where the charge sensitivity is maximal.

If the sweep is performed via the QDAC, the QDAC channel is updated to the
optimal voltage.  If via the OPX, the fitted optimal point is saved as
``sensor_opt`` for use by downstream nodes.

Prerequisites:
    - Calibrated IQ mixer / Octave on the readout line.
    - Calibrated time of flight, offsets and gains.
    - Registered SensorDot elements in QUAM.
    - (If using QDAC) Configured QdacSpec on the sensor gate.
"""


node = QualibrationNode[SensorDotTuningParameters, Quam](
    name="00_sensor_dot_tuning",
    description=description,
    parameters=SensorDotTuningParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Allow the user to locally set node parameters for debugging."""
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.run_in_video_mode
)
def create_qua_program(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Create a 1D sensor gate sweep QUA program for each sensor."""
    node.namespace["sensors"] = sensors = get_sensors(node)

    sensor_gate_names = node.parameters.sensor_gate_names
    if sensor_gate_names is None:
        sensor_gate_names = [s.name for s in sensors]

    programs = {}
    sweep_axes_all = {}
    for gate_name in sensor_gate_names:
        gate_obj = node.machine.get_component(gate_name)

        sweep_center = node.parameters.sweep_center
        if node.parameters.from_qdac and sweep_center is None:
            sweep_center = _read_qdac_voltage(node, gate_obj)

        center = sweep_center if sweep_center is not None else 0.0
        span = node.parameters.sweep_span
        n_pts = node.parameters.sweep_points
        v_sweep = np.linspace(center - span / 2, center + span / 2, n_pts)

        num_sensors = len(sensors)
        sweep_axes = {
            "sensors": xr.DataArray(sensors.get_names()),
            "x_volts": xr.DataArray(
                v_sweep,
                attrs={"long_name": f"{gate_name} voltage", "units": "V"},
            ),
        }

        vgs_id = gate_obj.voltage_sequence.gate_set.id

        if not node.parameters.from_qdac:
            with program() as qua_prog:
                seq = node.machine.voltage_sequences[vgs_id]
                I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
                    num_IQ_pairs=num_sensors
                )
                x = declare(fixed)
                for multiplexed_sensors in sensors.batch():
                    align()
                    with for_(n, 0, n < node.parameters.num_shots, n + 1):
                        save(n, n_st)
                        with for_(*from_array(x, v_sweep)):
                            seq.ramp_to_voltages(
                                {gate_obj.name: x},
                                duration=node.parameters.hold_duration,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                            if node.parameters.pre_measurement_delay > 0:
                                seq.step_to_voltages(
                                    {}, duration=node.parameters.pre_measurement_delay
                                )
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
        else:
            from calibration_utils.charge_stability.scan_modes import ScanMode

            scan_mode = ScanMode.from_name(node.parameters.scan_pattern)
            node.machine.connect_to_external_source(external_qdac=True)
            dc_list = node.machine.qdac.channel(
                gate_obj.physical_channel.qdac_spec.qdac_output_port
            ).dc_list(
                voltages=scan_mode.get_outer_loop(v_sweep),
                dwell_s=10e-6,
                stepped=True,
            )
            dc_list.start_on_external(trigger=1)

            with program() as qua_prog:
                seq = node.machine.voltage_sequences[vgs_id]
                I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
                    num_IQ_pairs=num_sensors
                )
                trig_counter = declare(int)
                for multiplexed_sensors in sensors.batch():
                    align()
                    with for_(n, 0, n < node.parameters.num_shots, n + 1):
                        save(n, n_st)
                        with for_(
                            trig_counter, 0,
                            trig_counter < int(len(v_sweep)),
                            trig_counter + 1,
                        ):
                            gate_obj.physical_channel.qdac_spec.opx_trigger_out.play(
                                "trigger"
                            )
                            wait(node.parameters.post_trigger_wait_ns // 4)
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

        programs[gate_name] = qua_prog
        sweep_axes_all[gate_name] = sweep_axes

    node.namespace["programs"] = programs
    node.namespace["sweep_axes_all"] = sweep_axes_all
    node.namespace["sensor_gate_names"] = sensor_gate_names


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Simulate the first QUA program for sanity-checking."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    first_key = next(iter(node.namespace["programs"]))
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["programs"][first_key], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.run_in_video_mode
)
def execute_qua_program(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Execute sensor gate sweeps and store raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    datasets = {}
    for gate_name, qua_prog in node.namespace["programs"].items():
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            data_fetcher = XarrayDataFetcher(
                job, node.namespace["sweep_axes_all"][gate_name]
            )
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            print(f"[{gate_name}] {job.execution_report()}")
        datasets[gate_name] = dataset
    node.results["ds_raw_all"] = datasets


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def analyse_data(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Fit a Lorentzian to each sensor sweep and extract the operating point."""
    fit_results = {}
    for gate_name, ds_raw in node.results["ds_raw_all"].items():
        I = ds_raw["I"].values
        Q = ds_raw["Q"].values
        amplitude = np.sqrt(I ** 2 + Q ** 2)

        if amplitude.ndim == 2:
            signal = amplitude[0]
        else:
            signal = amplitude

        v = ds_raw.coords["x_volts"].values

        result = fit_lorentzian(v, signal, side=node.parameters.operating_side)
        fit_results[gate_name] = result

    node.results["fit_results"] = fit_results


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Plot sensor sweeps with Lorentzian fits and marked operating points."""
    fit_results = node.results.get("fit_results", {})
    figures = {}

    for gate_name, ds_raw in node.results["ds_raw_all"].items():
        I = ds_raw["I"].values
        Q = ds_raw["Q"].values
        amplitude = np.sqrt(I ** 2 + Q ** 2)
        signal = amplitude[0] if amplitude.ndim == 2 else amplitude
        v = ds_raw.coords["x_volts"].values

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(v * 1e3, signal, "k.", markersize=2, label="Data")

        if gate_name in fit_results:
            fr = fit_results[gate_name]
            v_fit = np.linspace(v[0], v[-1], 500)
            ax.plot(
                v_fit * 1e3,
                lorentzian(v_fit, fr.x0, fr.gamma, fr.amplitude, fr.offset),
                "r-", linewidth=1.5, label="Lorentzian fit",
            )
            ax.axvline(fr.x0 * 1e3, color="blue", linestyle="--", alpha=0.6, label=f"x0 = {fr.x0 * 1e3:.2f} mV")
            ax.axvline(
                fr.optimal_voltage * 1e3, color="green", linestyle="-",
                linewidth=2, label=f"Operating pt = {fr.optimal_voltage * 1e3:.2f} mV",
            )

        ax.set_xlabel("Sensor gate voltage (mV)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title(f"Sensor tuning: {gate_name}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        figures[gate_name] = fig

    node.results["figures"] = figures
    if figures:
        node.results["figure"] = next(iter(figures.values()))


# %% {Update_state}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def update_state(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Set the sensor operating point — QDAC voltage or saved result."""
    fit_results = node.results.get("fit_results", {})

    for gate_name, fr in fit_results.items():
        if fr is None:
            continue
        if node.parameters.from_qdac:
            gate_obj = node.machine.get_component(gate_name)
            port = gate_obj.physical_channel.qdac_spec.qdac_output_port
            node.machine.qdac.channel(port).dc_constant_V(fr.optimal_voltage)
            print(f"[{gate_name}] QDAC updated to {fr.optimal_voltage * 1e3:.2f} mV")
        else:
            node.results.setdefault("sensor_opt", {})[gate_name] = fr.optimal_voltage
            print(f"[{gate_name}] sensor_opt = {fr.optimal_voltage * 1e3:.2f} mV")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[SensorDotTuningParameters, Quam]):
    """Save the node results and state."""
    # node.save()
    pass
