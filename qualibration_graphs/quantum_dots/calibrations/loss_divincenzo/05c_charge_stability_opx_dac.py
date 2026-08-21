# %% {Imports}
import time

import matplotlib.pyplot as plt
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam
from calibration_utils.charge_stability_opx import (
    analyse_raw_data,
    plot_all,
)
from calibration_utils.charge_stability_external_dac import (
    Parameters,
    get_voltage_arrays,
    validate_opx_limit,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.common_utils.experiment import get_sensors


description = """
2D CHARGE STABILITY MAP (OPX + external DAC)

This sequence measures a 2D charge-stability diagram by sweeping one fast virtual-gate
axis directly with the OPX and stepping one slow axis from the host through an external
DAC. At each slow-axis value, the QUA program pauses, the host applies the next DAC
voltage, and the OPX resumes the full fast-axis sweep while reading out one or more
sensor dots. Charge transitions appear as edges in the demodulated I/Q response.

The OPX-driven axis includes ``opx_offset`` directly in the fast sweep values. The
external-DAC axis resolves its center from ``dac_offset`` or, when ``dac_offset=None``,
from the currently applied DAC value at execution time.

Prerequisites:
    - IQ mixer/Octave calibrated on the readout line (01a_mixer_calibration).
    - Time of flight, offsets, and gains calibrated (01a_time_of_flight).
    - Sensor resonators calibrated (02a_resonator_spectroscopy, 02b_resonator_spectroscopy_vs_power).
    - QUAM initialized with readout amplitude/duration, QuantumDot and SensorDot elements.
    - External DAC axis mapped through the VirtualDCSet associated with the OPX fast axis gate set.

Datasets:
    - ``ds_raw``: untouched I/Q fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed maps plus edge-analysis outputs (when ``perform_edge_analysis=True``).
      Used by ``plot_data`` when edge analysis is enabled.

Results (``node.results["fit_results"][<sensor>]``, when ``perform_edge_analysis=True``):
    - ``success``: whether edge detection and line fitting completed.
    - ``segments``: fitted charge-transition line segments.
    - ``intersections``: detected triple-point locations.

Figures (``node.results["figures"]``):
    - ``"amplitude"``: |I + iQ| heatmap vs (x_volts, y_volts) for each sensor.
    - ``"phase"``: IQ phase heatmap vs (x_volts, y_volts) for each sensor.
    - ``"<sensor>_change_points"``: change-point overlays (when edge analysis enabled).
    - ``"<sensor>_line_fits"``: fitted transition lines (when edge analysis enabled).

State update:
    - None (diagnostic map; use VirtualGateSet voltage points or downstream nodes to set bias).
"""


node = QualibrationNode[Parameters, Quam](
    name="05c_charge_stability_opx_dac", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # ── Experiment parameters (Python side) ──────────────────────────────
    # Extract the relevant sensors from the node
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # The name of the OPX axis to be swept
    opx_axis_name = node.parameters.opx_fast_axis_name

    # Extract the voltage array and the existing voltage points
    opx_array, dac_array, vgs_id = get_voltage_arrays(node)
    node.namespace["voltage_points"] = node.machine.virtual_gate_sets[vgs_id].get_macros()

    # Ensure that no virtual value in the OPX array resolves beyond the OPX limit.
    validate_opx_limit(node, opx_axis_name, opx_array)

    # Register the sweep axes to be added to the dataset when fetching data.
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "y_volts": xr.DataArray(
            dac_array,
            attrs={"long_name": "voltage", "units": "V"},
        ),
        "x_volts": xr.DataArray(
            opx_array,
            attrs={"long_name": "voltage", "units": "V"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[vgs_id]

        # Real-time variables:
        # n  : shot counter within one slow-axis line
        # y_idxs : slow-axis line index, advanced by the host between pauses
        # x  : the current fast-axis OPX voltage
        # progress : monotonically increasing point counter exposed to the PC
        # I : the measured raw I value
        # Q : the measured raw Q value
        # I_st, Q_st : per-sensor streams of raw data
        # n_st : progress stream used by the host-side progress counter
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        y_idxs = declare(int)
        x = declare(fixed)
        progress = declare(int)

        # Loop over the multiplexed sensors in this outer array. The full 2D map will be repeated for each batch.
        for multiplexed_sensors in sensors.batch():
            # This defines the TOTAL HOLD DURATION on each pixel, INCLUDING the readout time.
            # So the duration for each pixel becomes opx_hold_duration + longest readout time in this batch.
            pixel_hold_duration = node.parameters.opx_hold_duration + max(
                s.readout_resonator.operations["readout"].length for s in multiplexed_sensors.values()
            )
            initial_x_value = float(opx_array[0])

            # ── OUTER LOOP: step through slow-axis DAC values ───────────────────────
            with for_(y_idxs, 0, y_idxs < len(dac_array), y_idxs + 1):

                # The host applies the next DAC value while the OPX is paused at the start of each line.
                # Skip the pause during simulation, otherwise the simulator would hang indefinitely.
                if not node.parameters.simulate:
                    pause()

                # ── MIDDLE LOOP: average over shots ───────────────────────
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    assign(progress, y_idxs * node.parameters.num_shots + n)
                    save(progress, n_st)

                    align()  # Start with a global align

                    # Optionally move to the first fast-axis point and allow the electrostatics to settle.
                    if node.parameters.per_line_wait > 0:
                        seq.step_to_voltages(
                            {opx_axis_name: initial_x_value},
                            duration=node.parameters.per_line_wait,
                            ramp_duration=node.parameters.opx_ramp_duration,
                        )
                        for i, s in multiplexed_sensors.items():
                            rr = s.readout_resonator
                            rr.wait((node.parameters.per_line_wait + node.parameters.opx_ramp_duration) // 4)

                    # ── INNER LOOP: sweep the fast OPX axis ────────────────
                    with for_(*from_array(x, opx_array)):
                        seq.ramp_to_voltages(
                            {opx_axis_name: x},
                            duration=pixel_hold_duration,
                            ramp_duration=node.parameters.opx_ramp_duration,
                        )
                        for i, s in multiplexed_sensors.items():
                            rr = s.readout_resonator
                            rr.wait((node.parameters.opx_ramp_duration + node.parameters.opx_hold_duration) // 4)
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                        align()
                    if node.parameters.per_line_compensation:
                        seq.apply_compensation_pulse(
                            go_to_zero=True, return_to_zero=True, max_voltage=node.parameters.max_compensation_voltage
                        )
                    seq.ramp_to_zero()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(opx_array)).buffer(node.parameters.num_shots).map(FUNCTIONS.average()).buffer(
                    len(dac_array)
                ).save(f"I{i}")
                Q_st[i].buffer(len(opx_array)).buffer(node.parameters.num_shots).map(FUNCTIONS.average()).buffer(
                    len(dac_array)
                ).save(f"Q{i}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX and simulate the QUA program."""
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()

    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        # "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX, execute the QUA program, and fetch raw I/Q into ``ds_raw``."""
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()

    # Prepare DAC — this node performs a hybrid 2D scan: the X axis is swept by the OPX in
    # QUA, while the Y axis is stepped externally. The QUA program calls pause() at the start
    # of each Y line (see create_qua_program); the loop below resumes the job only after
    # the DAC voltage has been updated.

    # Currently, the default behaviour uses the VirtualDCSet to resolve the offset and step the DAC.
    # What this means:
    # RESOLVING the offset:
    #   - If the offset is None, then the DAC offset should be the currently outputted value from the DAC.
    #   - If the offset is not None, then the DAC sweep should be centred on that value.
    # APPLY the offset:
    #   - Using the resolved offset, apply centre + sweep value to the DAC on each loop.
    # RESTORE the offset:
    #   - At the end of the 2D map, restore the offset to either what it was already outputting (if dac_offset = None) or the new offset

    # ── RESOLVE the offset, APPLY the offset and RESTORE the offset ─────────
    # Populate these with your own DAC API. Current default is via the VirtualDCSet.

    def resolve_dac_offset() -> float:
        """Resolve the absolute DAC center used for the slow-axis sweep."""
        dac_axis_name = node.parameters.dac_slow_axis_name
        vgs_id = node.namespace["axes_names"]["gate_set_id"]
        virtual_dc_set = node.machine.virtual_dc_sets[vgs_id]

        dac_offset = node.parameters.dac_offset
        if dac_offset is None:
            dac_offset = virtual_dc_set.get_voltage(dac_axis_name, requery=True)

        node.namespace["dac_offset"] = dac_offset
        return float(dac_offset)

    def apply_dac_value(dac_value: float, dac_offset: float) -> float:
        """Apply one slow-axis DAC value using the configured VirtualDCSet."""
        dac_axis_name = node.parameters.dac_slow_axis_name
        vgs_id = node.namespace["axes_names"]["gate_set_id"]
        virtual_dc_set = node.machine.virtual_dc_sets[vgs_id]

        dac_value_to_play = float(dac_offset + dac_value)
        virtual_dc_set.set_voltages({dac_axis_name: dac_value_to_play})
        return dac_value_to_play

    def restore_dac_offset(dac_offset: float) -> None:
        """Restore the slow-axis DAC to the resolved center after the scan finishes."""
        dac_axis_name = node.parameters.dac_slow_axis_name
        vgs_id = node.namespace["axes_names"]["gate_set_id"]
        virtual_dc_set = node.machine.virtual_dc_sets[vgs_id]
        virtual_dc_set.set_voltages({dac_axis_name: float(dac_offset)})

    # ── RESOLVE the DAC offset ─────────
    dac_offset = resolve_dac_offset()
    total_points = node.parameters.num_shots * len(node.namespace["dac_values"])

    try:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

            for i, dac_value in enumerate(node.namespace["dac_values"]):
                while not job.is_paused():
                    time.sleep(0.1)

                # ── APPLY the DAC value ─────────
                applied_value = apply_dac_value(dac_value, dac_offset)
                node.log(
                    f"Applying {applied_value} V to the DAC ({100 * (i + 1) / len(node.namespace['dac_values']):.1f}%)"
                )
                job.resume()

            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    total_points,
                    start_time=data_fetcher.t_start,
                    node=node,
                )
            node.log(job.execution_report())
    finally:
        # ── RESTORE the DAC offset ─────────
        restore_dac_offset(dac_offset)

    dataset = dataset.transpose("sensors", "x_volts", "y_volts")
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the sensors from the loaded node parameters
    node.namespace["sensors"] = [node.machine.sensor_dots[name] for name in node.parameters.sensor_names]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate or not node.parameters.perform_edge_analysis)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process ``ds_raw``, fit edge data, and store processed outputs in ``ds_fit``."""
    (
        node.results["ds_fit"],
        node.results["fit_results"],
        node.outcomes,
    ) = analyse_raw_data(node.results["ds_raw"], node, log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Build the node figures from the raw and fitted charge-stability data."""
    point_kwargs = {}
    if node.parameters.plot_points and "voltage_points" in node.namespace:
        pair_prefix = node.machine.find_quantum_dot_pair(
            node.parameters.opx_fast_axis_name, node.parameters.dac_slow_axis_name
        )
        point_kwargs = dict(
            voltage_points=node.namespace["voltage_points"],
            x_axis_name=node.parameters.opx_fast_axis_name,
            y_axis_name=node.parameters.dac_slow_axis_name,
            pair_prefix=pair_prefix,
        )

    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["sensors"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=node.results.get("fit_results"),
        perform_edge_analysis=node.parameters.perform_edge_analysis,
        **point_kwargs,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No QuAM state is updated for this diagnostic node."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()
