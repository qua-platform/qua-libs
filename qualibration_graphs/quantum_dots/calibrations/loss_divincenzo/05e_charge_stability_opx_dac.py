# %% {Imports}
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from qm.qua import *
from qm import generate_qua_script

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualang_tools.loops import from_array
from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.charge_stability import (
    OPXuDACParameters as Parameters,
    get_voltage_arrays,
    ScanMode,
    plot_raw_amplitude,
    plot_raw_phase,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_change_point_overlays,
    plot_line_fit_overlays,
    get_axis_names,
    pca_plotter,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates
from calibration_utils.common_utils.experiment import (
    get_dots,
    get_sensors,
)

description = """
            2D OPX CHARGE STABILITY MAP
This script involves a simple 2D voltage map, done by stepping the X and Y Quantum Dots
to their corresponding voltages, sending a readout pulse, and demodulating the 'I' and 'Q'
quadratures. This is performed solely using the OPX to sweep/step the axes.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having calibrated the resonators coupled to the SensorDot components (nodes 02a_resonator_spectroscopy.py, 02b_resonator_spectroscopy_vs_power.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration.
    - Having registered the QuantumDot elements and your SensorDot elements in your QUAM state.
"""


node = QualibrationNode[Parameters, Quam](
    name="05e_charge_stability_opx_dac", description=description, parameters=Parameters()
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
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    node.namespace["sensors"] = sensors = get_sensors(node)
    x_name = node.parameters.x_axis_name
    node.namespace["axes_names"] = {"x_axis": x_name}
    virtual_gate_set_name = "main_qpu"

    x_volts, y_volts = get_voltage_arrays(
        node
    )
    num_sensors = len(sensors)

    node.namespace["dac_values"] = y_volts
    opx_voltage_limit = +2.5
    max_x = np.max(np.abs(x_volts))
    max_x_physical = node.machine.virtual_gate_sets[virtual_gate_set_name].resolve_voltages({x_name: max_x})
    assert np.max(np.abs(list(max_x_physical.values()))) <= opx_voltage_limit

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "y_volts": xr.DataArray(
            y_volts,
            attrs={"long_name": "voltage", "units": "V"},
        ),
        "x_volts": xr.DataArray(
            x_volts,
            attrs={"long_name": "voltage", "units": "V"},
        ),
    }
    sensor = node.parameters.sensor_names[0]
    readout_time = node.machine.sensor_dots[sensor].readout_resonator.operations['readout'].length
    settle_time = 1000 #40000

    # node.namespace["sweep"]
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[virtual_gate_set_name]

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
            num_IQ_pairs=num_sensors
        )
        y_idxs = declare(int)
        x = declare(fixed)

        with for_(y_idxs, 0, y_idxs<len(y_volts), y_idxs + 1): 
            if not node.parameters.simulate:
                pause()
            if node.parameters.per_line_wait > 0: 
                seq.step_to_voltages({}, duration = node.parameters.per_line_wait)
            with for_(n, 0, n<node.parameters.num_shots, n+1):
                save(n, n_st)

                with for_(*from_array(x, x_volts)):
                    align()
                    seq.ramp_to_voltages({x_name: x}, duration = readout_time + settle_time, ramp_duration = node.parameters.ramp_duration)                    
                    for i, s in sensors.batch()[0].items():
                        rr = s.readout_resonator
                        rr.wait(node.parameters.ramp_duration + settle_time)
                        I[i], Q[i] = rr.measure(
                            "readout"
                        )
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()
                if node.parameters.per_line_compensation: 
                    seq.apply_compensation_pulse(go_to_zero = True, return_to_zero = True, max_voltage = node.parameters.max_compensation_voltage)
                seq.ramp_to_zero()

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(x_volts)).buffer(node.parameters.num_shots).map(FUNCTIONS.average()).buffer(len(y_volts)).save(f"I{i}")
                Q_st[i].buffer(len(x_volts)).buffer(node.parameters.num_shots).map(FUNCTIONS.average()).buffer(len(y_volts)).save(f"Q{i}")
                


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX and simulate the QUA program"""
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    serialised_programme = generate_qua_script(node.namespace["qua_program"])
    print(serialised_programme)

    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    qmm = node.machine.connect()

    # Prepare DAC — this node performs a hybrid 2D scan: the X axis is swept by the OPX in
    # QUA, while the Y axis is stepped externally. The QUA program calls pause() at the start
    # of each Y line (see create_qua_program); the loop below resumes the job only after
    # the DAC voltage has been updated.
    #
    # Before commenting out the raise below, replace this block with your lab-specific DAC
    # driver code that:
    #   1. Connects to the external source(s) used for the Y axis (e.g. instantiate dac1,
    #      dac2, or a single instrument handle).
    #   2. Exposes a `gates` object where getattr(gates, node.parameters.y_axis_name) returns
    #      a gate with get_voltage() and set_voltage(voltage) — the offset read at the start
    #      of the scan is added to each y value from node.namespace["dac_values"] below.
    #   3. Provides a close_dacs(...) helper (or equivalent) to disconnect/reset the hardware
    #      after the scan completes and the Y-axis offset is restored.
    # raise NotImplementedError(
    #     "Hardware execution for this node requires a DAC/gate-manager driver that is "
    #     "specific to your lab's setup. Plug in your own driver here to obtain a `gates` "
    #     "handle exposing `get_voltage`/`set_voltage` for `node.parameters.y_axis_name`."
    # )

    # Get the config from the machine
    config = node.machine.generate_config()

    y_name = node.parameters.y_axis_name

    gate = getattr(gates, y_name)
    # Query DAC offset
    dac_offset = gate.get_voltage()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).

    qm = qmm.open_qm(config)
    node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
    for i, y_value in enumerate(node.namespace["dac_values"]):
        while not job.is_paused(): 
            time.sleep(0.1)
        
        dac_value_to_play = dac_offset + y_value

        print(f"Applying {dac_value_to_play} to the DAC ({100*i/len(node.namespace["dac_values"])} %)")
        gate.set_voltage(dac_value_to_play)
        # print(dac_value_to_play)
        print(f"Voltage set to {gate.get_voltage()}")

        job.resume()

    # Display the progress bar
    data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
    for dataset in data_fetcher:
        progress_counter(
            data_fetcher.get("n", 0),
            node.parameters.num_shots,
            start_time=data_fetcher.t_start,
            node=node,
        )
    # Display the execution report to expose possible runtime errors
    print(job.execution_report())
    # Canonicalize to (sensors, x_volts, y_volts) for downstream processing.
    dataset = dataset.transpose("sensors", "x_volts", "y_volts")
    # Register the raw dataset, reordering if the scan mode requires it (e.g. spiral)
    node.results["ds_raw"] = dataset

    gate.set_voltage(dac_offset)
    
    # Close the DAC here if your lab-specific driver supports it


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the sensors from the loaded node parameters
    node.namespace["sensors"] = [
        node.machine.sensor_dots[name] for name in node.parameters.sensor_names
    ]


# %% {Analyse Data}
@node.run_action(
    skip_if=node.parameters.simulate
    or not node.parameters.perform_edge_analysis
)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: v.to_dict() for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by sensors.grid_location."""
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    ds_plot = node.results["ds_raw"].copy()

    point_kwargs = {}
    if node.parameters.plot_points and "voltage_points" in node.namespace:
        pair_prefix = node.machine.find_quantum_dot_pair(
            node.parameters.x_axis_name, node.parameters.y_axis_name
        )
        point_kwargs = dict(
            voltage_points=node.namespace["voltage_points"],
            x_axis_name=node.parameters.x_axis_name,
            y_axis_name=node.parameters.y_axis_name,
            pair_prefix=pair_prefix,
        )

    fig_amplitude = plot_raw_amplitude(
        ds_plot, node.namespace["sensors"], **point_kwargs
    )
    fig_pca = None
    if node.parameters.plot_pca:
        fig_pca = pca_plotter(ds_plot)

    fig_phase = plot_raw_phase(ds_plot, node.namespace["sensors"], **point_kwargs)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_amplitude,
        "phase": fig_phase,
    }
    if fig_pca is not None:
        node.results["figures"]["pca"] = fig_pca
    if node.parameters.perform_edge_analysis and "fit_results" in node.results:
        for sensor in node.namespace["sensors"]:
            sensor_data = ds_plot.sel(sensors=sensor.id)
            fit_params = node.results["fit_results"].get(sensor.id, {})
            fig_cp = plot_change_point_overlays(sensor_data, fit_params, sensor.id)
            node.results["figures"][f"{sensor.id}_change_points"] = fig_cp
            if fit_params.get("segments"):
                fig_lines = plot_line_fit_overlays(sensor_data, fit_params, sensor.id)
                node.results["figures"][f"{sensor.id}_line_fits"] = fig_lines


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
