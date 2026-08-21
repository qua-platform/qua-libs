# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.common_utils.experiment import get_dots, get_sensors
from calibration_utils.hello_qua import (
    Parameters,
    extract_longest_readout_time,
    extract_vgs_id,
    process_raw_dataset,
    log_processed_summary,
    plot_all,
)

from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

description = """
        HELLO QUA — CONNECTIVITY & SANITY CHECK

Basic script to play with the QUA program and test the QOP connectivity. For each
selected quantum dot, this node sweeps a single gate voltage over a small span and reads
out the response on each selected sensor via RF reflectometry, then plots the raw I/Q
traces. It is a diagnostic tool, not a calibration: it does not fit or extract any
physical parameter, and does not write anything to the QUAM state.

Prerequisites:
    - QUAM initialized and channels wired (``quam_config/populate_quam_state_*.py``).
    - Sensor-dot readout resonators configured (readout amplitude/duration set, even if
      not yet optimally tuned).

Datasets:
    - ``ds_raw``: shot-averaged ``I``/``Q`` vs (``sensors``, ``quantum_dots``, ``voltage``),
      plus derived ``amplitude``/``phase`` once processed.

Figures:
    - One figure per quantum dot, with one I/Q-vs-voltage subplot per sensor.

State update:
    - None. This node is a connectivity check; no QUAM state is modified.
"""


node = QualibrationNode[Parameters, Quam](name="00_hello_qua", description=description, parameters=Parameters())


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

    # Extract the quantum dots and sensors to be used in this measurement
    node.namespace["quantum_dots"] = quantum_dots = get_dots(node)
    node.namespace["sensors"] = sensors = get_sensors(node)

    # Ensure that the machine is set up to track the integrated voltage
    node.machine.reset_voltage_sequence(extract_vgs_id(quantum_dots), track_integrated_voltage=True)

    # Extract the number of sensors, and the maximum length of the readout pulse in the list of sensors.
    # We extract this, so that we can allow the voltage to dwell exactly for the readout time.
    num_sensors, max_readout_length = len(sensors), extract_longest_readout_time(sensors)

    # Number of shots per sweep point
    n_avg = node.parameters.num_shots

    # Extract how long to stay on each point
    dwell_time_on_voltage = node.parameters.dwell_time
    ramp_duration_to_voltage = node.parameters.ramp_duration

    # Optional offset via the OPX
    v_center = node.parameters.v_center
    if v_center is None:
        v_center = 0.0

    # Construct the array of voltage values
    voltages = np.linspace(
        v_center - node.parameters.v_span / 2, v_center + node.parameters.v_span / 2, node.parameters.n_points
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "quantum_dots": xr.DataArray(quantum_dots.get_names()),
        "voltage": xr.DataArray(voltages, attrs={"long_name": "voltage", "units": ""}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I_st, Q_st   : buffers collecting I/Q before transfer to PC
        #   I, Q         : QUA variables storing the outcome of the measurements to be saved into the streams above
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        #   v            : QUA variable holding the voltage value to apply to the plunger
        I_st = {qd.name: [declare_output_stream() for _ in sensors] for qd in quantum_dots}
        Q_st = {qd.name: [declare_output_stream() for _ in sensors] for qd in quantum_dots}
        I = {qd.name: [declare(fixed) for _ in sensors] for qd in quantum_dots}
        Q = {qd.name: [declare(fixed) for _ in sensors] for qd in quantum_dots}
        n_st = declare_output_stream()
        n = declare(int)
        v = declare(fixed)

        # ── OUTER LOOP: repeat the full sweep n_avg times ──
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # Tell the PC which shot we are on

            # Python loop over the relevant quantum dots
            for qd in quantum_dots:
                # Extract the quantum dot's run-time helper for voltage stepping and ramping
                seq = qd.voltage_sequence

                # Start with a global align
                align()

                # ── INNER LOOP: Sweep the voltage  ───────────────────────
                with for_(*from_array(v, voltages)):

                    # Use the VoltageSequence run-time helper to ramp to voltages
                    # This can be used with any physical or virtual voltage in the gate set
                    seq.ramp_to_voltages(
                        voltages={qd.name: v},
                        duration=dwell_time_on_voltage
                        + max_readout_length,  # First dwell, then wait for the longest readout pulse
                        ramp_duration=ramp_duration_to_voltage,
                    )

                    # Measure each batch, multiplexed by sensors
                    for multiplexed_sensors in sensors.batch():
                        for i, sensor in multiplexed_sensors.items():
                            # Select the resonator tied to the sensor
                            rr = sensor.readout_resonator
                            # Have the readout resonator wait for the ramp + dwell time
                            rr.wait((ramp_duration_to_voltage + dwell_time_on_voltage) // 4)
                            # Measure using said resonator
                            rr.measure("readout", qua_vars=(I[qd.name][i], Q[qd.name][i]))
                            # Save data
                            save(I[qd.name][i], I_st[qd.name][i])
                            save(Q[qd.name][i], Q_st[qd.name][i])

                seq.apply_compensation_pulse(max_voltage=node.parameters.max_compensation_voltage)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for qd in quantum_dots:
                for i in range(num_sensors):
                    # Each save() above is one voltage point.
                    # .buffer(len(voltages)) : group points along the voltage axis
                    # .average() : group points along the repetitions axis
                    # Result : 1D trace I(voltages), Q(voltages) per quantum dot
                    I_st[qd.name][i].buffer(len(voltages)).average().save(f"I_{qd.name}_{i}")
                    Q_st[qd.name][i].buffer(len(voltages)).average().save(f"Q_{qd.name}_{i}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    # Connect to the QOP
    qmm = node.machine.connect(timeout=500)
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
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, {"voltage": node.namespace["sweep_axes"]["voltage"]})
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["quantum_dots"] = get_dots(node)
    node.namespace["sensors"] = get_sensors(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Derive amplitude and phase from the raw I/Q dataset (keeps ds_raw's original I/Q columns intact)."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"."""
    log_processed_summary(
        node.results["ds_raw"],
        node.namespace["quantum_dots"],
        node.namespace["sensors"],
        log_callable=node.log,
    )
    node.outcomes = {qd.name: "successful" for qd in node.namespace["quantum_dots"]}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot I and Q vs voltage for each quantum dot, showing all sensor responses."""
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["quantum_dots"],
        node.namespace["sensors"],
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No QuAM state is updated by this connectivity check."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
