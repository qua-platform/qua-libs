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

from calibration_utils.common_utils.experiment import get_dots
from calibration_utils.hello_qua import (
    Parameters,
)

from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

description = """
        HELLO QUA — CONNECTIVITY & SANITY CHECK

Basic script to play with the QUA program and test the QOP connectivity. 

For each selected quantum dot, this node sweeps a single gate voltage over a small span. 
It is a diagnostic tool, not a calibration: it does not fit or extract any physical parameter, 
and does not write anything to the QUAM state.

Prerequisites:
    - QUAM initialized and channels wired (``quam_config/populate_quam_state_*.py``).
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

    # Number of shots per sweep point
    n_avg = node.parameters.num_shots

    # Extract the quantum dots and sensors to be used in this measurement
    node.namespace["quantum_dot"] = quantum_dots = get_dots(node)

    # Set up a symmetric gate sweep
    volts = np.linspace(-0.01, 0.01, 11)

    # Set up voltage tracking. The gate_set name is hard-coded here, but can be extracted from
    # the quantum dot via quantum_dot.voltage_sequence.gate_set.name
    node.machine.reset_voltage_sequence("main_qpu", track_integrated_voltage=True)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "quantum_dots": xr.DataArray(quantum_dots),
        "voltage": xr.DataArray(volts, attrs={"long_name": "voltage", "units": ""}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I_st, Q_st   : buffers collecting I/Q before transfer to PC
        #   I, Q         : QUA variables storing the outcome of the measurements to be saved into the streams above
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        #   v            : QUA variable holding the voltage value to apply to the plunger
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=1)
        v = declare(fixed)

        # ── OUTER LOOP: repeat the full sweep n_avg times ──
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)  # Tell the PC which shot we are on

            for quantum_dot in quantum_dots:
                # Extract the quantum dot's run-time helper for voltage stepping and ramping
                seq = quantum_dot.voltage_sequence

                # Start with a global align
                align()

                # ── INNER LOOP: Sweep the voltage  ───────────────────────
                with for_(*from_array(v, volts)):

                    # Use the VoltageSequence run-time helper to ramp to voltages
                    # This can be used with any physical or virtual voltage in the gate set
                    seq.ramp_to_voltages(
                        voltages={quantum_dot.name: v},
                        duration=1000,
                        ramp_duration=1000,
                    )

                # Apply the compensation pulse
                seq.apply_compensation_pulse(max_voltage=0.01)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            # This example doesn't save the I/Q per sensor per QD

            # for i in range(len(quantum_dots)):
            # Each save() above is one voltage point.
            # .buffer(len(voltages)) : group points along the voltage axis
            # .average() : group points along the repetitions axis
            # Result : 1D trace I(voltages), Q(voltages) per quantum dot
            # I_st[i].buffer(len(volts)).average().save(f"I_{qd_name}_{i}")
            # Q_st[i].buffer(len(volts)).average().save(f"Q_{qd_name}_{i}")


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


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
