"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate frequencies dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly from the node parameters.

The data is post-processed to determine the qubit resonance frequency and the width of the peak.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the desired working point, independent, joint or arbitrary, in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    - Update the qubit frequency in the state, as well as the expected x180 amplitude and IQ rotation angle.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.1
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 150
    frequency_step_in_mhz: float = 0.25
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    target_peak_width: Optional[float] = 2e6
    arbitrary_flux_bias: Optional[float] = None
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="03a_Qubit_Spectroscopy", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
operation = node.parameters.operation  # The qubit operation to play
n_avg = 25  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
operation_len = node.parameters.operation_len_in_ns
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
flux_point = node.parameters.flux_point_joint_or_independent
qubit_freqs = {q.name: q.xy.RF_frequency for q in qubits}  # for opx

# Set the qubit frequency for a given flux point
if node.parameters.arbitrary_flux_bias is not None:
    arb_flux_bias_offset = {q.name: node.parameters.arbitrary_flux_bias for q in qubits}
    detunings = {q.name: q.freq_vs_flux_01_quad_term * arb_flux_bias_offset[q.name] ** 2 for q in qubits}
elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
    detunings = {
        q.name: 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name] for q in qubits
    }
    arb_flux_bias_offset = {q.name: np.sqrt(detunings[q.name] / q.freq_vs_flux_01_quad_term) for q in qubits}

else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}


target_peak_width = node.parameters.target_peak_width
if target_peak_width is None:
    target_peak_width = (
        3e6  # the desired width of the response to the saturation pulse (including saturation amp), in Hz
    )

with program() as qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, _, _ = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency
    n = [declare(int) for _ in range(num_qubits)]
    n_st = [declare_stream() for _ in range(num_qubits)]
    for i, qubit in enumerate(qubits):
        with for_(n[i], 0, n[i] < n_avg+i, n[i] + 1):
            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                # Play the saturation pulse
                qubit.xy.wait(qubit.z.settle_time * u.ns)
                qubit.xy.play(
                    operation,
                    amplitude_scale=operation_amp,
                )
                # Wait for the qubit to decay to the ground state
                qubit.resonator.wait(1000*machine.depletion_time * u.ns)
            # save data
            save(n[i], n_st[i])


        align()

    with stream_processing():
        # n_st.save_all("n")
        for i in range(num_qubits):
            n_st[i].save_all(f"n{i+1}")



with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(qubit_spec)
    #
    # results = fetching_tool(job, ["n"], mode="live")
    # while results.is_processing():
    #     # Fetch results
    #     n = results.fetch_all()[0]
    #     print(len(n))
    #     # Progress bar
    #     progress_counter(n[-1], n_avg, start_time=results.start_time)

    for i in range(num_qubits):
        print(f"qubit {i+1}\n")
        print(job.result_handles.get(f"n{1}").fetch_all())
        print(job.result_handles.get(f"n{2}").fetch_all())
        print(job.result_handles.get(f"n{3}").fetch_all())
        # job.result_handles.get(f"n{i+1}").wait_for_values(1)
        n = job.result_handles.get(f"n{i+1}").fetch_all()
        while n[-1]["value"] < n_avg-1+i:
            n = job.result_handles.get(f"n{i+1}").fetch_all()
            # progress_counter(n, n_avg)
            print(n)