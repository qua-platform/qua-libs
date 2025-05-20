# %%
"""
    XY-Z delay as describe in page 108 at https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf
"""
import warnings

from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal

from scipy.optimize import curve_fit

from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.save_utils import save_node

class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 200
    zeros_before_after_pulse: int = 36  # Beginning/End of the flux pulse (before we put zeros to see the rising time)
    z_pulse_amplitude: float = 0.1  # defines how much you want to detune the qubit in frequency
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "active"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(
    name="14_XY_Z_delay",
    parameters=Parameters()
)

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
import matplotlib.pyplot as plt
from qualang_tools.bakery import baking
import numpy as np

from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

tracked_qubits = []

for q in qubits:
    with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
        q.xy.operations["x180"].alpha = -1.0
        tracked_qubits.append(q)

config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

num_qubits = len(qubits)

# %%
###################
# The QUA program #
###################

n_avg = node.parameters.num_averages  # Number of averages
total_zeros = 2 * node.parameters.zeros_before_after_pulse

flux_waveform_list = {}

for qubit in qubits:
    flux_waveform_list[qubit.xy.name] = [node.parameters.z_pulse_amplitude] * qubit.xy.operations['x180'].length


def baked_waveform(waveform, qb):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration

    for i in range(0, 2 * node.parameters.zeros_before_after_pulse):
        with baking(config, padding_method="none") as b:
            wf = [0.0] * i + waveform + [0.0] * (2 * node.parameters.zeros_before_after_pulse - i)
            I_wf = [0.0] * (node.parameters.zeros_before_after_pulse) + \
                   config['waveforms'][qb.xy.name + '.x180_DragCosine.wf.I']['samples'] + [0.0] * (
                       node.parameters.zeros_before_after_pulse)
            Q_wf = [0.0] * (node.parameters.zeros_before_after_pulse) + \
                   config['waveforms'][qb.xy.name + '.x180_DragCosine.wf.Q']['samples'] + [0.0] * (
                       node.parameters.zeros_before_after_pulse)

            assert len(wf) == len(I_wf) == len(Q_wf), \
                f"Lengths of wf ({len(wf)}), I_wf ({len(I_wf)}), and Q_wf ({len(Q_wf)}) must be the same."

            b.add_op("flux_pulse", qb.z.name, wf)
            b.add_op("x180", qb.xy.name, [I_wf, Q_wf])

            b.play("flux_pulse", qb.z.name)
            b.play("x180", qb.xy.name)

        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)

    return pulse_segments


delay_segments = {}
# Baked flux pulse segments with 1ns resolution

for i, qubit in enumerate(qubits):
    delay_segments[qubit.xy.name] = baked_waveform(flux_waveform_list[qubit.xy.name], qubit)

relative_time = np.arange(-node.parameters.zeros_before_after_pulse, node.parameters.zeros_before_after_pulse,
                          1)  # x-axis for plotting - Must be in ns.
number_of_segments = 2 * node.parameters.zeros_before_after_pulse

n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

# %%

with program() as xy_z_delay_calibration:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    segment = declare(int)  # QUA variable for the flux pulse segment index
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            for init_state in ["x180", "I"]:
                with for_(segment, 0, segment < number_of_segments, segment + 1):
                    qubit.align()

                    # qubit reset
                    if reset_type == "active":
                        active_reset(qubit)
                    else:
                        qubit.resonator.wait(machine.thermalization_time * u.ns)
                        qubit.align()

                    qubit.align()

                    if init_state == "x180":
                        qubit.xy.play("x180")
                    elif init_state == "I":
                        qubit.xy.wait(qubit.xy.operations['x180'].length)

                    with switch_(segment):

                        for j in range(0, number_of_segments):
                            with case_(j):
                                delay_segments[qubit.xy.name][j].run()

                    qubit.align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_stream[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_stream[i].buffer(number_of_segments).buffer(2).average().save(f"state{i + 1}")

# %%

###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, xy_z_delay_calibration, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    quit()
else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(xy_z_delay_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # fetch results
            n = results.fetch_all()[0]
            # progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"relative_time": relative_time, "sequence": [0, 1]})

# %%

ds = ds.assign_coords({'relative_time': (['relative_time'], relative_time)})
ds.relative_time.attrs['long_name'] = 'timing_delay'
ds.relative_time.attrs['units'] = 'nS'
node.results = {}
node.results['ds'] = ds

# %%
# Define a smooth model for each line
def smooth_model(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d


# Define the first derivative of the sigmoid
def sigmoid_derivative(x, a, b, c, d):
    exp_term = np.exp(-b * (x - c))
    return (a * b * exp_term) / ((1 + exp_term) ** 2)


# Find the left boundary of the transition region
def find_transition_boundary(params, x_range, threshold=0.0001):
    a, b, c, d = params
    for x in x_range:
        if sigmoid_derivative(x, a, b, c, d) > threshold:
            return x
    return None  # If no boundary is found in the range
# %%
grid = QubitGrid(ds, [q.grid_location for q in qubits])


flux_delays = []
for ax, qubit in grid_iter(grid):
    x = ds.relative_time
    qubit = qubit['qubit']
    y1 = ds.state.sel(qubit=qubit).sel(sequence=0)
    y2 = ds.state.sel(qubit=qubit).sel(sequence=1)

    ax.plot(x, y1, label=r"$|0\rangle$", color="b", linewidth=2)
    ax.plot(x, y2, label=r"$|1\rangle$", color="r", linewidth=2)

    plt.figure(figsize=(8, 6))

    try:
        # Fit both lines to the smooth model
        bounds = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, 1])  # d between 0 and 1
        params1, _ = curve_fit(smooth_model, x, y1, p0=[1, 1, 5, 0], bounds=bounds, maxfev=10000)
        params2, _ = curve_fit(smooth_model, x, y2, p0=[1, -1, 5, 0], bounds=bounds, maxfev=10000)

        # Extended x-range for boundary search
        extended_x = np.linspace(x.min() - len(x), x.max() + len(x), 1000)

        crossing_point = find_transition_boundary(params2, extended_x, threshold=0.01)
        flux_delay = crossing_point - machine.qubits[qubit].xy.operations['x180'].length // 2
        flux_delays.append(flux_delay)

        ax.plot(extended_x, smooth_model(extended_x, *params1), color="blue", linestyle="--")
        ax.plot(extended_x, smooth_model(extended_x, *params2), color="red", linestyle="--")

        if flux_delay is not None:
            ax.axvline(crossing_point, color="black", linestyle="--", alpha=0.5)
            ax.axvline(flux_delay, color="green", linestyle="-", alpha=0.5, label="Delay")

        ax.set_xlabel("Relative Time")
        ax.set_ylabel("State")
        ax.set_title(f"{qubit}")

        margin = 2
        ax.set_xlim(flux_delay - margin, max(x) + margin)
        ax.legend()

    except Exception as e:
        warnings.warn(f"Fitting for qubit {qubit} failed with error: {e}")
        flux_delays.append(None)

grid.fig.suptitle('XY Z Delay Fitting')
plt.tight_layout()
plt.show()

node.results['figure'] = grid.fig

# %%
for qubit in tracked_qubits:
    qubit.revert_changes()

with node.record_state_updates():
    for i, q in enumerate(qubits):
        if flux_delays[i] is not None:
            q.z.opx_output.delay += round(flux_delays[i])

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
save_node(node)

# %%