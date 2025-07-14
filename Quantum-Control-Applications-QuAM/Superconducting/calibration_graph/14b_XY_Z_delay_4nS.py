# %%
"""
    XY-Z delay as describe in page 108 at https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf
"""
import warnings

from datetime import datetime, timezone, timedelta
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal

from scipy.optimize import curve_fit
from quam_libs.lib.save_utils import get_node_id, save_node

class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 100
    # z_pulse_amplitude: float = 0.1  # defines how much you want to detune the qubit in frequency
    delay_span: int = 50 # in clock cycles
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(
    name="14b_XY_Z_delay_4nS",
    parameters=Parameters()
)
node_id = get_node_id(node)

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
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

relative_time = np.arange(-node.parameters.delay_span + 4, node.parameters.delay_span,
                          1)  # x-axis for plotting - Must be in clock cycles.

n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

# %%

with program() as xy_z_delay_calibration:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    t = declare(int)  # QUA variable for the flux pulse segment index

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])
    
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            for init_state in ["x180", "I"]:
                with for_each_(t, relative_time):
                    qubit.align()

                    # qubit reset
                    if reset_type == "active":
                        active_reset(qubit)
                    else:
                        qubit.resonator.wait(machine.thermalization_time * u.ns)
                        qubit.align()

                    qubit.align()
                    with strict_timing_():
                        if init_state == "x180":
                            qubit.xy.play("x180")
                        elif init_state == "I":
                            qubit.xy.wait(qubit.xy.operations['x180'].length // 4)
                        qubit.z.wait(qubit.xy.operations['x180'].length // 4)
                        qubit.z.wait(node.parameters.delay_span)
                        qubit.z.play("const")
                        qubit.xy.wait(node.parameters.delay_span + t)
                        qubit.xy.play("x180")

                    qubit.align()
                    readout_state(qubit, state[i])
                    save(state[i], state_stream[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_stream[i].buffer(len(relative_time)).buffer(2).average().save(f"state{i + 1}")

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
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
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
ds = fetch_results_as_xarray(handles, qubits, {"relative_time": relative_time*4, "sequence": [0, 1]})
ds.relative_time.attrs['long_name'] = 'timing_delay'
ds.relative_time.attrs['units'] = 'nS'
node.results = {}
node.results['ds'] = ds

# %%

# find where the valus of ds.state.sel(sequence=0) is above 0.5 and return the mean of the relative_time
delays_0 = (ds.state.sel(sequence=0).where(ds.state.sel(sequence=0) > 0.5) / ds.state.sel(sequence=0).where(ds.state.sel(sequence=0) > 0.5) * ds.relative_time).mean(dim="relative_time")
delays_1 = (ds.state.sel(sequence=1).where(ds.state.sel(sequence=1) < 0.5) / ds.state.sel(sequence=1).where(ds.state.sel(sequence=1) < 0.5) * ds.relative_time).mean(dim="relative_time")
delays = (delays_0 + delays_1) / 2
# %%

# %%
grid = QubitGrid(ds, [q.grid_location for q in qubits])


flux_delays = []
for ax, qubit in grid_iter(grid):
    # x = ds.relative_time
    qubit = qubit['qubit']
    ds.state.sel(qubit=qubit).plot(hue = "sequence", ax = ax)
    ax.axvline(delays_0.sel(qubit=qubit), color="C0", linestyle="--", label="|0>")
    ax.axvline(delays_1.sel(qubit=qubit), color="C1", linestyle="--", label="|1>")
    
    ax.set_xlabel("Relative Time")
    ax.set_ylabel("State")
    ax.set_title(f"{qubit}")

    ax.legend()

grid.fig.suptitle(f'XY Z Delay Fitting \n {date_time} GMT+3 #{node_id} \n reset type = {node.parameters.reset_type_thermal_or_active}')
plt.tight_layout()
plt.show()

node.results['figure'] = grid.fig

# %%

with node.record_state_updates():
    for i, q in enumerate(qubits):
        if delays.sel(qubit=q.name) is not None:
            q.z.opx_output.delay -= int(np.round(delays.sel(qubit=q.name).values))

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
save_node(node)

# %%