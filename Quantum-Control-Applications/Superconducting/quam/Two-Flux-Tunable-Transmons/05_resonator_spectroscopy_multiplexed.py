"""
        RESONATOR SPECTROSCOPY MULTIPLEXED
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for both resonators simultaneously.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout frequency in the state.

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the readout pulse amplitude and duration in the state.
    - Specify the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, labeled as f_res and f_opt, in the state for both resonators.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig

from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from components import QuAM
from macros import node_save

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("state.json")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
rr1 = machine.active_qubits[0].resonator
rr2 = machine.active_qubits[1].resonator

###################
# The QUA program #
###################

n_avg = 100  # The number of averages
# The frequency sweep around the resonator resonance frequency f_opt
dfs = np.arange(-4e6, +4e6, 0.1e6)
# You can adjust the IF frequency here to manually adjust the resonator frequencies instead of updating the state
# rr1.intermediate_frequency = -50 * u.MHz
# rr2.intermediate_frequency = 50 * u.MHz


with program() as multi_res_spec:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I = [declare(fixed) for _ in range(2)]
    Q = [declare(fixed) for _ in range(2)]
    I_st = [declare_stream() for _ in range(2)]
    Q_st = [declare_stream() for _ in range(2)]
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # wait for the resonators to deplete
            wait(machine.get_depletion_time * u.ns, rr1.name, rr2.name)

            # resonator 1
            update_frequency(rr1.name, df + rr1.intermediate_frequency)
            rr1.measure("readout", qua_vars=(I[0], Q[0]))
            save(I[0], I_st[0])
            save(Q[0], Q_st[0])

            # rr2.align(rr1.name)  # Uncomment to measure sequentially
            # resonator 2
            update_frequency(rr2.name, df + rr2.intermediate_frequency)
            rr2.measure("readout", qua_vars=(I[1], Q[1]))
            save(I[1], I_st[1])
            save(Q[1], Q_st[1])

    with stream_processing():
        # resonator 1
        I_st[0].buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dfs)).average().save("Q2")

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, multi_res_spec, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_res_spec)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["I1", "Q1", "I2", "Q2"], mode="live")
    # Prepare the figures for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Live plotting
    while results.is_processing():
        # Fetch results
        I1, Q1, I2, Q2 = results.fetch_all()
        # Data analysis
        s1 = u.demod2volts(I1 + 1j * Q1, rr1.operations["readout"].length)
        s2 = u.demod2volts(I2 + 1j * Q2, rr2.operations["readout"].length)
        # Plot
        plt.subplot(221)
        plt.suptitle("Multiplexed resonator spectroscopy")
        plt.cla()
        plt.plot(rr1.intermediate_frequency / u.MHz + dfs / u.MHz, np.abs(s1), ".")
        plt.title(f"{rr1.name}")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.subplot(222)
        plt.cla()
        plt.plot(rr2.intermediate_frequency / u.MHz + dfs / u.MHz, np.abs(s2), ".")
        plt.title(f"{rr2.name}")
        plt.subplot(223)
        plt.cla()
        plt.plot(rr1.intermediate_frequency / u.MHz + dfs / u.MHz, signal.detrend(np.unwrap(np.angle(s1))), ".")
        plt.ylabel("Phase [rad]")
        plt.xlabel("Readout frequency [MHz]")
        plt.subplot(224)
        plt.cla()
        plt.plot(rr2.intermediate_frequency / u.MHz + dfs / u.MHz, signal.detrend(np.unwrap(np.angle(s2))), ".")
        plt.xlabel("Readout frequency [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {
        f"{rr1.name}_frequencies": rr1.intermediate_frequency + dfs,
        f"{rr1.name}_R": np.abs(s1),
        f"{rr1.name}_phase": signal.detrend(np.unwrap(np.angle(s1))),
        f"{rr2.name}_frequencies": rr2.intermediate_frequency + dfs,
        f"{rr2.name}_R": np.abs(s2),
        f"{rr2.name}_phase": signal.detrend(np.unwrap(np.angle(s2))),
        f"figure_raw": fig,
    }

    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        fig_fit = plt.figure()
        plt.suptitle("Multiplexed resonator spectroscopy")
        plt.subplot(121)
        res_1 = fit.reflection_resonator_spectroscopy((rr1.intermediate_frequency + dfs) / u.MHz, np.abs(s1), plot=True)
        plt.legend((f"f = {res_1['f'][0]:.3f} MHz",))
        plt.xlabel(f"{rr1.name} IF [MHz]")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.title(f"{rr1.name}")
        rr1.intermediate_frequency = int(res_1["f"][0] * u.MHz)
        data[f"{rr1.name}"] = {"resonator_frequency": int(rr1.intermediate_frequency), "successful_fit": True}
        plt.subplot(122)
        res_2 = fit.reflection_resonator_spectroscopy((rr2.intermediate_frequency + dfs) / u.MHz, np.abs(s2), plot=True)
        plt.legend((f"f = {res_2['f'][0]:.3f} MHz",))
        plt.xlabel(f"{rr2.name} IF [MHz]")
        plt.title(f"{rr2.name}")
        plt.tight_layout()
        rr2.intermediate_frequency = int(res_2["f"][0] * u.MHz)
        data[f"{rr2.name}"] = {"resonator_frequency": int(rr2.intermediate_frequency), "successful_fit": True}
        data["figure_fit"] = fig_fit
    except (Exception,):
        data["successful_fit"] = False
        pass

    # Save data from the node
    node_save("resonator_spectroscopy_multiplexed", data, machine)
