# %% {Imports}
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import declare, declare_stream, fixed, for_, program, save, stream_processing

from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit

from qualibrate import NodeParameters, QualibrationNode

from configuration_quam_lf_fem_and_mw_fem import config, machine, state_path

u = unit(coerce_to_integer=True)


# %% {Parameters}
class Parameters(NodeParameters):
    simulate: bool = False
    simulation_duration_ns: int = 10_000
    n_avg: int = 100
    f_min_mhz: float = 30.0
    f_max_mhz: float = 70.0
    df_khz: float = 100.0
    update_state: bool = False


# %% {Node initialisation}
node = QualibrationNode[Parameters, None](
    name="04_resonator_spectroscopy_qualibrate",
    description="""
        RESONATOR SPECTROSCOPY
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout intermediate frequency on the QUAM resonator channel
(``machine.channels["resonator"].intermediate_frequency``).

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer / down-converter connected to the readout line.
    - Define the readout pulse amplitude and duration in the QUAM state.
    - Specify the expected resonator depletion time in the QUAM state.

Before proceeding to the next node:
    - Update the resonator intermediate frequency in the QUAM state.
""",
    parameters=Parameters(),
)


# %% {Create QUA program}
@node.run_action()
def create_qua_program(node: QualibrationNode[Parameters, None]):
    resonator = machine.channels["resonator"]

    n_avg = node.parameters.n_avg
    f_min = int(node.parameters.f_min_mhz * u.MHz)
    f_max = int(node.parameters.f_max_mhz * u.MHz)
    df = int(node.parameters.df_khz * u.kHz)
    frequencies = np.arange(f_min, f_max + 1, df)

    node.namespace["frequencies"] = frequencies
    node.namespace["readout_len"] = int(resonator.operations["readout"].length)
    res_out_port = resonator.opx_output.get_reference_value()
    node.namespace["resonator_LO"] = int(res_out_port.upconverter_frequency)

    with program() as resonator_spec:
        n = declare(int)
        f = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n_st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(f, frequencies)):
                resonator.update_frequency(f)
                resonator.measure("readout", qua_vars=(I, Q))
                resonator.wait(machine.depletion_time * u.ns)
                save(I, I_st)
                save(Q, Q_st)
            save(n, n_st)

        with stream_processing():
            I_st.buffer(len(frequencies)).average().save("I")
            Q_st.buffer(len(frequencies)).average().save("Q")
            n_st.save("iteration")

    node.namespace["qua_program"] = resonator_spec


# %% {Simulate}
@node.run_action(skip_if=not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, None]):
    qmm = QuantumMachinesManager(
        host=machine.qop_ip, port=machine.qop_port, cluster_name=machine.cluster_name
    )
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns)
    job = qmm.simulate(config, node.namespace["qua_program"], simulation_config)

    samples = job.get_simulated_samples()
    fig = plt.figure()
    samples.con1.plot()

    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))

    node.results["simulation"] = {
        "figure": fig,
        "samples": samples,
        "waveform_report": waveform_report.to_dict(),
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, None]):
    qmm = QuantumMachinesManager(
        host=machine.qop_ip, port=machine.qop_port, cluster_name=machine.cluster_name
    )
    qm = qmm.open_qm(config)
    job = qm.execute(node.namespace["qua_program"])

    frequencies = node.namespace["frequencies"]
    readout_len = node.namespace["readout_len"]
    resonator_LO = node.namespace["resonator_LO"]
    n_avg = node.parameters.n_avg

    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        I, Q, iteration = results.fetch_all()
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)
        phase = np.angle(S)
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        plt.suptitle(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
        ax1 = plt.subplot(211)
        plt.cla()
        plt.plot(frequencies / u.MHz, R, ".")
        plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.subplot(212, sharex=ax1)
        plt.cla()
        plt.plot(frequencies / u.MHz, sp_signal.detrend(np.unwrap(phase)), ".")
        plt.xlabel("Intermediate frequency [MHz]")
        plt.ylabel("Phase [rad]")
        plt.pause(0.1)
        plt.tight_layout()

    qm.close()

    node.results["raw_data"] = {
        "frequencies_hz": frequencies, "I": I, "Q": Q,
        "amplitude_v": R, "phase_rad": phase,
    }
    node.results["figures"] = {"live": fig}


# %% {Analyse}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, None]):
    raw = node.results.get("raw_data")
    if raw is None:
        return

    frequencies = raw["frequencies_hz"]
    R = raw["amplitude_v"]
    resonator_LO = node.namespace["resonator_LO"]

    fit_results = {"success": False}
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        fig_fit = plt.figure()
        res_spec_fit = fit.reflection_resonator_spectroscopy(frequencies / u.MHz, R, plot=True)
        plt.title(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
        plt.xlabel("Intermediate frequency [MHz]")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")

        fitted_if_mhz = float(res_spec_fit["f"][0])
        node.log(f"Resonator resonance frequency to update in the config: resonator_IF = {fitted_if_mhz:.6f} MHz")
        fit_results = {
            "success": True,
            "intermediate_frequency_mhz": fitted_if_mhz,
            "intermediate_frequency_hz": fitted_if_mhz * u.MHz,
        }
        node.results.setdefault("figures", {})["fit"] = fig_fit
    except Exception:
        pass

    node.results["fit_results"] = fit_results
    node.outcomes = {"resonator": "successful" if fit_results["success"] else "failed"}


# %% {Update state}
@node.run_action(skip_if=node.parameters.simulate or not node.parameters.update_state)
def update_state(node: QualibrationNode[Parameters, None]):
    fit_results = node.results.get("fit_results", {})
    if not fit_results.get("success"):
        return

    new_if_hz = int(fit_results["intermediate_frequency_hz"])
    with node.record_state_updates():
        machine.channels["resonator"].intermediate_frequency = new_if_hz
    machine.save(state_path)


# %% {Save}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, None]):
    node.save()
