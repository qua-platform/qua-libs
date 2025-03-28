from typing import Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from qm import SimulationConfig, QuantumMachinesManager, Program
from qm.results.simulator_samples import SimulatorSamples
from qm.waveform_report import WaveformReport

from quam_experiments.experiments.time_of_flight.parameters import Parameters


def simulate_and_plot(
    qmm: QuantumMachinesManager,
    config: dict,
    program: Program,
    node_parameters: Parameters,
) -> Tuple[SimulatorSamples, Figure, Union[WaveformReport, None]]:
    """
    Simulates a QUA program and plots the simulated samples.
    Also generates the corresponding waveform report if applicable.

    Parameters:
    qmm (QuantumMachinesManager): The Quantum Machines Manager instance.
    config (dict): The configuration dictionary for the OPX.
    program (Program): The QUA program to be simulated.
    node_parameters (Parameters): Parameters for the node, including ``simulation_duration_ns`` and ``use_waveform_report``.

    Returns:
    Tuple[SimulatorSamples, Figure, Dict]: A tuple containing the simulated samples, the figure of the plotted samples,
                                           and the waveform report if applicable.
    """

    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node_parameters.simulation_duration_ns // 4)

    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, program, simulation_config)

    # Plot the simulated samples
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)

    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()

    if node_parameters.use_waveform_report:
        wf_report = job.get_simulated_waveform_report()
        # todo: make save_path use node's storage location
        # todo can we serialize the full report, or at least the plotly figure?
        wf_report.create_plot(samples, plot=True, save_path="./")
        return samples, fig, wf_report

    return samples, fig, None
