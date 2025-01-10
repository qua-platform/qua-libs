from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from qm import SimulationConfig, QuantumMachinesManager, Program

from quam_libs.experiments.time_of_flight.parameters import Parameters


def simulate_and_plot(qmm: QuantumMachinesManager, config: dict, program: Program, node_parameters: Parameters) -> Figure:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node_parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns

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

    return fig, samples
