"""
A script used to calibrate the corrections for mixer imbalances
"""

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from configuration import *


###################
# The QUA program #
###################
with program() as mixer_cal:
    with infinite_loop_():
        play("const", "ensemble")


################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulate_config = SimulationConfig(
        duration=2000,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, mixer_cal, simulate_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    qm = qmm.open_qm(config)

    job = qm.execute(mixer_cal)  # execute QUA program

    """
    When done, the halt command can be called and the offsets can be written directly into the config file.
    """
    # job.halt()
    """"
    These are the 2 commands used to correct for mixer imperfections. The first is used to set the DC of the I and Q
    channels to compensate for the LO leakage. Since this compensation depends on the I & Q powers, it is advised to
    run this step with no input power, so that there is no LO leakage while the pulses are not played.
    
    The 2nd command is used to correct for the phase and amplitude mismatches between the channels.
    
    The output of the IQ Mixer should be connected to a spectrum analyzer and values should be chosen as to minimize the
    unwanted peaks.
    
    If python can read the output of the spectrum analyzer, then this process can be automated and the correct values can
    found using an optimization method such as Nelder-Mead:
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
    """
    # qm.set_output_dc_offset_by_element('ensemble', ('I', 'Q'), (-0.001, 0.003))
    # qm.set_mixer_correction('mixer_ensemble', int(ensemble_IF), int(ensemble_LO), IQ_imbalance(0.015, 0.01))
