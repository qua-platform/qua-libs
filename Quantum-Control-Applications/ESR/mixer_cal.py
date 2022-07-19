"""
A script used to calibrate the corrections for mixer imbalances
"""

from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager
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

qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=2000,
        simulation_interface=LoopbackInterface(([("con1", 3, "con1", 1), ("con1", 4, "con1", 2)]), latency=180),
    )
    job = qmm.simulate(config, mixer_cal, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

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
