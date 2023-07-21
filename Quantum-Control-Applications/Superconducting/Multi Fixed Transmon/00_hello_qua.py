from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from quam import QuAM
from configuration import build_config, octave_config
import matplotlib.pyplot as plt
from qm.octave import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
with program() as hello_qua:
    with infinite_loop_():
        for i in range(9):
            play("cw", machine.qubits[i].name)
            align()
            play('readout', machine.resonators[i].name)
            align()


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulation = False

if simulation:
    simulation_config = SimulationConfig(duration=8000)
    job = qmm.simulate(config, hello_qua, simulation_config, flags=['auto-element-thread'])
    # job.get_simulated_samples().con1.plot()
    # job.get_simulated_samples().con2.plot()
    # plt.show()
    # samples = job.get_simulated_samples()
    # waveform_report = job.get_simulated_waveform_report().to_dict()
    # for i in range(18):
    #     print(i)
    #     print("-" * 40)
    #     print(waveform_report['analog_waveforms'][2*i]['element'])
    #     print(waveform_report['analog_waveforms'][2*i]['timestamp'])
    #     print(waveform_report['analog_waveforms'][2*i]['pulser'])
    #     print(waveform_report['analog_waveforms'][2*i]['output_ports'])
    #     print("-" * 40)

else:
    qm = qmm.open_qm(config)
    # print(qm.get_config()['mixers'])
    job = qm.execute(hello_qua, flags=['auto-element-thread'])
