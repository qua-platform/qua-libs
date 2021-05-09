from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from bakary import *

from RamseyGauss_configuration import *

from time import sleep
from matplotlib import pyplot as plt

dephasingStep = 0
number_of_pulses = 32

baking_list = []  # Stores the baking objects
for i in range(number_of_pulses): # Create 16 different baked sequences
    with baking(config, padding_method="left") as b:
        init_delay = number_of_pulses  # Put initial delay to ensure that all of the pulses will have the same length

        b.frame_rotation(dephasingStep, 'Drive')
        b.wait(init_delay, 'Drive')  # This is to compensate for the extra delay the Resonator is experiencing.

        # Play uploads the sample in the original config file (here we use an existing pulse in the config)
        b.play("gauss_drive", 'Drive', amp=1)  # duration Tpihalf+16
        b.play_at('gauss_drive', 'Drive', init_delay - i)  # duration Tpihalf

    # Append the baking object in the list to call it from the QUA program
    baking_list.append(b)
# You can retrieve and see the pulse you built for each baking object by modifying
# index of the waveform
plt.figure()
for i in range(number_of_pulses):
    baked_pulse = config["waveforms"][f"Drive_baked_wf_I_{i}"]["samples"]
    t = np.arange(0, len(baked_pulse), 1)
    plt.plot(t, baked_pulse)
