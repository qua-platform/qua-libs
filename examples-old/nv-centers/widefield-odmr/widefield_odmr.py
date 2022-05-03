"""
widefield_odmr.py: NV center ODMR measurement with camera triggering
Author: Gal Winer - Quantum Machines
Created: 13/12/2020
Created on QUA version: 0.6.393
"""

import matplotlib.pyplot as plt
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *

from camera_mock_lib import *
from configuration import *

QMm = QuantumMachinesManager()
QM1 = QMm.open_qm(config)

##############
# parameters #
##############
N = 101
f_start = int(-100e6)
f_end = int(100e6)
f_step = int((f_end - f_start) / N)

totalTime = 10e6  # 10 ms
fastReadout = 10e3  # 10 micro seconds
normalReadout = 100e3  # should be 10ms, but set at 100us for clarity
nAverages = int(totalTime / fastReadout)

simulate = True
case = 1  # 1 for faster CW, 2 for normal CW, 3 for pulsed

###########
# CW ODMR #
###########
# Readout is being done for 10ms with MW and 10ms without MW.

with program() as CW_ODMR:
    freq = declare(int)

    with for_(freq, f_start, freq <= f_end, freq + f_step):
        update_frequency("NV", freq)

        # Meas with MW
        play("init", "AOM", duration=normalReadout // 4)  # init
        play("CW", "NV", duration=normalReadout // 4)  # MW
        play("trigger", "camera", duration=normalReadout // 4)  # camera
        wait(1000 // 4, "camera")  # camera rearming (can be removed if camera will be configured
        # to take 200 frames from a single trigger)

        # Reference without MW
        align("AOM", "camera")
        play("init", "AOM", duration=normalReadout // 4)  # init
        play("trigger", "camera", duration=normalReadout // 4)  # camera
        wait(1000 // 4, "camera")  # camera rearming (can be removed if camera will be configured
        # to take 200 frames from a single trigger)

################
# fast readout #
################
# Phantom camera has 1MHz FPS
# can trigger for a 10us readout
# no need for reference

with program() as CW_ODMR_fast_readout:
    freq = declare(int)
    i = declare(int)

    with for_(i, 0, i < nAverages, i + 1):
        ###
        with for_(freq, f_start, freq <= f_end, freq + f_step):
            update_frequency("NV", freq)

            play("init", "AOM", duration=fastReadout // 4)  # init
            play("CW", "NV", duration=fastReadout // 4)  # MW
            play("trigger", "camera", duration=fastReadout * 200 // 4)
            wait(1000 // 4, "camera")

###############
# pulsed ODMR #
###############
# Phantom camera has 1MHz FPS
# Can trigger for a 10us readout and compare 1st 1us and last 1us.

with program() as PULSED_ODMR:
    freq = declare(int)
    play("init", "AOM")  # init
    wait(1000 // 4, "AOM")  # For clarity

    with for_(freq, f_start, freq <= f_end, freq + f_step):
        update_frequency("NV", freq)

        play("pi", "NV")  # MW
        align("NV", "AOM", "camera")  # Wait for pi pulse to end
        play("init", "AOM")  # readout
        play("trigger", "camera")  # camera
        wait(1000 // 4, "AOM")  # For camera rearming & clarity

if simulate:

    # Case 1
    job = QM1.simulate(CW_ODMR, SimulationConfig(60000))
    res = job.result_handles
    samples = job.get_simulated_samples()
    AOM = samples.con1.analog["3"] / 0.2 / 2 + 0.5
    MW_I = samples.con1.analog["1"] / 0.2 / 2 - 1.5
    MW_Q = samples.con1.analog["2"] / 0.2 / 2 - 1.5
    # MW = np.sqrt((samples.con1.analog['1']) ** 2 + (samples.con1.analog['2']) ** 2) / 0.2 - 2
    camera = samples.con1.digital["1"] - 4

    plt.figure()
    plt.plot(AOM)
    # plt.plot(MW)
    plt.plot(MW_I)
    plt.plot(MW_Q)
    plt.plot(camera)
    plt.yticks([-3.5, -1.5, 0.5], ["Camera", "MW", "AOM"], rotation="vertical", va="center")
    plt.xlabel("t [ns]")
    plt.title("CW ODMR")

    # Case 2
    job = QM1.simulate(CW_ODMR_fast_readout, SimulationConfig(6000))
    samples = job.get_simulated_samples()
    AOM = samples.con1.analog["3"] / 0.2 / 2 + 0.5
    MW_I = samples.con1.analog["1"] / 0.2 / 2 - 1.5
    MW_Q = samples.con1.analog["2"] / 0.2 / 2 - 1.5
    # MW = np.sqrt((samples.con1.analog['1'])**2+(samples.con1.analog['2'])**2)/0.2 - 2
    camera = samples.con1.digital["1"] - 4

    plt.figure()
    plt.plot(AOM)
    # plt.plot(MW)
    plt.plot(MW_I)
    plt.plot(MW_Q)
    plt.plot(camera)
    plt.yticks([-3.5, -1.5, 0.5], ["Camera", "MW", "AOM"], rotation="vertical", va="center")
    plt.xlabel("t [ns]")
    plt.title("CW ODMR - fast readout")

    # Case 3
    job = QM1.simulate(PULSED_ODMR, SimulationConfig(10000))
    samples = job.get_simulated_samples()
    AOM = samples.con1.analog["3"] / 0.2 / 2 + 0.5
    MW_I = samples.con1.analog["1"] / 0.2 / 2 - 1.5
    MW_Q = samples.con1.analog["2"] / 0.2 / 2 - 1.5
    # MW = np.sqrt((samples.con1.analog['1']) ** 2 + (samples.con1.analog['2']) ** 2) / 0.2 - 2
    camera = samples.con1.digital["1"] - 4

    plt.figure()
    plt.plot(AOM)
    # plt.plot(MW)
    plt.plot(MW_I)
    plt.plot(MW_Q)
    plt.plot(camera)
    plt.yticks([-3.5, -1.5, 0.5], ["Camera", "MW", "AOM"], rotation="vertical", va="center")
    plt.xlabel("t [ns]")
    plt.title("Pulsed ODMR")

    plt.show()
else:
    if case == 1:  # Fast
        cam = cam_mock()  # create the camera object
        cam.allocate_buffer(N * nAverages)  # allocate new space in camera
        cam.arm()  # trigger arm
        job = QM1.execute(CW_ODMR_fast_readout)

        result_array = cam.get_image()

    elif case == 2:  # Normal
        cam = cam_mock()  # create the camera object
        cam.allocate_buffer(2 * N)  # allocate new space in camera
        cam.arm()  # trigger arm
        job = QM1.execute(CW_ODMR)

        result_array = cam.get_image()

    elif case == 3:  # Pulsed
        cam = cam_mock()  # create the camera object
        cam.allocate_buffer(N * nAverages)  # allocate new space in camera
        cam.arm()  # trigger arm
        job = QM1.execute(PULSED_ODMR)

        result_array = cam.get_image()
