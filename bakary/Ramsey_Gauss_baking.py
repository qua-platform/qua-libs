from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from bakary import *
import matplotlib.pyplot as plt

import numpy as np
from importlib import reload

from Configuration_Maker_RamseyGauss import *

from time import sleep
from matplotlib import pyplot as plt


Resonator_TOF = 332
Readout_pulse_length = 100
Fastload_length = 60
Drive_pulse_length = 200

FreqCW = 6.6069 #meas freq
freq = 9.283 #drive freq

Tpihalf = 16
wait_time_cc = 100
npts = 64

amplitude_pihalf = 0.1
angle = np.pi / 4 + 0.65
dephasing0 = 0 #phase at the origin of the 2nd Tpihalf gauss pulse
#dephasingStep = np.pi/7 #beware, pulses with weird shapes for some values of dephasingStep
dephasingStep = 0
Drive_gauss_pulse_length = Tpihalf
Drive_Amp = 0.1
gauss_drive_amp = 0.1
gauss_drive_mu = 0
gauss_drive_sigma = Drive_gauss_pulse_length/6
Inverse_Readout_pulse_length = 1e9 / Readout_pulse_length
drive_cc = int(Tpihalf/4) + 4 #12cc = 48ns for Tpihalf=32


configRamseyGauss = configMakerRamseyGauss(Resonator_TOF = Resonator_TOF, Readout_pulse_length = Readout_pulse_length,
                                      Load_pulse_length = Fastload_length, Resonator_freq = FreqCW*1e9,
                                      Drive_freq = freq*1e9, Tpihalf = Tpihalf)

dmax = int(npts/4)
period_IF = 31.25e6

baking_list = [] #Stores the baking objects
for i in range(16):
    with baking(configRamseyGauss, padding_method="left") as b:
        init_delay = 19   # Put initial delay to ensure synchronicity between
        # the end of the pulse and the trigger of the resonator

        # Create and add gaussian sample to be added for Ramsey experiment
        gauss_drive_I = gauss(gauss_drive_amp, gauss_drive_mu, gauss_drive_sigma, Drive_gauss_pulse_length)
        gauss_drive_Q = [0] * Drive_gauss_pulse_length
        b.add_Op("gauss_drive", "Drive", [gauss_drive_I, gauss_drive_Q])


        b.frame_rotation(dephasingStep, 'Drive')
        b.wait(init_delay, 'Drive')  # This is to compensate for the extra delay the Resonator is experiencing.

        # Play uploads the sample in the original config file
        b.play("gauss_drive", 'Drive', amp=1)  # duration Tpihalf+16
        b.play_at('gauss_drive', 'Drive', init_delay-i)  # duration Tpihalf #Add error for negative values

    # Append the baking object in the list to call it from the QUA program
    baking_list.append(b)

# You can retrieve and see the pulse you built for each baking object by modifying
# index of the waveform (replace the "15" below by a number between 0 and 15)
baked_pulse = configRamseyGauss["waveforms"]["Drive_baked_wf_I_15"]["samples"]
t = np.arange(0, len(baked_pulse), 1)
# plt.plot(t, baked_pulse)

with program() as RamseyGauss:  # to measure Rabi flops every 1ns starting from 0ns

    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)
    dephasing = declare(fixed)
    assign(dephasing, dephasing0)

    d = declare(int)
    pw = declare(int)

    I_stream = declare_stream()
    Q_stream = declare_stream()
    param_stream = declare_stream()
    phase_stream = declare_stream()

    frame_rotation(angle, 'Resonator')
    frame_rotation(dephasing, 'Drive')
    with infinite_loop_():
        with for_(d, 0, d < dmax, d + 4):
            for j in range(16):
                align("Drive", "Resonator")
                wait(8, "Drive", "Resonator")  # Add wait time to deal with computational overhead
                wait(10, "Drive")
                reset_phase("Drive")
                baking_list[j].run()  # Run the baked waveform associated to baking object j

                wait(drive_cc + d, 'Resonator')
                reset_phase('Resonator')
                play("chargecav", "Resonator")  # to charge the cavity
                measure("readout", "Resonator", None, demod.full("integW_cos", I1, "out1"),
                        demod.full("integW_sin", Q1, "out1"),
                        demod.full("integW_cos", I2, "out2"),
                        demod.full("integW_sin", Q2, "out2"))
                assign(I, I1 + Q2)  # summing over all the items in the vectors before assigning to the final I and Q variables
                assign(Q, I2 - Q1)
                assign(dephasing, dephasing + dephasingStep)
                assign(pw, 4 * d + j)  # delay in ns between the 2 pihalf gauss pulses
                save(I, I_stream)
                save(Q, Q_stream)
                save(pw, param_stream)
                save(dephasing, phase_stream)  # dephasing of the 2nd gauss pulse wrt 1st one
        reset_frame('Drive', 'Drive2')

    with stream_processing():
        I_stream.buffer(1000, npts).save('Iall')
        Q_stream.buffer(1000, npts).save('Qall')
        I_stream.buffer(npts).average().save('I')
        Q_stream.buffer(npts).average().save('Q')
        param_stream.buffer(npts).average().save('param')
        phase_stream.buffer(npts).average().save('dephasing')


simulate = True


qmm = QuantumMachinesManager("3.122.60.129")
qm = qmm.open_qm(configRamseyGauss)

if simulate :
    job = qm.simulate(RamseyGauss,
                      SimulationConfig(32 * (wait_time_cc + Readout_pulse_length + Fastload_length + drive_cc)))
    samps = job.get_simulated_samples()

    an1 = samps.con1.analog['1'].tolist()
    an3 = samps.con1.analog['3'].tolist()
    dig1 = samps.con1.digital['1']
    dig3 = samps.con1.digital['3']
    plt.figure()
    plt.plot(an1)
    plt.plot(an3)
    # plt.axis(xmin = 16500, xmax = 20000, ymin=-0.011, ymax=0.011)

    print('End prog')
    plt.show()
else:
    job = qm.execute(RamseyGauss)
    res = job.result_handles
    sleep(2)

# I = np.array(res.I.fetch_all()) * Inverse_Readout_pulse_length
# Q = np.array(res.Q.fetch_all()) * Inverse_Readout_pulse_length
# time = np.array(res.param.fetch_all())  # in ns

"""
plt.ion()
fig, ax = plt.subplots(2,1)
I_plot, = ax[0].plot(dur, I_vs_dur)
Q_plot, = ax[1].plot(dur, Q_vs_dur)
ax[0].grid()
ax[1].grid()
plt.show()

ax[0].set_title("Rabi 1ns", fontsize=18)
ax[1].set_xlabel("drive duration (ns)", fontsize=14)
ax[0].set_ylabel("I", fontsize=14)
ax[1].set_ylabel("Q", fontsize=14)
while True:
    print("tic")
    I_vs_dur = np.array(res.I.fetch_all()) * Inverse_Readout_pulse_length
    Q_vs_dur = np.array(res.Q.fetch_all()) * Inverse_Readout_pulse_length
    dur = np.array(res.param.fetch_all())  # in ns
    I_plot.set_ydata(I_vs_dur)
    Q_plot.set_ydata(Q_vs_dur)
    fig.canvas.draw()
    fig.canvas.flush_events()
    sleep(2)
"""