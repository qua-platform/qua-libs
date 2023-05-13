"""
PDH_PID_octave.py: This example code is an implementation
of Pound Drever Hall technique to find and lock on a resonator frequency

Version: 0.1

"""
## Imports
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from PDH_config_octave import *
import matplotlib.pyplot as plt
from qm import SimulationConfig # It is also possible to simulate the IQ signals
from qm.octave import *
import os
import csv

## Octave configuration
octave_config = QmOctaveConfig()
octave_config.add_device_info('octave1', qop_ip, octave_port)
octave_config.set_opx_octave_mapping([("con1", "octave1")])
octave_config.set_calibration_db(os.getcwd()) # The database can be placed anywhere you like
qmm = QuantumMachinesManager(host=qop_ip, port=opx_port, octave=octave_config)

qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_10)

qm = qmm.open_qm(config)

element="RR"
qm.octave.set_lo_frequency(element, rr_LO)
qm.octave.set_lo_source(element, OctaveLOSource.Internal)
qm.octave.set_rf_output_gain(element, -4)
qm.octave.set_rf_output_mode(element, RFOutputMode.on)

qm.octave.set_qua_element_octave_rf_in_port(element,"octave1", 1)
qm.octave.set_downconversion(element, lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.envelope, if_mode_q=IFMode.envelope) #Dmd2LO

qm.octave.calibrate_element(element, [(rr_LO, rr_IF)])

qm = qmm.open_qm(config)



## PDH PID code
# Output channels are 1 and 2 to go to the RF_OUT_1, all PDH parameters are in the config file
# It is best to calibrate the mixer first to have clean signal
# Always make sure that the carrier IF frequency is larger than the sideband as it should


f_init = int(50e6)
f_final = int(100e6)
df = int(10e3)
freq_array_size = int((f_final - f_init) / df)
N_averaging = 1

gain_P = 0.1
gain_I = 0.1
gain_D = 0.0
alpha = 0.1
PID_length = int(1e5)  # how many shots to run the PID loop
averaging_length_for_plotting = int(1)  # averaging every N shots to not transfer the whole vector of size PID_length
initial_frequency = int(-41.737e6)
error_scaling = -1e6

with program() as PDH_PID:
    adc_st = declare_stream(adc_trace=True)
    pound_signal = declare(fixed)
    pound_signal_st = declare_stream()
    f = declare(int, value=initial_frequency)  # initial resonator frequency
    scan_freq = declare(int)
    f_st = declare_stream()
    n = declare(int)
    ind = declare(int)
    error = declare(int)

    error_st = declare_stream()
    old_error = declare(int, value=0)
    integrator_error = declare(int, value=0)
    derivative_error = declare(int)

    # get raw ADC data first
    update_frequency('RR', f)
    reset_phase('RR')
    reset_phase('Pound_demod')
    play('pound_pulse', 'RR')
    play('pound_pulse', 'RR')
    align()
    play('pound_pulse', 'RR')
    measure('pound_demod_pulse', 'Pound_demod', adc_st, demod.full('integ_pound', pound_signal, 'out1'))
    play('pound_pulse', 'RR')

    # outside loop is for averaging many spectra
    with for_(n, 0, n < N_averaging, n + 1):
        # loop over the frequency
        # with for_(f, f_init, f<f_final, f+df ):

        with for_(ind, 0, ind < PID_length, ind + 1):
            update_frequency('RR', f)
            save(f, f_st)
            reset_phase('RR')
            reset_phase('Pound_demod')
            play('pound_pulse', 'RR')
            play('pound_pulse', 'RR')
            align()
            play('pound_pulse', 'RR')
            measure('pound_demod_pulse', 'Pound_demod', None, demod.full('integ_pound', pound_signal, 'out1'))
            play('pound_pulse', 'RR')
            save(pound_signal, pound_signal_st)
            assign(error, Cast.mul_int_by_fixed(error_scaling, pound_signal))

            assign(integrator_error, Cast.mul_int_by_fixed(integrator_error, 1 - alpha) +
                   Cast.mul_int_by_fixed(error, alpha))
            assign(derivative_error, old_error - error)
            assign(f, f + Cast.mul_int_by_fixed(error, gain_P) +
                   Cast.mul_int_by_fixed(integrator_error, gain_I) +
                   Cast.mul_int_by_fixed(derivative_error, gain_D))

            assign(old_error, error)
            save(error, 'error_test')
            save(error, error_st)

        ind2 = declare(int)
        scan_steps = 100
        scan_step_size = 100
        averaging = 1024

        error_calib = declare(int, value=np.zeros(scan_steps, dtype=int).tolist())
        scan_freq_list = declare(int, value=np.zeros(scan_steps, dtype=int).tolist())
        with for_(ind2, 0, ind2 < averaging, ind2 + 1):

            with for_(ind, 0, ind < scan_steps, ind + 1):
                assign(scan_freq, f - scan_step_size * scan_steps / 2 + scan_step_size * ind)
                update_frequency('RR', scan_freq)
                assign(scan_freq_list[ind], scan_freq)
                #assign(error, 0)

                reset_phase('RR')
                reset_phase('Pound_demod')
                play('pound_pulse', 'RR')
                align()
                play('pound_pulse', 'RR')
                measure('pound_demod_pulse', 'Pound_demod', None, demod.full('integ_pound', pound_signal, 'out1'))
                play('pound_pulse', 'RR')
                assign(error_calib[ind], error_calib[ind] + Cast.mul_int_by_fixed(error_scaling, pound_signal))

                #assign(error, error >> int(np.log2(averaging)))
                #save(error, 'scan_PDH_error')
        with for_(ind, 0, ind < scan_steps, ind+1):
            assign(error_calib[ind], error_calib[ind] >> int(np.log2(averaging)))
            save(error_calib[ind], 'scan_PDH_error')
            save(scan_freq_list[ind], 'scan_freq')
        # with for_(ind, 0, ind < scan_steps, ind + 1):
        #     assign(scan_freq, f - scan_step_size * scan_steps / 2 + scan_step_size * ind)
        #     update_frequency('RR', scan_freq)
        #     save(scan_freq, 'scan_freq')
        #     assign(error, 0)
        #
        #     with for_(ind2, 0, ind2 < averaging, ind2 + 1):
        #         reset_phase('RR')
        #         reset_phase('Pound_demod')
        #         play('pound_pulse', 'RR')
        #         align()
        #         play('pound_pulse', 'RR')
        #         measure('pound_demod_pulse', 'Pound_demod', None, demod.full('integ_pound', pound_signal, 'out1'))
        #         play('pound_pulse', 'RR')
        #         assign(error, error + Cast.mul_int_by_fixed(error_scaling, pound_signal))
        #
        #     assign(error, error >> int(np.log2(averaging)))
        #     save(error, 'scan_PDH_error')


            # # one last time to get raw ADC data
            # update_frequency('RR', f+1000000)
            # reset_phase('RR')
            # reset_phase('Pound_demod')
            # play('pound_pulse', 'RR')
            # align()
            # play('pound_pulse', 'RR')
            # measure('pound_demod_pulse', 'Pound_demod', adc_st, demod.full('integ_pound', pound_signal, 'out1'))
            # play('pound_pulse', 'RR')

    # stream processing handling
    with stream_processing():
        adc_st.input1().save('adc1')  # see the raw data
        # pound_signal_st.buffer(100).average().save('pound_signal') # buffer it into vectors the length of the frequency scan and average over the averaging loop
        error_st.buffer(averaging_length_for_plotting).average().save_all('error')
        pound_signal_st.buffer(averaging_length_for_plotting).average().save(
            'pound_signal')  # buffer it into vectors the length of the frequency scan and average over the averaging loop
        f_st.buffer(averaging_length_for_plotting).average().save_all('f')

job = qm.execute(PDH_PID)
job.result_handles.wait_for_all_values()
res = job.result_handles
res.wait_for_all_values()
adc1 = res.get("adc1").fetch_all()
pound_signal = res.get('pound_signal').fetch_all()
error = res.get('error').fetch_all()
scan_PDH_error = res.get('scan_PDH_error').fetch_all()['value'].tolist()
scan_freq = res.get('scan_freq').fetch_all()['value'].tolist()
f = res.get('f').fetch_all()
freq = [x / 1e6 for x in np.linspace(f_init, f_final, int((f_final - f_init) / df))]

error_flat = [er[0] for er in error['value']]
f_flat = [ff[0] for ff in f['value']]
mean_f = np.mean(f_flat)
resonator_freq = rr_LO + mean_f
frac_f = [(freq - mean_f) / (resonator_freq) for freq in f_flat]

from scipy.optimize import curve_fit


def func(x, a, b):
    return [a * xind + b for xind in x]


popt, pcov = curve_fit(func, scan_freq, scan_PDH_error)
error_to_Hz = popt[0]

error_Hz = [(er * error_to_Hz) for er in error_flat]
error_frac = [(er * error_to_Hz) / resonator_freq for er in error_flat]

with open('error_frac_freq_withlock2.csv', 'w', newline='') as file_error_freq:
    writer = csv.writer(file_error_freq)
    for index in range(error_frac.__len__()):
        writer.writerow([error_frac[index]])

plt.figure(figsize=(16, 12))
plt.plot(error_Hz, linewidth=5, markersize=10)
plt.title('PDH error signal in Hz', fontsize=30)
plt.xlabel('Shot no.', fontsize=30)
plt.ylabel('PDH Error signal [Hz]', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

plt.figure(figsize=(16, 12))
plt.plot(scan_freq, scan_PDH_error, linewidth=5, markersize=10)
plt.title('PDH/freq slope fit: a=%5.3f, b=%5.3f' % tuple(popt), fontsize=30)
plt.xlabel('Resonator frequency', fontsize=30)
plt.ylabel('PDH error signal', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.plot(scan_freq, func(scan_freq, popt[0], popt[1]), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

plt.figure(figsize=(16, 12))
plt.plot(error, linewidth=5, markersize=10)
plt.title('PDH error signal', fontsize=30)
plt.xlabel('Shot no.', fontsize=30)
plt.ylabel('PDH Error signal', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

plt.figure(figsize=(16, 12))
plt.plot(f, linewidth=5, markersize=10)
plt.title('Resonator frequency', fontsize=30)
plt.xlabel('Shot no.', fontsize=30)
plt.ylabel('Resonator_frequency', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

# plt.figure(figsize=(16,12))
# plt.plot(pound_signal,linewidth=5, markersize=10)
# plt.title('PDH on resonance', fontsize=30)
# plt.xlabel('Shot no.', fontsize=30)
# plt.ylabel('PDH signal', fontsize=30)
# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 20)
# plt.grid()

# j=0
# error_flat = np.zeros(100000*1000)
# error_flat = error_flat.tolist()
# for index1 in range(100000):
#     for index2 in range(1000):
#         error_flat[j] = error[index1][0][index2]
#         j = j+1


#
plt.figure(figsize=(16, 12))
plt.plot(adc1)
plt.title('Raw acquired data (for the first shot)', fontsize=30)
plt.xlabel('Time [ns]', fontsize=30)
plt.ylabel('Raw ADC sample', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

# sampling rate
sr = 1e9
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

X = np.abs(np.fft.fft(adc1))
N = len(adc1)
n = np.arange(N)
T = N/sr
#freq = n/T

plt.figure(figsize=(16, 12))
freq = np.fft.fftfreq(len(adc1), d=1)
#fft = np.abs(np.fft.fft(adc1))
plt.plot(freq * 1e3, X, lw=5)
plt.title('FFT of raw acquired data (for the first shot)', fontsize=30)
plt.xlabel('Frequency [MHz]', fontsize=30)
plt.ylabel('FT of signal [V.s]', fontsize=30)
plt.yscale("log")
plt.grid()
## Simulate
# It is also possible simulate the signal to see that everything is phase coherent etc.. simulate for 20000 clock cycles or 80us
# job = qmm.simulate(config, PDH_PID, SimulationConfig(20000))
# samps = job.get_simulated_samples()
# RR_I = samps.con1.analog['1']
# RR_Q = samps.con1.analog['2']
#
# plt.figure(figsize=(16,12))
# plt.plot(RR_I)
# plt.title('Simulated PDH output signal', fontsize=30)
# plt.xlabel('Time [ns]', fontsize=30)
# plt.ylabel('Raw DAC signal', fontsize=30)
# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 20)
# plt.grid()
# you can also plot the other quadrature if you want:
# plt.figure(figsize=(10,8))
# plt.plot(RR_Q)

#
# with for_(ind, 0, ind < PID_length, ind + 1):
#     update_frequency('RR', f)
#     reset_phase('RR')
#     reset_phase('Pound_demod')
#     play('pound_pulse', 'RR')
#     measure('pound_demod_pulse', 'Pound_demod', None, demod.full('integ_pound', pound_signal, 'out1'))
#     save(pound_signal, pound_signal_st)
#     assign(error, Cast.mul_int_by_fixed(-1e8, pound_signal))
#     assign(integrator_error, Cast.mul_int_by_fixed(integrator_error, 1 - alpha) +
#            Cast.mul_int_by_fixed(error, alpha))
#     assign(derivative_error, old_error - error)
#     assign(f, f + Cast.mul_int_by_fixed(error, gain_P) +
#            Cast.mul_int_by_fixed(integrator_error, gain_I) +
#            Cast.mul_int_by_fixed(derivative_error, gain_D))
