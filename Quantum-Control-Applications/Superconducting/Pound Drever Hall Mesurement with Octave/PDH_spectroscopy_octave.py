"""
PDH_spectroscopy_octave.py: This example code is an implementation
of Pound Drever Hall technique to find and lock on a resonator frequency

Version: 0.1

"""
## Imports
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from PDH_config_octave import *
import matplotlib.pyplot as plt
from qm import SimulationConfig # it is also possible to simulate the baseband signals
from qm.octave import *
import os


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
qm.octave.set_rf_output_gain(element, -10)
qm.octave.set_rf_output_mode(element, RFOutputMode.on)

qm.octave.set_qua_element_octave_rf_in_port(element,"octave1", 1)
qm.octave.set_downconversion(element, lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.envelope, if_mode_q=IFMode.envelope) #Dmd2LO

qm.octave.calibrate_element(element, [(rr_LO, rr_IF)])

qm = qmm.open_qm(config)


## PDH spectroscopy code
# Output channels are 1 and 2 to go to the RF_OUT_1, all PDH parameters are in the config file
# It is best to calibrate the mixer first to have clean signal
# Always make sure that the carrier IF frequency is larger than the sideband as it should


f_init = int(-70e6) # This is where the sweep starts in Hz
f_final = int(-10e6) # This is where the sweep ends in Hz
df = int(10e3) # Sweep resolution in Hz
freq_array_size = int((f_final-f_init)/df)
N_averaging = 1 # how many sweeps to run



with program() as PDH_spectroscopy:
    adc_st = declare_stream(adc_trace=True)
    pound_signal = declare(fixed)
    pound_signal_int = declare(int)
    pound_signal_st = declare_stream()
    f = declare(int)
    n = declare(int)

    # outside loop is for averaging many spectra
    with for_(n, 0, n<N_averaging, n+1):
        # loop over the frequency
        with for_(f, f_init, f<f_final, f+df ):
            # update the frequency
            update_frequency('RR', f)
            # reset the phase of the carrier oscillator so that we always start with the same phase
            reset_phase('RR')
            # reset the phase of the demodulator element so that we always demodulate with the same phase
            reset_phase('Pound_demod')
            # play one shot of the pound pulse to the resonator ('RR') to reach steady state for the readout
            # this can be made much shorter by creating a shorter 'pound_pulse' no need to wait 50us every time but it's conservative
            # try removing this and see what you get
            for python_index in range(2):
                play('pound_pulse', 'RR')

            # align all the elements so that they all start after the first play() command
            align()
            # This is the actual pound pulse we will demodulate
            play('pound_pulse', 'RR')
            # measure the pound signal with a separate element which oscillates at the sideband frequency
            # and save the result into the pound_signal variable. Save also the raw acquired data into adc_st
            measure('pound_demod_pulse', 'Pound_demod', adc_st, demod.full('integ_pound', pound_signal, 'out1'))
            # play another pound pulse to have a continuous pound signal independent of the time of flight of the demodulation signal
            assign(pound_signal_int, Cast.mul_int_by_fixed(1e8, pound_signal))

            play('pound_pulse', 'RR')

            # save the pound signal into a stream
            save(pound_signal_int, pound_signal_st)

    # stream processing handling
    with stream_processing():
        adc_st.input1().save('adc1') # see the raw data
        pound_signal_st.buffer(freq_array_size).average().save('pound_signal') # buffer it into vectors the length of the frequency scan and average over the averaging loop

## Execute the program

job = qm.execute(PDH_spectroscopy)
job.result_handles.wait_for_all_values()
res = job.result_handles
res.wait_for_all_values()

adc1 = res.get("adc1").fetch_all()
pound_signal = res.get('pound_signal').fetch_all()

freq = [x/1e6 for x in np.linspace(f_init, f_final, int((f_final-f_init)/df))]

plt.figure(figsize=(16,12))
plt.plot(freq,pound_signal,linewidth=5, markersize=10)
plt.title('PDH spectroscopy', fontsize=30)
plt.xlabel('Frequency [MHz]', fontsize=30)
plt.ylabel('PDH signal', fontsize=30)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.grid()

plt.figure(figsize=(16,12))
plt.plot(adc1)
plt.title('Raw acquired data (for the last shot)', fontsize=30)
plt.xlabel('Time [ns]', fontsize=30)
plt.ylabel('Raw ADC sample', fontsize=30)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.grid()

## Simulate
# It is also possible to simulate the signal to see that everything is phase coherent etc.. simulate for 20000 clock cycles or 80us
#
# job = qmm.simulate(config, PDH_spectroscopy, SimulationConfig(20000))
# samps = job.get_simulated_samples()
# RR_I = samps.con1.analog['1']
# RR_Q = samps.con1.analog['2']

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