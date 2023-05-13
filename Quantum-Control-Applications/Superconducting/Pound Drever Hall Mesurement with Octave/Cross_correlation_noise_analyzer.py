"""
Cross_correlation_noise_analyzer.py:

Version: 0.1

"""
## Imports
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from PDH_config_octave import *
import matplotlib.pyplot as plt
from qm import SimulationConfig # It is also possible to simulate the IQ signals
from qm.octave import *
from scipy.fft import fft, fftfreq
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
qm.octave.set_rf_output_gain(element, -4)
qm.octave.set_rf_output_mode(element, RFOutputMode.on)

qm.octave.set_qua_element_octave_rf_in_port(element,"octave1", 1)
qm.octave.set_downconversion(element, lo_source=RFInputLOSource.Internal, if_mode_i=IFMode.envelope, if_mode_q=IFMode.envelope) #Dmd2LO

qm.octave.calibrate_element(element, [(rr_LO, rr_IF)])

qm = qmm.open_qm(config)

## Cross Correlation Noise Analyzer Code
pound_samples_per_chunk = 1*int(2*250e6/pound_modulation_freq) #in clocks = 100ns to catch 2 periods of the PDH signal
pound_vector_size = int(pound_pulse_length/4/pound_samples_per_chunk)
correlation_length = 4096
averaging_length = 8
fft_averaging_length = 10
with program() as noise_analyzer:
    adc_st = declare_stream(adc_trace=True)
    adc_st2 = declare_stream(adc_trace=True)

    pound_signalI = declare(fixed)
    pound_signalQ = declare(fixed)
    dt = declare(int)
    pound_vector_I1 = declare(fixed, value=np.zeros(pound_vector_size))
    pound_vector_Q1 = declare(fixed, value=np.zeros(pound_vector_size))
    pound_vector_I2 = declare(fixed, value=np.zeros(pound_vector_size))
    pound_vector_Q2 = declare(fixed, value=np.zeros(pound_vector_size))

    pound_vector_I1_st = declare_stream()
    pound_vector_Q1_st = declare_stream()
    pound_vector_I2_st = declare_stream()
    pound_vector_Q2_st = declare_stream()


    pound_index = declare(int)
    index = declare(int)
    j = declare(int)
    j2 = declare(int)
    avg = declare(int)
    fft_avg = declare(int)

    corr_I = declare(fixed, value=np.zeros(correlation_length))
    corr_Q = declare(fixed, value=np.zeros(correlation_length))
    corr_I_st = declare_stream()
    corr_Q_st = declare_stream()

    align()

    play('pound_pulse', 'RR')
    align()
    with for_(pound_index, 0, pound_index<5*averaging_length*correlation_length*fft_averaging_length, pound_index+1):
        play('pound_pulse', 'RR')
        #frame_rotation_2pi(0.5, 'Pound_simulator')

    measure('pound_demod_pulse', 'Pound_demod', adc_st,
            demod.sliced('integ_pound', pound_vector_I1, pound_samples_per_chunk, 'out1'),
            demod.sliced('integ_pound', pound_vector_Q1, pound_samples_per_chunk, 'out2'))

    measure('pound_demod_pulse', 'Pound_demod_2', adc_st2,
            demod.sliced('integ_pound', pound_vector_I2, pound_samples_per_chunk, 'out1'),
            demod.sliced('integ_pound', pound_vector_Q2, pound_samples_per_chunk, 'out2'))

    with for_(fft_avg, 0, fft_avg < fft_averaging_length, fft_avg+1):
        with for_(avg, 0, avg < averaging_length, avg+1):
            with for_(j, 0, j < correlation_length, j+1):
                measure('pound_demod_pulse', 'Pound_demod', None,
                    demod.sliced('integ_pound', pound_vector_I1, pound_samples_per_chunk, 'out1'),
                    demod.sliced('integ_pound', pound_vector_Q1, pound_samples_per_chunk, 'out2'))
                wait(pound_samples_per_chunk*(j), 'Pound_demod_2')
                measure('pound_demod_pulse', 'Pound_demod_2', None,
                    demod.sliced('integ_pound', pound_vector_I2, pound_samples_per_chunk, 'out1'),
                    demod.sliced('integ_pound', pound_vector_Q2, pound_samples_per_chunk, 'out2'))

                assign(corr_I[j], corr_I[j]+Math.dot(pound_vector_I1,pound_vector_I2))
                assign(corr_Q[j], corr_Q[j]+Math.dot(pound_vector_Q1, pound_vector_Q2))


        with for_(j2, 0, j2 < correlation_length, j2 + 1):
            assign(corr_I[j2], corr_I[j2]>>int(np.log2(averaging_length)))
            assign(corr_Q[j2], corr_Q[j2]>>int(np.log2(averaging_length)))
            save(corr_I[j2], corr_I_st)
            save(corr_Q[j2], corr_Q_st)
            assign(corr_I[j2], 0.0)
            assign(corr_Q[j2], 0.0)

    with for_(index, 0, index < pound_vector_size, index + 1):
        save(pound_vector_I1[index], pound_vector_I1_st)
        save(pound_vector_Q1[index], pound_vector_Q1_st)
        save(pound_vector_I2[index], pound_vector_I2_st)
        save(pound_vector_Q2[index], pound_vector_Q2_st)

    with stream_processing():
        adc_st.input1().save('adcI1') # see the raw data
        adc_st.input2().save('adcQ1')
        adc_st2.input1().save('adcI2') # see the raw data
        adc_st2.input2().save('adcQ2')
        pound_vector_I1_st.buffer(pound_vector_size).save_all('pound_vector_I1')
        pound_vector_Q1_st.buffer(pound_vector_size).save_all('pound_vector_Q1')
        pound_vector_I2_st.buffer(pound_vector_size).save_all('pound_vector_I2')
        pound_vector_Q2_st.buffer(pound_vector_size).save_all('pound_vector_Q2')
        corr_I_st.buffer(correlation_length).save_all('corr_I')
        corr_Q_st.buffer(correlation_length).save_all('corr_Q')
        corr_I_st.buffer(correlation_length).fft('abs').average().save('PSD_I')

## Run the Program
# After program execution We can plot the cross correlation and the FFT of the auto correlation.
# The FFT can be done in python or online in the stream processing
job = qm.execute(noise_analyzer)
job.result_handles.wait_for_all_values()
res = job.result_handles
adcI1 = res.get("adcI1").fetch_all()
adcQ1 = res.get("adcQ1").fetch_all()
adcI2 = res.get("adcI2").fetch_all()
adcQ2 = res.get("adcQ2").fetch_all()
pound_vector_I1 = res.get('pound_vector_I1').fetch_all()[0][0]
pound_vector_Q1 = res.get('pound_vector_Q1').fetch_all()[0][0]
pound_vector_I2 = res.get('pound_vector_I2').fetch_all()[0][0]
pound_vector_Q2 = res.get('pound_vector_Q2').fetch_all()[0][0]
corr_I = res.get('corr_I').fetch_all()['value'][0]
corr_Q = res.get('corr_Q').fetch_all()['value'][0]
PSD=res.get('PSD_I').fetch_all()

# Number of sample points
N = corr_I.__len__()
# sample spacing
T = pound_samples_per_chunk*4e-9
xI = np.linspace(0.0, N*T, N, endpoint=False)
yI = corr_I
yfI = fft(yI)
xfI = fftfreq(N, T)[:N//2]

xQ = np.linspace(0.0, N*T, N, endpoint=False)
yQ = corr_Q
yfQ = fft(yQ) #PSD
xfQ = fftfreq(N, T)[:N//2]

plt.figure(figsize=(16,12))
plt.loglog(xfI, 2.0/N * np.abs(yfI[0:N//2]),linewidth=5, markersize=10)
plt.loglog(xfQ, 2.0/N * np.abs(yfQ[0:N//2]),linewidth=5, markersize=10)
plt.title('FFT of autocorrelation', fontsize=30)
plt.xlabel('Freq (Hz)', fontsize=30)
plt.ylabel('FFT Amplitude |X(freq)|', fontsize=30)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.grid()

plt.figure(figsize=(16,12))
plt.loglog(xfI,2.0/N * np.abs(PSD[0:N//2]),linewidth=5, markersize=10)
plt.title('FFT from stream processing', fontsize=30)
plt.xlabel('Freq (Hz)', fontsize=30)
plt.ylabel('FFT Amplitude |X(freq)|', fontsize=30)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)
plt.grid()



## Simulate the signals
# job = qmm.simulate(config, noise_analyzer, SimulationConfig(2000))
# samps = job.get_simulated_samples()
# pound_simulated_signal9 = samps.con1.analog['9']
# pound_simulated_signal10 = samps.con1.analog['10']
#
#
# plt.figure(figsize=(16,12))
# plt.plot(pound_vector_I1,linewidth=5, markersize=10)
# plt.plot(pound_vector_Q1,linewidth=5, markersize=10)
# plt.plot(pound_vector_I2,linewidth=5, markersize=10)
# plt.plot(pound_vector_Q2,linewidth=5, markersize=10)
# plt.title('pound demod vectors', fontsize=30)
# plt.xlabel('Shot no.', fontsize=30)
# plt.ylabel('I/Q values', fontsize=30)
# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 20)
# plt.grid()
#
#
#
# plt.figure(figsize=(16,12))
# plt.plot([(x+1)*pound_samples_per_chunk*4 for x in range(corr_I.__len__())], corr_I,linewidth=5, markersize=10)
# plt.plot([(x+1)*pound_samples_per_chunk*4 for x in range(corr_Q.__len__())], corr_Q,linewidth=5, markersize=10)
# plt.title('autocorrelation I/Q', fontsize=30)
# plt.xlabel('dt', fontsize=30)
# plt.ylabel('Raw ADC data', fontsize=30)
# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 20)
# plt.grid()
#
# plt.figure(figsize=(16,12))
# plt.plot(adcI1,linewidth=5, markersize=10)
# plt.plot(adcQ1,linewidth=5, markersize=10)
# plt.plot(adcI2,linewidth=5, markersize=10)
# plt.plot(adcQ2,linewidth=5, markersize=10)
# plt.title('Raw ADC data', fontsize=30)
# plt.xlabel('Shot no.', fontsize=30)
# plt.ylabel('Raw ADC data', fontsize=30)
# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 20)
# plt.grid()
#
# plt.figure(figsize=(16,12))
# plt.plot(pound_simulated_signal9,linewidth=5, markersize=10)
# plt.plot(pound_simulated_signal10,linewidth=5, markersize=10)
# plt.title('Simulation output', fontsize=30)
# plt.xlabel('Shot no.', fontsize=30)
# plt.ylabel('Volt', fontsize=30)
# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 20)
# plt.grid()