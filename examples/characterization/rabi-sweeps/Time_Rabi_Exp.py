# Importing the necessary from qm
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

## Definition of the sample for Gaussian pulse
gauss_pulse_len = 20 # nsec
Amp=0.2     #Pulse Amplitude
gauss_arg = np.linspace(-3, 3, gauss_pulse_len)
gauss_wf = np.exp(-gauss_arg**2/2)
gauss_wf = Amp * gauss_wf / np.max(gauss_wf)

t_max = 0.500 #Maximum pulse duration
dt = 0.001 #timestep
N_t = int(t_max / dt) #Number of timesteps
N_max=3
## Setting up the configuration of the experimental setup
## Embedded in a Python dictionary
config = {
    'version': 1,

    'controllers': {  # Define the QM device and its outputs, in this case:
        'con1': {  # 2 analog outputs for the in-phase and out-of phase components
            'type': 'opx1',  # of the qubit (I & Q), and 2 other analog outputs for the coupled readout resonator
            'analog_outputs': {
                1: {'offset': 0.032},
                2: {'offset': 0.041},
                3: {'offset': -0.024},
                4: {'offset': 0.115},
            },
            'analog_inputs' : {
                1: {'offset': +0.0},

            }

        }
    },

    'elements': {  # Define the elements composing the quantum system, i.e the qubit+ readout resonator (RR)
        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),  # Connect the component to one output of the OPX
                'Q': ('con1', 2),
                'lo_frequency': 5.10e7,
                'mixer': 'mixer_qubit'  ##Associate a mixer entity to control the IQ mixing process
            },
            'intermediate_frequency': 5.15e3,  # Resonant frequency of the qubit
            'operations': {  # Define the set of operations doable on the qubit, each operation is related
                'gauss_pulse': 'gauss_pulse_in'  # to a pulse
            },
        },
        'RR': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': 6.00e7,
                'mixer': 'mixer_res'
            },
            'intermediate_frequency': 6.12e3,
            'operations': {
                'meas_pulse': 'meas_pulse_in',
            },
            'time_of_flight': 28,  # Measurement parameters
            'smearing': 0,
            'outputs': {
                'out1': ('con1', 1)
            }

        },
    },

    'pulses': {  # Pulse definition
        'meas_pulse_in': {
            'operation': 'measurement',
            'length': 20,
            'waveforms': {
                'I': 'exc_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
            },
            'digital_marker': 'marker1'
        },
        'gauss_pulse_in': {
            'operation': 'control',
            'length': 20,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            },
        }
    },

    'waveforms': {
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
        'exc_wf': {
                'type': 'constant',
                'sample': 0.479
        },

        'gauss_wf': {
                'type': 'arbitrary',
                'samples': gauss_wf.tolist()
        }

    },

    'digital_waveforms': {
        'marker1': {
            'samples': [(1, 4), (0, 2), (1, 1), (1, 0)]
        }
    },

    'integration_weights': {#Define integration weights for measurement demodulation
        'integW1': {
            'cosine': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                       4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                       4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                       4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            'sine': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        'integW2': {
            'cosine': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'sine': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                     4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                     4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                     4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        }
    },

    'mixers': {  #Potential corrections to be brought related to the IQ mixing scheme
        'mixer_res': [
            {'intermediate_frequency': 6.12e3, 'lo_frequency': 6.00e7, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ],
        'mixer_qubit': [
            {'intermediate_frequency': 5.15e3, 'lo_frequency': 5.10e7, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ],
    }
}
qmManager = QuantumMachinesManager("3.122.60.129") # Reach OPX's IP address
my_qm = qmManager.open_qm(config)  #Generate a Quantum Machine based on the configuration described above

with program() as timeRabiProg:  #Time Rabi QUA program
    I = declare(fixed)      #QUA variables declaration
    Q = declare(fixed)
    t = declare(fixed)  #Sweeping parameter over the set of durations
    Nrep = declare(int)  #Number of repetitions of the experiment
    I_stream = declare_stream()  # Declare streams to store I and Q components
    Q_stream = declare_stream()

    with for_(t, 0.00, t <= t_max, t + dt):  # Sweep from 0 to 50 *4 ns the pulse duration
        with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do a 100 times the experiment to obtain statistics

            play('gauss_pulse', 'qubit',duration=t)
            align("qubit", "RR")
            measure('meas_pulse', 'RR', 'samples', ('integW1', I), ('integW2', Q))
            save(I, I_stream)
            save(Q, Q_stream)

        save(t, 't')
    with stream_processing():
        I_stream.buffer(N_max).save_all('I')
        Q_stream.buffer(N_max).save_all('Q')

my_job = my_qm.simulate(timeRabiProg,
                   SimulationConfig(int(100000),simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)]))) ##Use Loopback interface for simulation of the output of the resonator readout
time.sleep(1.0)
my_timeRabi_results = my_job.result_handles
I1=my_timeRabi_results.I.fetch_all()['value']
Q1=my_timeRabi_results.Q.fetch_all()['value']
t1=my_timeRabi_results.t.fetch_all()['value']

samples = my_job.get_simulated_samples()

#Processing the data
def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit


I_avg=[]
Q_avg=[]
for i in range(len(I1)):
    I_avg.append((np.mean(I1[i])))
    Q_avg.append((np.mean(Q1[i])))

#Build a fitting tool for finding the right amplitude
# #(initial parameters to be adapted according to qubit and RR frequencies)
I_params, I_fit = fit_function(t1,
                                 I_avg,
                                 lambda x, A, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi)),
                                 [0.002, 0.15, 0])
Q_params, Q_fit = fit_function(t1,
                                 Q_avg,
                                 lambda x, A, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi)),
                                 [0.002, 0.15, 0])

plt.figure()
plt.plot(t1,I_avg,marker='x',color='blue',label='I-component')
plt.plot(t1,Q_avg,marker='o',color='green',label='Q-component')
plt.plot(t1,I_fit,color='red',label='Sinusoidal fit')
plt.plot(t1,Q_fit,color='black',label='Sinusoidal fit')
plt.xlabel('Amplitude [a.u]')
plt.ylabel('Measured signal [a.u]')
plt.axvline(I_params[1]/2, color='red', linestyle='--')
plt.axvline(I_params[1], color='red', linestyle='--')
plt.annotate("", xy=(I_params[1], 0), xytext=(I_params[1]/2,0), arrowprops=dict(arrowstyle="<->", color='red'))
plt.annotate("$\pi$", xy=(I_params[1]/2-0.03, 0.1), color='red')
plt.show()

print("The duration required to perform a X gate is",I_params[1]/2)