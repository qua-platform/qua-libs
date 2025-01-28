import numpy as np
from qualang_tools.units import unit
from threading import Thread
u = unit()

######################
# AUXILIARY FUNCTIONS:
######################

def steps_calculator(offsets,step_time=20000,N_ss=60):
    '''Calculates useful values for update_offset function :
    step size given an offset array,
    small step size, 
    small step time'''
    step = np.mean(np.diff(offsets))
    #set small step size
    small_step = step/N_ss #small step size
    if np.abs(small_step) < 2**-16: # 2**-16 is the smallest step size possible
        if step<0:
            N_ss = int(np.ceil(np.abs(step)/2**-16))
        elif step>0:
            N_ss = int(np.floor(np.abs(step)/2**-16))
        small_step = step/N_ss
        print(f"small step size too small, using {N_ss} small steps instead")
    small_step_time = int(step_time/N_ss) #time that a small step takes in clock cycles (4 ns)
    return (N_ss,step, small_step, step_time, small_step_time)

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    return [float(x) for x in gauss_wave]

class PM100D:
    def __init__(self,name):
        rm = pyvisa.ResourceManager()
        # self.Instrument = rm.get_instrument(name)
        self.Instrument = rm.open_resource(name)
    
    def read_value(self):
        self.Instrument.write('READ?')
        return float(self.Instrument.read())

# For python 2
# class ThreadWithReturnValue(Thread):
#     def __init__(self, group=None, target=None, name=None,
#                  args=(), kwargs={}, Verbose=None):
#         Thread.__init__(self, group, target, name, args, kwargs, Verbose)
#         self._return = None
#     def run(self):
#         if self._Thread__target is not None:
#             self._return = self._Thread__target(*self._Thread__args,
#                                                 **self._Thread__kwargs)
#     def join(self):
#         Thread.join(self)
#         return self._return
    
# For python 3
class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

######################
# Network parameters #
######################
qop_ip = "10.236.8.121"  # Write the QM router IP address
cluster_name = "Cluster_83"  # Write your cluster_name if version >= QOP220
qop_port = 9510  # Write the QOP port if version < QOP220
PM1_address = r"USB0::0x1313::0x8078::P2222258::INSTR"
PM2_address = r"USB0::0x1313::0x8078::P2222259::INSTR"
####################
# USEFUL VARIABLES #
####################

# Parameters used in the other scripts
config_angle = 1

# Phase modulation
phase_modulation_IF = 250 * u.kHz
phase_mod_amplitude = 0.48
phase_mod_len = 100*u.us # resonance frequency of 30 kHz -> avoid around 33.3 us
# phase_mod_len = 200*u.us  #change for test with sam

# AOM
AOM_IF = 0
const_amplitude = 0.1
const_len = 100

# Filter cavity
offset_amplitude = 0.25  # Fixed do not change
offset_len = 16  # Fixed do not change
setpoint_filter_cavity_1 = 0.0 #the voltage offset on the filter cavity
setpoint_filter_cavity_2 = 0.0 #the voltage offset on the filter cavity
setpoint_filter_cavity_3 = 0.0 #the voltage offset on the filter cavity
setpoint_modulator = 0.0 #the voltage offset on the phase modulator
# Photo-diode
readout_len = phase_mod_len
time_of_flight = 192

#SPD trigger parameters
triggON_len=25*u.us #length of laser pulses
triggOFF_len=275*u.us #time during which laser is off (t_QP)

#Slow lock parameters
offsets = np.linspace(-0.5, 0.5, 1001) # This way, we sweep around setpoint
step_time=100
N_ss=60
N_ss,step, small_step, step_time, small_step_time=steps_calculator(offsets)

RFswitch_lookup = {
    "filters_transmission": 0,
	"filter_cavity_1": 1,
	"filter_cavity_2": 2,
	"filter_cavity_3": 3,
}
opticalswitches_lookup = {
    "0000": 0,
	"0001": 1,
	"0010": 2,
	"0011": 3,
    "0100": 4,
	"0101": 5,
	"0110": 6,
	"0111": 7,
    "1000": 8,
	"1001": 9,
	"1010": 10,
	"1011": 11,
    "1100": 12,
	"1101": 13,
	"1110": 14,
	"1111": 15,
}

################
# CONFIGURATION:
################

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                6: {"offset": setpoint_filter_cavity_1},
                7: {"offset": setpoint_filter_cavity_2},
                8: {"offset": setpoint_modulator},
                10: {"offset": setpoint_filter_cavity_3},
            },
            "digital_outputs": {
                1: {}, #Optical switch 1
                2: {}, #Optical switch 2
                3: {}, #Optical switch 3
                4: {}, #Optical switch 4
                5: {}, # SPD/AOM trigger
                7: {}, #RF switch pin0
                8: {}, #RF switch pin1
                9: {}, #RF switch pin2
            },
            "analog_inputs": {
                2: {"offset": 0},
            },
        }
    },
    "elements": {
        "trigger_SPD": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 2),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "trigON": "trigON_pulse",
                "trigOFF": "trigOFF_pulse",
            },
        },
        "RFswitch_pin0": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 7),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "switchON": "switchON_pulse",
                "switchOFF": "switchOFF_pulse",
            },
        },
        "RFswitch_pin1": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 8),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "switchON": "switchON_pulse",
                "switchOFF": "switchOFF_pulse",
            },
        },
        "RFswitch_pin2": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 9),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "switchON": "switchON_pulse",
                "switchOFF": "switchOFF_pulse",
            },
        },
        "opticalswitch_1": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 1),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "switchON": "switchON_pulse",
                "switchOFF": "switchOFF_pulse",
            },
        },
        "opticalswitch_2": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 2),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "switchON": "switchON_pulse",
                "switchOFF": "switchOFF_pulse",
            },
        },
        "opticalswitch_3": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 3),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "switchON": "switchON_pulse",
                "switchOFF": "switchOFF_pulse",
            },
        },
        "opticalswitch_4": {
            "digitalInputs": {
                'in0': {
                    'port': ('con1', 4),
                    'delay': 0,
                    'buffer': 0,
                  },
            },
            "operations": {
                "switchON": "switchON_pulse",
                "switchOFF": "switchOFF_pulse",
            },
        },
        "phase_modulator": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "intermediate_frequency": phase_modulation_IF,
            "operations": {
                "cw": "phase_mod_pulse",
            },
        },
        "filter_cavity_1": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "offset": "offset_pulse",
            },
            'sticky': {'analog': True, 'duration': 60}
        },
        "filter_cavity_2": {
            "singleInput": {
                "port": ("con1", 7),
            },
        "operations": {
            "offset": "offset_pulse",
        },
        'sticky': {'analog': True, 'duration': 60}
        },
        "filter_cavity_3": {
            "singleInput": {
                "port": ("con1", 10),
            },
        "operations": {
            "offset": "offset_pulse",
        },
        'sticky': {'analog': True, 'duration': 60}
        },
        "detector_DC": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "readout": "DC_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "detector_AC": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "intermediate_frequency": phase_modulation_IF,
            "operations": {
                "readout": "AC_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
        "offset_pulse": {
            "operation": "control",
            "length": offset_len,
            "waveforms": {
                "single": "offset_wf",
            },
        },
        "phase_mod_pulse": {
            "operation": "control",
            "length": phase_mod_len,
            "waveforms": {
                "single": "phase_mod_wf",
            },
        },
        "cw_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "single": "const_wf",
            },
        },
        "DC_readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "constant": "constant_weights",
            },
            "digital_marker": "ON",
        },
        "AC_readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "constant": "constant_weights",
            },
            "digital_marker": "ON",
        },
        "switchON_pulse": {
            "operation": "control",
            "length": readout_len,
            "digital_marker": "ON",
        },
        "switchOFF_pulse": {
            "operation": "control",
            "length": readout_len,
            "digital_marker": "OFF",
        },
        "trigON_pulse": {
            "operation": "control",
            "length": triggON_len,
            "digital_marker": "ON",
        },
        "trigOFF_pulse": {
            "operation": "control",
            "length": triggOFF_len,
            "digital_marker": "OFF",
        },

    },
    "waveforms": {
        "phase_mod_wf": {"type": "constant", "sample": phase_mod_amplitude},
        "const_wf": {"type": "constant", "sample": const_amplitude},
        "offset_wf": {"type": "constant", "sample": offset_amplitude},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]},
                          "OFF": {"samples": [(0, 0)]}},
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
    },
}

