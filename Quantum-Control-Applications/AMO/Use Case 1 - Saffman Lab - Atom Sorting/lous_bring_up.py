"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qm import generate_qua_script
from qualang_tools.units import unit
import numpy as np

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)


qop_ip = "172.16.33.101"  # Write the QM router IP address
cluster_name = "Cluster_83"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

g_amp = 0.35
g_len = 400
g_sig = g_len/5

# Reduced sampling rate for generating long pulses without memory issues
sampling_rate = 1000e6
column_selector_if = 100e6

def gaussian(amplitude, length, sigma):
    """
    :param amplitude: maximum amplitude [V] (float)
    :param length: pulse duration [ns] (int)
    :param sigma: standard deviation (float)
    :return:
    """
    t = np.arange(length, step=1e9 / sampling_rate)  # An array of size pulse length in ns
    center = (length - 1e9 / sampling_rate) / 2
    gauss_wave = amplitude * np.exp(-((t - center) ** 2) / (2 * sigma**2)) 
    return gauss_wave

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                # Row AOD tone
                1: {"offset": 0.0},    
            },
        },
    },
    "elements": {
        # column_selector is used to control the column AOD
        "column_selector": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "intermediate_frequency": column_selector_if,
            "operations": {
                "gaussian": "gaussian_pulse",
            },
        },
    },
    "pulses": {
        "gaussian_pulse": {
            "operation": "control",
            "length": g_len,
            "waveforms": {
                "single": "gaussian_wf",
            },
        },
    },
    "waveforms": {
        "gaussian_wf": {
            "type": "arbitrary",
            'is_overridable': True,
            "samples": gaussian(g_amp, g_len, g_sig),
            #"sampling_rate": sampling_rate,
            #"maxAllowedError": 1e-2,
        },
    },
}

update_amp = 0.5 #[0, 1.99]
update_duration = 0.8 * u.us
cut_pulse = 0.4 * u.us

###################
# The QUA program #
###################

with program() as demo:
    with infinite_loop_():
        play("gaussian", "column_selector")
        #update_frequency("column_selector", 30e6)
        #play("gaussian", "column_selector")
        #update_frequency("column_selector", column_selector_if)
        #play("gaussian" * amp(update_amp), "column_selector", duration = update_duration, truncate = cut_pulse)#phase/frame rotation
        #play(ramp(0.0001), "column_selector", duration = 2 * u.us) #ramp rate in V/ns
        

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=5_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, demo, simulation_config)
    # Plot the simulated samples
    samples = job.get_simulated_samples().con1.plot()
    plt.show()
    #waveform_report = job.get_simulated_waveform_report()
    #waveform_report.create_plot(samples, plot=True, save_path="./")
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(demo)
    '''program_id = qm.compile(demo)
    pending_job = qm.queue.add_compiled(program_id, overrides={
        'waveforms': {
            "gaussian_wf": gaussian(0.5*g_amp, g_len, g_sig)}
    })'''