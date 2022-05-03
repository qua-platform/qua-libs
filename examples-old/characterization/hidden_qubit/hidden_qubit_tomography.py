"""hidden_qubit_tomography.py: Performing process tomography with one control and one hidden qubit
Author: Arthur Strauss - Quantum Machines
Created: 08/01/2021
QUA version used : 0.8.439
"""

# QM imports
from configuration import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import QuantumMachine


qm1 = QuantumMachinesManager()
QM = qm1.open_qm(config)

N_shots = 100
# We plug in IO variables coefficients coming from a previously performed state discrimination procedure
QM.set_io1_value(1.0)
QM.set_io2_value(1.0)

state_prep = {
    "00": [],
    "00-i10": ["Rx_π_2"],
    "-i10": ["Rx_π"],
    "00+10": ["Ry_π_2"],
    "01": ["Rx_π", "iSWAP"],
    "-i11": ["Rx_π", "iSWAP", "Rx_π"],
    "01-i11": ["Rx_π", "iSWAP", "Rx_π_2"],
    "01+11": ["Rx_π", "iSWAP", "Ry_π_2"],
    "00+01": ["Rx_π_2", "iSWAP"],
    "-i10-i11": ["Rx_π_2", "iSWAP", "Rx_π"],
    "00+01-i10-i11": ["Rx_π_2", "iSWAP", "Rx_π_2"],
    "00+01+10+11": ["Rx_π_2", "iSWAP", "Ry_π_2"],
    "00+i01": ["Ry_π_2", "iSWAP"],
    "-i10+11": ["Ry_π_2", "iSWAP", "Rx_π"],
    "00+i01-i10+11": ["Ry_π_2", "iSWAP", "Rx_π_2"],
    "00+i01+10+i11": ["Ry_π_2", "iSWAP", "Ry_π_2"],
}
tomography_set = {
    "ID__σ_x": ["Rx_π_2", "iSWAP", "CPHASE", "readout"],
    "ID__σ_y": ["Ry_π_2", "iSWAP", "CPHASE", "readout"],
    "ID__σ_z": ["iSWAP", "readout"],
    "σ_z__ID": ["readout"],
    "σ_z__σ_x": ["Rx_π_2", "iSWAP", "readout"],
    "σ_z__σ_y": ["Ry_π_2", "iSWAP", "readout"],
    "σ_z__σ_z": ["Ry_π_2", "iSWAP", "Ry_π_2", "CPHASE", "Ry_π_2", "readout"],
    "σ_y__ID": ["Rx_π_2", "readout"],
    "σ_y__σ_x": ["Rx_π_2", "CPHASE", "readout"],
    "σ_y__σ_y": ["Ry_π_2", "iSWAP", "Rx_π_2", "readout"],
    "σ_y__σ_z": ["Ry_π_2", "iSWAP", "Rx_π_2", "CPHASE", "readout"],
    "σ_x__ID": ["Ry_π_2", "readout"],
    "σ_x__σ_x": ["Rx_π_2", "iSWAP", "Rx_π_2", "readout"],
    "σ_x__σ_y": ["Ry_π_2", "iSWAP", "Ry_π_2", "readout"],
    "σ_x__σ_z": ["Ry_π_2", "iSWAP", "Ry_π_2", "CPHASE", "readout"],
}
processes = {
    "1": ["Rx_π_2"],
    "2": ["Ry_π_2"],
    "3": ["iSWAP"],
    "4": ["CPHASE"]
    # Add any arbitrary process we'd like to characterize here, as a sequence of previous elementary gates
}

with program() as hidden_qubit_tomography:
    N = declare(fixed)
    with for_(var=N, init=0, cond=N < N_shots, update=N + 1):
        for process in processes.keys():
            for input_state in state_prep.keys():
                for readout_operator in tomography_set.keys():
                    for pulse in state_prep[input_state]:
                        play_pulse(pulse)
                    for pulse in processes[process]:
                        play_pulse(pulse)

                    for op in tomography_set[readout_operator]:
                        play_readout(op)  # See def of the function, remains to define the real readout operation
