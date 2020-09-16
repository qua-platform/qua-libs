from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import matplotlib.pyplot as plt
from quaProgram import *
import networkx as nx
import numpy as np

config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},
            },
            'analog_inputs': {
                1: {'offset': 0},  # SET READ
            },
        }
    },

    'elements': {

        'qe1': {
            'singleInput': {
                'port': ('con1', 1)
            },
            'intermediate_frequency': 100e6,
            'digitalInputs': {
                'digital_input1': {
                    'port': ('con1', 1),
                    'delay': 0,
                    'buffer': 0
                }
            },
            'outputs': {
                'output1': ('con1', 1)
            },
            'time_of_flight': 100,
            'smearing': 70,
            'operations': {
                'playOp': 'constPulse',
                # 'digOp': 'digPulse'
            },
        },
    },

    "pulses": {
        "constPulse": {
            'operation': 'measurement',
            'length': 1000,
            'waveforms': {
                'single': 'const_wf'
            },
            'digital_marker': 'digital_waveform_high',
            'integration_weights': {
                'integ1': 'integW1',
            }
        },
    },

    'integration_weights': {
        'integW1': {
            'cosine': [1] * 700,  # [x*x/700 for x in range(700)],
            'sine': [0.0] * 700
        },
    },
    'digital_waveforms': {
        'digital_waveform_high': {
            'samples': [(1, 50)]
        }
    },
    "waveforms": {
        'const_wf': {
            'type': 'constant',
            'sample': 0.1
        },

    },
}

# Open communication with the server.
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.

QM1 = QMm.open_qm(config)


def executor(prog):
    return QM1.simulate(prog,
                        SimulationConfig(int(2e3), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))


def qua_wrap(var_name, W):
    with program() as qua_prog:
        A = declare(fixed, size=100)

        qua_stream = declare_stream()
        measure('playOp', 'qe1', qua_stream, demod.moving_window("integ1", A, 7, W))

        # save array
        i = declare(int)
        with for_(i, 0, i < A.length(), i + 1):
            save(A[i], var_name)

        with stream_processing():
            qua_stream.input1().save_all('qua1_data')
    return qua_prog


qua1 = QuaProgram('qua1', qua_wrap, {'var_name': 'A', 'W': 1}, {'A': None,'qua_stream': None})
qua2 = QuaProgram('qua2', qua_wrap, {'var_name': 'B', 'W': 3}, {'B': None,'qua_stream': None})
qua3 = QuaProgram('qua3', qua_wrap, {'var_name': 'C', 'W': 7}, {'C': None,})

prog_graph = nx.DiGraph()
prog_graph.add_node('qua1', prog=qua1)
prog_graph.add_node('qua2', prog=qua2)
prog_graph.add_node('qua3', prog=qua3)
prog_graph.add_edges_from([('qua1', 'qua2'), ('qua2', 'qua3')])


runner = QuaGraphExecutor(executor, prog_graph)
runner.execute()
runner.plot()

for n in prog_graph.nodes():
    qua_prog = prog_graph.nodes[n]['prog']
    data = qua_prog.get_output()
    for k in data.keys():
        plt.figure()
        plt.plot(data[k])
        plt.show()
