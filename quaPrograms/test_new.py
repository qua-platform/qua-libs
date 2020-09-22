from ProgramNode import *
from graphviz import Source
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface

config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                5: {'offset': +0.0},
            },
            'analog_inputs': {
                1: {'offset': 0},  # SET READ
            },
        }
    },

    'elements': {

        'qe1': {
            'singleInput': {
                'port': ('con1', 5)
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

QM = QMm.open_qm(config)


def qua_wrap(var_name):
    with program() as qua_prog1:
        A = declare(fixed, value=2)
        save(A, var_name)
    return qua_prog1


def py_func(s, ab):
    return {'v': s + ab[0]}


def qua_wrap2(d, a):
    with program() as qua_prog:
        A = declare(fixed, value=a + d['ass'][0])
        save(A, 'aas')
    return qua_prog


sim_args = {'simulate': SimulationConfig(int(1e3))}

a = QuaNode(1, _program=qua_wrap, _input={'var_name': 'ass'}, _output_vars={'ass'},
            _quantum_machine=QM, _simulation_kwargs=sim_args)
b = PyNode(2, _program=py_func, _input={'s': 0.55, 'ab': a.output('ass')}, _output_vars={'v'})

c = QuaNode(3, _program=qua_wrap2, _input={'d': a.output(), 'a': b.output('v')}, _output_vars={'aas'},
            _quantum_machine=QM, _simulation_kwargs=sim_args)
d = PyNode(4, _program=lambda x: {'x': x}, _input={'x': 1}, _output_vars={'x'})
g = ProgramGraph()
g.add_nodes([a, b, c, d])
g.add_edges({(d, a)})
# g.remove_nodes({c})
print("Put the following in webgraphviz.com:")
print(g.export_dot_graph())
g.run({a})
print(c.result)
# graph_plot = Source(g.export_dot_graph())
# graph_plot.render('g', view=True)

