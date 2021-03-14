from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import QuantumMachine
from qm import LoopbackInterface
from qm import SimulationConfig
from bakary import *
from qm.qua import *

π = np.pi
t = np.linspace(-3, 3, 16)
sigma= 1
gauss = np.exp(-t ** 2 / (2*sigma**2))
lmda = 0.5
alpha = -1
d_gauss = lmda * (-t) * gauss / alpha
config = {

    'version': 1,

    'controllers': {

        "con1": {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': +0.0},  # qubit 1 I
                2: {'offset': +0.0},  # qubit 1 Q
                3: {'offset': +0.0},  # flux line
            },
            'analog_inputs': {
                1: {'offset': +0.0},
            }
        },

    },

    'elements': {

        "qe1": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
            },
            'intermediate_frequency': 0,
            'operations': {
                'playOp': "mixedConst",
                'gaussOp': "mixedGauss",
                'dragOp': "DragPulse",

            },
        },
        "fluxline": {
            'singleInput': {
                "port": ("con1", 3),
            },
            'intermediate_frequency': 0,
            'hold_offset': {'duration': 100},
            'operations': {
                'iSWAP': "constPulse",

            },
        }

    },

    "pulses": {
        "mixedConst": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'const_wf'
            }
        },
        "mixedGauss": {
            'operation': 'control',
            'length': len(gauss),
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },

        "constPulse": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'single': 'const_wf'
            }
        },
        "DragPulse": {
            'operation': 'control',
            'length': len(gauss),
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'd_gauss_wf'
            }

        },
        "gaussianPulse": {
            'operation': 'control',
            'length': len(gauss),
            'waveforms': {
                'single': 'gauss_wf'
            }
        },

    },

    "waveforms": {
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
        'const_wf': {
            'type': 'constant',
            'sample': 0.2
        },
        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss
        },
        'd_gauss_wf': {
            'type': 'arbitrary',
            'samples': d_gauss
        },

    },
}

# def gauss(amplitude, mu, sigma, length):
#     t = np.linspace(-length / 2, length / 2, length)
#     gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
#     return [float(x) for x in gauss_wave]
# return [amplitude for x in t]

b_list = []
for i in range(10):
    with baking(config=config, padding_method="symmetric_r") as b:
        b.add_Op("Drag_pulse", "qe1", [list(gauss), list(d_gauss)])
        b.play("Drag_pulse", "qe1", amp=0.05 * i)
    b_list.append(b)

s = deterministic_run(b_list)
qmm = QuantumMachinesManager("3.122.60.129")
QM = qmm.open_qm(config)

with program() as prog:
    j = declare(int, value=1)
    #with for_(j, 0, cond=j < 10, update=j + 1):
    s(j)

job = qmm.simulate(config, prog,
                   SimulationConfig(int(10000//4), simulation_interface=LoopbackInterface(
                       [("con1", 1, "con1", 1)])))  # Use LoopbackInterface to simulate the response of the qubit

samples = job.get_simulated_samples()

samples.con1.plot()



