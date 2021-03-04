from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import QuantumMachine
from qm import LoopbackInterface
from qm import SimulationConfig
from bakary import *
from qm.qua import *

π = np.pi
t = np.linspace(-3, 3, 16)
gauss = 0.4 * np.exp(-t ** 2 / 2)
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


def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


def gauss_der(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_der_wave = amplitude * (-2 * (t - mu)) * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_der_wave = gauss_der_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_der_wave]


with baking(config=config, padding_method="symmetric_r") as b:
    # gaussianOp = gauss(100, 0.4, 3, 1, 8)
    # derGaussOp = gauss_der(100, 0.4, 3, 1, 8)
    # b.add_Op('Gauss_Op_b', "qe1", [gaussianOp, derGaussOp])
    const_Op = [0.3, 0.3, 0.3, 0.3, 0.3]
    const_Op2 = [0.2, 0.2, 0.2, 0.3, 0.4]
    b.add_Op("const_Op_b", "fluxline", const_Op) #Add marker name as optional
    b.add_Op("const_Op2", "qe1", [const_Op, const_Op2])
    Op3 = [1., 1., 1.]
    Op4 = [2., 2., 2.]
    b.add_Op("Op3", "qe1", [Op3, Op4])
    b.play("const_Op2", "qe1")
    print("wait")
    # b.play_at("Op3", "qe1", t=2)
    b.wait(-3, "qe1")
    b.play("Op3", "qe1")
    b.play("Op3", "qe1")

    b.ramp(0.2, 6, "fluxline")
    b.align("qe1", "fluxline")

# with baking(config=config, padding_method="symmetric_r") as b2:
#     # gaussianOp = gauss(100, 0.4, 3, 1, 8)
#     # derGaussOp = gauss_der(100, 0.4, 3, 1, 8)
#     # b.add_Op('Gauss_Op_b', "qe1", [gaussianOp, derGaussOp])
#     const_Op = [0.3, 0.3, 0.3, 0.3, 0.3]
#     const_Op2 = [0.2, 0.2, 0.2, 0.3, 0.4]
#     b2.add_Op("const_Op_b", "fluxline", const_Op) #Add marker name as optional
#     b2.add_Op("const_Op2", "qe1", [const_Op, const_Op2])
#     b2.play("const_Op2", "qe1")
#     b2.ramp(0.2, 6, "fluxline")
#     b2.align("qe1", "fluxline")



qmm = QuantumMachinesManager("3.122.60.129")
QM = qmm.open_qm(config)

with program() as prog:
    b.run()


job = qmm.simulate(config, prog,
                   SimulationConfig(int(1000), simulation_interface=LoopbackInterface(
                       [("con1", 1, "con1", 1)])))  # Use LoopbackInterface to simulate the response of the qubit

samples = job.get_simulated_samples()

samples.con1.plot()


