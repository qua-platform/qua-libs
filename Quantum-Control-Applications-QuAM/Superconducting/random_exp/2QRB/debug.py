
# Single QUA script generated at 2025-02-13 11:56:19.159966
# QUA library version: 1.2.1

from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(int, )
    v3 = declare(int, )
    v4 = declare(int, )
    v5 = declare(int, )
    v6 = declare(int, )
    v7 = declare(int, value=2)
    v8 = declare(int, value=4)
    v9 = declare(int, value=16)
    v10 = declare(int, value=8)
    v11 = declare(int, value=32)
    v12 = declare(int, value=64)
    a1 = declare(int, value=[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    a2 = declare(int, value=[8, 8, 64, 1, 1, 64, 8, 64, 16, 64, 1, 8, 64, 8, 1, 64, 8, 8, 64, 1, 8, 64, 8, 1, 64, 16, 64, 1, 8, 64, 8, 1, 64, 8, 8, 64, 1, 8, 64, 8, 1, 64, 16, 64, 1, 8, 64, 8, 1, 64, 8, 8, 64, 1, 8, 64, 8, 1, 64, 16, 64, 1, 8, 64, 8, 1, 64, 8, 64, 32])
    v13 = declare(fixed, )
    v14 = declare(fixed, )
    v15 = declare(fixed, )
    v16 = declare(fixed, )
    with for_(v5,0,(v5<100),(v5+1)):
        r1 = declare_stream()
        save(v5, r1)
        wait(12500, "qubitC2.resonator")
        align()
        with for_(v4,0,(v4<70),(v4+1)):
            with if_((v9==a2[v4]), unsafe=True):
                align("qubitC2.xy", "qubitC2.z", "qubitC2.resonator", "qubitC1.xy", "qubitC1.z", "qubitC1.resonator", "qubitC4.xy", "qubitC4.z", "qubitC4.resonator")
                play("const"*amp(0.36), "qubitC4.z", duration=23)
                frame_rotation_2pi(0.0, "qubitC4.xy")
                play("x180"*amp(0.0), "qubitC4.xy", duration=4)
                align("qubitC2.xy", "qubitC2.z", "qubitC2.resonator", "qubitC1.xy", "qubitC1.z", "qubitC1.resonator", "qubitC4.xy", "qubitC4.z", "qubitC4.resonator")
                play("Cz_unipolar.flux_pulse_control_qubitC1", "qubitC2.z")
                align("qubitC2.xy", "qubitC2.z", "qubitC2.resonator", "qubitC1.xy", "qubitC1.z", "qubitC1.resonator", "qubitC4.xy", "qubitC4.z", "qubitC4.resonator")
                frame_rotation_2pi(0.4917559640023226, "qubitC2.xy")
                frame_rotation_2pi(0.017082629644135955, "qubitC1.xy")
                play("x180"*amp(0.0), "qubitC2.xy", duration=4)
                play("x180"*amp(0.0), "qubitC1.xy", duration=4)
                align("qubitC2.xy", "qubitC2.z", "qubitC2.resonator", "qubitC1.xy", "qubitC1.z", "qubitC1.resonator", "qubitC4.xy", "qubitC4.z", "qubitC4.resonator")
                align("qubitC2.xy", "qubitC2.z", "qubitC2.resonator", "qubitC1.xy", "qubitC1.z", "qubitC1.resonator", "qubitC4.xy", "qubitC4.z", "qubitC4.resonator")
            with elif_((v12==a2[v4])):
                align()
            with elif_((v11==a2[v4])):
                measure("readout", "qubitC2.resonator", None, dual_demod.full("iw1", "iw2", v13), dual_demod.full("iw3", "iw1", v14))
                assign(v1, Cast.to_int((v13>0.0002507775242372948)))
                wait(625, "qubitC2.resonator")
                measure("readout", "qubitC4.resonator", None, dual_demod.full("iw1", "iw2", v15), dual_demod.full("iw3", "iw1", v16))
                assign(v2, Cast.to_int((v15>0.000779668352087324)))
                wait(625, "qubitC4.resonator")
                assign(v3, ((Cast.to_int(v1)<<1)+Cast.to_int(v2)))
                r2 = declare_stream()
                save(v3, r2)
                wait(50000, "qubitC2.resonator")
            with elif_((v10==a2[v4])):
                frame_rotation_2pi(0.25, "qubitC2.xy")
            with elif_((v7==a2[v4])):
                play("x90", "qubitC2.xy")
            with elif_((v8==a2[v4])):
                play("x180", "qubitC2.xy")
    with stream_processing():
        r1.save("n")
        r2.buffer(1).buffer(1).buffer(100).save("state1")


config = {
    "version": 1,
    "controllers": {
        "con3": {
            "analog_outputs": {
                "1": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "2": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
            },
            "analog_inputs": {
                "1": {
                    "gain_db": 0,
                    "shareable": False,
                    "offset": 0.018978428548177082,
                },
                "2": {
                    "gain_db": 0,
                    "shareable": False,
                    "offset": 0.01666322998046875,
                },
            },
        },
        "con1": {
            "analog_outputs": {
                "3": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "4": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "7": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "8": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "9": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "10": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "5": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "6": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "1": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "2": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
            },
            "analog_inputs": {
                "1": {
                    "gain_db": 0,
                    "shareable": False,
                    "offset": 0.018978428548177082,
                },
                "2": {
                    "gain_db": 0,
                    "shareable": False,
                    "offset": 0.01666322998046875,
                },
            },
        },
        "con2": {
            "analog_outputs": {
                "1": {
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.7437646298079738, -1.5474800067682553, 0.9638397736653462, -0.09863832640526682, -0.1525128963855616, 0.16087758901132937, -0.09051889113973684, 0.010077582254409289, 0.027090116685094576, 0.00033456171915884417, -0.023190809959393446, -0.006128571097164163, 0.0360283330805724, -0.03864695005187116, 0.01870973771548932, 0.032648599368585274, -0.07137166532104003, 0.03161425556665239, 0.020460680461686618, -0.012764200827653933, 0.003101094270150914, -0.02041290517345084, 0.01920022493232309, -0.002568138962594708, 0.008915921192951131, -0.01062843563049461, -0.0226370653075193, 0.045729720758981615, -0.02947007539168625, 0.006815779492935853],
                        "feedback": [0.9763347431904402, 0.8106993503581621],
                    },
                    "offset": 0.0,
                },
                "2": {
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.7516121544766298, -1.9900000000000002, 1.9352042705802737, -0.9391889804531905, 0.2560644771647802, 0.0622433374241618, -0.13781447384058654, 0.07253115151545006, 0.007170716680246397, -0.020358308641562968, -0.009300247208301833, 0.02580573838921365, -0.02819064606791724, 0.02233397470094777, -0.006095696153540007, -0.0026071488754044417, 0.0036733424820162637, -0.025602299378631815, 0.036947053112735856, -0.007752460082591457, -0.017146474427837174, 0.01542945596577327, -0.00021144883384914472, -0.013632865939604102, 0.017421689234526017, -0.012362467979236172, -0.0016104040917614966, 0.01622023982253356, -0.015885057774454616, 0.0051438745487504215],
                        "feedback": [0.9967005270060615, 0.9633584230394913],
                    },
                    "offset": 0.0,
                },
                "3": {
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.4615102116871844, -0.9844153841725033, 0.8617726427003987, -0.7865602449145748, 0.6966284135425775, -0.38845876334383916, 0.29793158268536934, -0.29409701936781474, 0.20869968861210988, -0.10254049942400192, 0.04557950632739816, -0.0189023102296481, 0.016472666176946954, -0.06339552097679621, 0.07537183271094809, 0.023932502404315128, -0.10935043620716622, 0.07656930841518865, -0.01922165930378962, 0.007908023395297123, 0.018280825322080675, -0.09818188557009226, 0.1396180303557303, -0.07665883440172865, 0.025040434050241707, -0.07035734178615066, 0.11050746321359928, -0.07044311107294293, 0.017851054460900014, -0.0010647441292200319],
                        "feedback": [0.9979573425801123, 0.9730675990337674],
                    },
                    "offset": 0.0,
                },
                "4": {
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.7282743233046902, -1.471131009932805, 0.8496962507290762, -0.022380144883349558, -0.1844587784789683, 0.18097801315018458, -0.10569973585271662, 0.01139040408765742, 0.03910721019202143, -0.005248985025044176, -0.05110632779021178, 0.05371329375151368, -0.02381837653035794, -0.004868089050569556, 0.037269884282889076, -0.062231724409859704, 0.051949994736028314, -0.023745429810177455, -0.005616786010352695, 0.0280981651278796, -0.03261722284368266, 0.025263321836419175, -0.006843487643150184, -0.0038878100767963373, -0.006896053134760072, 0.011929893011419383, 0.002618162155459523, -0.010401845109661835, 0.003455397632138337, 0.00045243533493845956],
                        "feedback": [0.9662288085546595, 0.8077992061763967],
                    },
                    "offset": 0.0,
                },
                "5": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "8": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "9": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
                "10": {
                    "delay": 0,
                    "shareable": False,
                    "offset": 0.0,
                },
            },
        },
    },
    "elements": {
        "qubitC1.xy": {
            "operations": {
                "x180_DragCosine": "qubitC1.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC1.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC1.xy.x180_Square.pulse",
                "x90_Square": "qubitC1.xy.x90_Square.pulse",
                "-x90_Square": "qubitC1.xy.-x90_Square.pulse",
                "y180_Square": "qubitC1.xy.y180_Square.pulse",
                "y90_Square": "qubitC1.xy.y90_Square.pulse",
                "-y90_Square": "qubitC1.xy.-y90_Square.pulse",
                "x180": "qubitC1.xy.x180_DragCosine.pulse",
                "x90": "qubitC1.xy.x90_DragCosine.pulse",
                "-x90": "qubitC1.xy.-x90_DragCosine.pulse",
                "y180": "qubitC1.xy.y180_DragCosine.pulse",
                "y90": "qubitC1.xy.y90_DragCosine.pulse",
                "-y90": "qubitC1.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC1.xy.saturation.pulse",
                "EF_x180": "qubitC1.xy.EF_x180.pulse",
            },
            "intermediate_frequency": -161579893.20423436,
            "RF_inputs": {
                "port": ('oct1', 2),
            },
        },
        "qubitC1.z": {
            "operations": {
                "const": "qubitC1.z.const.pulse",
                "z0": "qubitC1.z.z0.pulse",
                "z90": "qubitC1.z.z90.pulse",
                "z180": "qubitC1.z.z180.pulse",
                "-z90": "qubitC1.z.-z90.pulse",
            },
            "singleInput": {
                "port": ('con2', 1),
            },
        },
        "qubitC1.resonator": {
            "operations": {
                "readout": "qubitC1.resonator.readout.pulse",
                "const": "qubitC1.resonator.const.pulse",
            },
            "intermediate_frequency": -193703946.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct1', 1),
            },
            "RF_inputs": {
                "port": ('oct1', 1),
            },
        },
        "qubitC2.xy": {
            "operations": {
                "x180_DragCosine": "qubitC2.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC2.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC2.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC2.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC2.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC2.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC2.xy.x180_Square.pulse",
                "x90_Square": "qubitC2.xy.x90_Square.pulse",
                "-x90_Square": "qubitC2.xy.-x90_Square.pulse",
                "y180_Square": "qubitC2.xy.y180_Square.pulse",
                "y90_Square": "qubitC2.xy.y90_Square.pulse",
                "-y90_Square": "qubitC2.xy.-y90_Square.pulse",
                "x180": "qubitC2.xy.x180_Square.pulse",
                "x90": "qubitC2.xy.x90_Square.pulse",
                "-x90": "qubitC2.xy.-x90_DragCosine.pulse",
                "y180": "qubitC2.xy.y180_DragCosine.pulse",
                "y90": "qubitC2.xy.y90_DragCosine.pulse",
                "-y90": "qubitC2.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC2.xy.saturation.pulse",
                "EF_x180": "qubitC2.xy.EF_x180.pulse",
            },
            "intermediate_frequency": -102724629.50200598,
            "RF_inputs": {
                "port": ('oct1', 4),
            },
        },
        "qubitC2.z": {
            "operations": {
                "const": "qubitC2.z.const.pulse",
                "z0": "qubitC2.z.z0.pulse",
                "z90": "qubitC2.z.z90.pulse",
                "z180": "qubitC2.z.z180.pulse",
                "-z90": "qubitC2.z.-z90.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC4": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse",
                "Cz.CZ_snz_qubitC4": "qubitC2.z.Cz.CZ_snz_qubitC4.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC1": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse",
                "Cz_SNZ.CZ_snz_qubitC1": "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.pulse",
            },
            "singleInput": {
                "port": ('con2', 2),
            },
        },
        "qubitC2.resonator": {
            "operations": {
                "readout": "qubitC2.resonator.readout.pulse",
                "const": "qubitC2.resonator.const.pulse",
            },
            "intermediate_frequency": 66883846.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct1', 1),
            },
            "RF_inputs": {
                "port": ('oct1', 1),
            },
        },
        "qubitC3.xy": {
            "operations": {
                "x180_DragCosine": "qubitC3.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC3.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC3.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC3.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC3.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC3.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC3.xy.x180_Square.pulse",
                "x90_Square": "qubitC3.xy.x90_Square.pulse",
                "-x90_Square": "qubitC3.xy.-x90_Square.pulse",
                "y180_Square": "qubitC3.xy.y180_Square.pulse",
                "y90_Square": "qubitC3.xy.y90_Square.pulse",
                "-y90_Square": "qubitC3.xy.-y90_Square.pulse",
                "x180": "qubitC3.xy.x180_DragCosine.pulse",
                "x90": "qubitC3.xy.x90_DragCosine.pulse",
                "-x90": "qubitC3.xy.-x90_DragCosine.pulse",
                "y180": "qubitC3.xy.y180_DragCosine.pulse",
                "y90": "qubitC3.xy.y90_DragCosine.pulse",
                "-y90": "qubitC3.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC3.xy.saturation.pulse",
                "EF_x180": "qubitC3.xy.EF_x180.pulse",
            },
            "intermediate_frequency": -235169505.05035493,
            "RF_inputs": {
                "port": ('oct1', 5),
            },
        },
        "qubitC3.z": {
            "operations": {
                "const": "qubitC3.z.const.pulse",
                "z0": "qubitC3.z.z0.pulse",
                "z90": "qubitC3.z.z90.pulse",
                "z180": "qubitC3.z.z180.pulse",
                "-z90": "qubitC3.z.-z90.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC4": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse",
                "Cz.CZ_snz_qubitC4": "qubitC3.z.Cz.CZ_snz_qubitC4.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC1": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse",
                "Cz.CZ_snz_qubitC1": "qubitC3.z.Cz.CZ_snz_qubitC1.pulse",
            },
            "singleInput": {
                "port": ('con2', 3),
            },
        },
        "qubitC3.resonator": {
            "operations": {
                "readout": "qubitC3.resonator.readout.pulse",
                "const": "qubitC3.resonator.const.pulse",
            },
            "intermediate_frequency": 173982913.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct2', 1),
            },
            "RF_inputs": {
                "port": ('oct2', 1),
            },
        },
        "qubitC4.xy": {
            "operations": {
                "x180_DragCosine": "qubitC4.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC4.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC4.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC4.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC4.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC4.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC4.xy.x180_Square.pulse",
                "x90_Square": "qubitC4.xy.x90_Square.pulse",
                "-x90_Square": "qubitC4.xy.-x90_Square.pulse",
                "y180_Square": "qubitC4.xy.y180_Square.pulse",
                "y90_Square": "qubitC4.xy.y90_Square.pulse",
                "-y90_Square": "qubitC4.xy.-y90_Square.pulse",
                "x180": "qubitC4.xy.x180_DragCosine.pulse",
                "x90": "qubitC4.xy.x90_DragCosine.pulse",
                "-x90": "qubitC4.xy.-x90_DragCosine.pulse",
                "y180": "qubitC4.xy.y180_DragCosine.pulse",
                "y90": "qubitC4.xy.y90_DragCosine.pulse",
                "-y90": "qubitC4.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC4.xy.saturation.pulse",
                "EF_x180": "qubitC4.xy.EF_x180.pulse",
            },
            "intermediate_frequency": -115634512.22316144,
            "RF_inputs": {
                "port": ('oct1', 3),
            },
        },
        "qubitC4.z": {
            "operations": {
                "const": "qubitC4.z.const.pulse",
                "z0": "qubitC4.z.z0.pulse",
                "z90": "qubitC4.z.z90.pulse",
                "z180": "qubitC4.z.z180.pulse",
                "-z90": "qubitC4.z.-z90.pulse",
            },
            "singleInput": {
                "port": ('con2', 4),
            },
        },
        "qubitC4.resonator": {
            "operations": {
                "readout": "qubitC4.resonator.readout.pulse",
                "const": "qubitC4.resonator.const.pulse",
            },
            "intermediate_frequency": -85478390.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct2', 1),
            },
            "RF_inputs": {
                "port": ('oct2', 1),
            },
        },
        "qubitC5.z": {
            "operations": {
                "const": "qubitC5.z.const.pulse",
            },
            "singleInput": {
                "port": ('con2', 5),
            },
        },
        "qubitC5.resonator": {
            "operations": {
                "readout": "qubitC5.resonator.readout.pulse",
                "const": "qubitC5.resonator.const.pulse",
            },
            "intermediate_frequency": 315375522.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct2', 1),
            },
            "RF_inputs": {
                "port": ('oct2', 1),
            },
        },
        "qubitB2.z": {
            "operations": {
                "const": "qubitB2.z.const.pulse",
            },
            "singleInput": {
                "port": ('con2', 2),
            },
        },
        "qubitB2.resonator": {
            "operations": {
                "readout": "qubitB2.resonator.readout.pulse",
                "const": "qubitB2.resonator.const.pulse",
            },
            "intermediate_frequency": -176200000.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct1', 1),
            },
            "RF_inputs": {
                "port": ('oct1', 1),
            },
        },
        "qubitB4.z": {
            "operations": {
                "const": "qubitB4.z.const.pulse",
            },
            "singleInput": {
                "port": ('con2', 9),
            },
        },
        "qubitB4.resonator": {
            "operations": {
                "readout": "qubitB4.resonator.readout.pulse",
                "const": "qubitB4.resonator.const.pulse",
            },
            "intermediate_frequency": 137955078.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct1', 1),
            },
            "RF_inputs": {
                "port": ('oct1', 1),
            },
        },
        "qubitB5.z": {
            "operations": {
                "const": "qubitB5.z.const.pulse",
            },
            "singleInput": {
                "port": ('con2', 10),
            },
        },
        "qubitB5.resonator": {
            "operations": {
                "readout": "qubitB5.resonator.readout.pulse",
                "const": "qubitB5.resonator.const.pulse",
            },
            "intermediate_frequency": 64300000.0,
            "smearing": 0,
            "time_of_flight": 264,
            "RF_outputs": {
                "port": ('oct1', 1),
            },
            "RF_inputs": {
                "port": ('oct1', 1),
            },
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "qubitC1.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.x180_DragCosine.wf.I",
                "Q": "qubitC1.xy.x180_DragCosine.wf.Q",
            },
        },
        "qubitC1.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.x90_DragCosine.wf.I",
                "Q": "qubitC1.xy.x90_DragCosine.wf.Q",
            },
        },
        "qubitC1.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC1.xy.-x90_DragCosine.wf.Q",
            },
        },
        "qubitC1.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.y180_DragCosine.wf.I",
                "Q": "qubitC1.xy.y180_DragCosine.wf.Q",
            },
        },
        "qubitC1.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.y90_DragCosine.wf.I",
                "Q": "qubitC1.xy.y90_DragCosine.wf.Q",
            },
        },
        "qubitC1.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC1.xy.-y90_DragCosine.wf.Q",
            },
        },
        "qubitC1.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.x180_Square.wf.I",
                "Q": "qubitC1.xy.x180_Square.wf.Q",
            },
        },
        "qubitC1.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.x90_Square.wf.I",
                "Q": "qubitC1.xy.x90_Square.wf.Q",
            },
        },
        "qubitC1.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.-x90_Square.wf.I",
                "Q": "qubitC1.xy.-x90_Square.wf.Q",
            },
        },
        "qubitC1.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.y180_Square.wf.I",
                "Q": "qubitC1.xy.y180_Square.wf.Q",
            },
        },
        "qubitC1.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.y90_Square.wf.I",
                "Q": "qubitC1.xy.y90_Square.wf.Q",
            },
        },
        "qubitC1.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.-y90_Square.wf.I",
                "Q": "qubitC1.xy.-y90_Square.wf.Q",
            },
        },
        "qubitC1.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.saturation.wf.I",
                "Q": "qubitC1.xy.saturation.wf.Q",
            },
        },
        "qubitC1.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.xy.EF_x180.wf.I",
                "Q": "qubitC1.xy.EF_x180.wf.Q",
            },
        },
        "qubitC1.z.const.pulse": {
            "operation": "control",
            "length": 60,
            "waveforms": {
                "single": "qubitC1.z.const.wf",
            },
        },
        "qubitC1.z.z0.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.z0.wf",
            },
        },
        "qubitC1.z.z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.z90.wf",
            },
        },
        "qubitC1.z.z180.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.z180.wf",
            },
        },
        "qubitC1.z.-z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.-z90.wf",
            },
        },
        "qubitC1.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC1.resonator.readout.wf.I",
                "Q": "qubitC1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC1.resonator.readout.iw1",
                "iw2": "qubitC1.resonator.readout.iw2",
                "iw3": "qubitC1.resonator.readout.iw3",
            },
        },
        "qubitC1.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitC1.resonator.const.wf.I",
                "Q": "qubitC1.resonator.const.wf.Q",
            },
        },
        "qubitC2.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.x180_DragCosine.wf.I",
                "Q": "qubitC2.xy.x180_DragCosine.wf.Q",
            },
        },
        "qubitC2.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.x90_DragCosine.wf.I",
                "Q": "qubitC2.xy.x90_DragCosine.wf.Q",
            },
        },
        "qubitC2.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC2.xy.-x90_DragCosine.wf.Q",
            },
        },
        "qubitC2.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.y180_DragCosine.wf.I",
                "Q": "qubitC2.xy.y180_DragCosine.wf.Q",
            },
        },
        "qubitC2.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.y90_DragCosine.wf.I",
                "Q": "qubitC2.xy.y90_DragCosine.wf.Q",
            },
        },
        "qubitC2.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC2.xy.-y90_DragCosine.wf.Q",
            },
        },
        "qubitC2.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.x180_Square.wf.I",
                "Q": "qubitC2.xy.x180_Square.wf.Q",
            },
        },
        "qubitC2.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.x90_Square.wf.I",
                "Q": "qubitC2.xy.x90_Square.wf.Q",
            },
        },
        "qubitC2.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.-x90_Square.wf.I",
                "Q": "qubitC2.xy.-x90_Square.wf.Q",
            },
        },
        "qubitC2.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.y180_Square.wf.I",
                "Q": "qubitC2.xy.y180_Square.wf.Q",
            },
        },
        "qubitC2.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.y90_Square.wf.I",
                "Q": "qubitC2.xy.y90_Square.wf.Q",
            },
        },
        "qubitC2.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.-y90_Square.wf.I",
                "Q": "qubitC2.xy.-y90_Square.wf.Q",
            },
        },
        "qubitC2.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.saturation.wf.I",
                "Q": "qubitC2.xy.saturation.wf.Q",
            },
        },
        "qubitC2.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.xy.EF_x180.wf.I",
                "Q": "qubitC2.xy.EF_x180.wf.Q",
            },
        },
        "qubitC2.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "qubitC2.z.const.wf",
            },
        },
        "qubitC2.z.z0.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.z0.wf",
            },
        },
        "qubitC2.z.z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.z90.wf",
            },
        },
        "qubitC2.z.z180.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.z180.wf",
            },
        },
        "qubitC2.z.-z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.-z90.wf",
            },
        },
        "qubitC2.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC2.resonator.readout.wf.I",
                "Q": "qubitC2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC2.resonator.readout.iw1",
                "iw2": "qubitC2.resonator.readout.iw2",
                "iw3": "qubitC2.resonator.readout.iw3",
            },
        },
        "qubitC2.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitC2.resonator.const.wf.I",
                "Q": "qubitC2.resonator.const.wf.Q",
            },
        },
        "qubitC3.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.x180_DragCosine.wf.I",
                "Q": "qubitC3.xy.x180_DragCosine.wf.Q",
            },
        },
        "qubitC3.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.x90_DragCosine.wf.I",
                "Q": "qubitC3.xy.x90_DragCosine.wf.Q",
            },
        },
        "qubitC3.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC3.xy.-x90_DragCosine.wf.Q",
            },
        },
        "qubitC3.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.y180_DragCosine.wf.I",
                "Q": "qubitC3.xy.y180_DragCosine.wf.Q",
            },
        },
        "qubitC3.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.y90_DragCosine.wf.I",
                "Q": "qubitC3.xy.y90_DragCosine.wf.Q",
            },
        },
        "qubitC3.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC3.xy.-y90_DragCosine.wf.Q",
            },
        },
        "qubitC3.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.x180_Square.wf.I",
                "Q": "qubitC3.xy.x180_Square.wf.Q",
            },
        },
        "qubitC3.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.x90_Square.wf.I",
                "Q": "qubitC3.xy.x90_Square.wf.Q",
            },
        },
        "qubitC3.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.-x90_Square.wf.I",
                "Q": "qubitC3.xy.-x90_Square.wf.Q",
            },
        },
        "qubitC3.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.y180_Square.wf.I",
                "Q": "qubitC3.xy.y180_Square.wf.Q",
            },
        },
        "qubitC3.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.y90_Square.wf.I",
                "Q": "qubitC3.xy.y90_Square.wf.Q",
            },
        },
        "qubitC3.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.-y90_Square.wf.I",
                "Q": "qubitC3.xy.-y90_Square.wf.Q",
            },
        },
        "qubitC3.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.saturation.wf.I",
                "Q": "qubitC3.xy.saturation.wf.Q",
            },
        },
        "qubitC3.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.xy.EF_x180.wf.I",
                "Q": "qubitC3.xy.EF_x180.wf.Q",
            },
        },
        "qubitC3.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "qubitC3.z.const.wf",
            },
        },
        "qubitC3.z.z0.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.z0.wf",
            },
        },
        "qubitC3.z.z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.z90.wf",
            },
        },
        "qubitC3.z.z180.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.z180.wf",
            },
        },
        "qubitC3.z.-z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.-z90.wf",
            },
        },
        "qubitC3.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC3.resonator.readout.wf.I",
                "Q": "qubitC3.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC3.resonator.readout.iw1",
                "iw2": "qubitC3.resonator.readout.iw2",
                "iw3": "qubitC3.resonator.readout.iw3",
            },
        },
        "qubitC3.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitC3.resonator.const.wf.I",
                "Q": "qubitC3.resonator.const.wf.Q",
            },
        },
        "qubitC4.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.x180_DragCosine.wf.I",
                "Q": "qubitC4.xy.x180_DragCosine.wf.Q",
            },
        },
        "qubitC4.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.x90_DragCosine.wf.I",
                "Q": "qubitC4.xy.x90_DragCosine.wf.Q",
            },
        },
        "qubitC4.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC4.xy.-x90_DragCosine.wf.Q",
            },
        },
        "qubitC4.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.y180_DragCosine.wf.I",
                "Q": "qubitC4.xy.y180_DragCosine.wf.Q",
            },
        },
        "qubitC4.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.y90_DragCosine.wf.I",
                "Q": "qubitC4.xy.y90_DragCosine.wf.Q",
            },
        },
        "qubitC4.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC4.xy.-y90_DragCosine.wf.Q",
            },
        },
        "qubitC4.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.x180_Square.wf.I",
                "Q": "qubitC4.xy.x180_Square.wf.Q",
            },
        },
        "qubitC4.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.x90_Square.wf.I",
                "Q": "qubitC4.xy.x90_Square.wf.Q",
            },
        },
        "qubitC4.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.-x90_Square.wf.I",
                "Q": "qubitC4.xy.-x90_Square.wf.Q",
            },
        },
        "qubitC4.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.y180_Square.wf.I",
                "Q": "qubitC4.xy.y180_Square.wf.Q",
            },
        },
        "qubitC4.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.y90_Square.wf.I",
                "Q": "qubitC4.xy.y90_Square.wf.Q",
            },
        },
        "qubitC4.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 100,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.-y90_Square.wf.I",
                "Q": "qubitC4.xy.-y90_Square.wf.Q",
            },
        },
        "qubitC4.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.saturation.wf.I",
                "Q": "qubitC4.xy.saturation.wf.Q",
            },
        },
        "qubitC4.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 32,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.xy.EF_x180.wf.I",
                "Q": "qubitC4.xy.EF_x180.wf.Q",
            },
        },
        "qubitC4.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "qubitC4.z.const.wf",
            },
        },
        "qubitC4.z.z0.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.z0.wf",
            },
        },
        "qubitC4.z.z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.z90.wf",
            },
        },
        "qubitC4.z.z180.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.z180.wf",
            },
        },
        "qubitC4.z.-z90.pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.-z90.wf",
            },
        },
        "qubitC4.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC4.resonator.readout.wf.I",
                "Q": "qubitC4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC4.resonator.readout.iw1",
                "iw2": "qubitC4.resonator.readout.iw2",
                "iw3": "qubitC4.resonator.readout.iw3",
            },
        },
        "qubitC4.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitC4.resonator.const.wf.I",
                "Q": "qubitC4.resonator.const.wf.Q",
            },
        },
        "qubitC5.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "qubitC5.z.const.wf",
            },
        },
        "qubitC5.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitC5.resonator.readout.wf.I",
                "Q": "qubitC5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC5.resonator.readout.iw1",
                "iw2": "qubitC5.resonator.readout.iw2",
                "iw3": "qubitC5.resonator.readout.iw3",
            },
        },
        "qubitC5.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitC5.resonator.const.wf.I",
                "Q": "qubitC5.resonator.const.wf.Q",
            },
        },
        "qubitB2.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "qubitB2.z.const.wf",
            },
        },
        "qubitB2.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitB2.resonator.readout.wf.I",
                "Q": "qubitB2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitB2.resonator.readout.iw1",
                "iw2": "qubitB2.resonator.readout.iw2",
                "iw3": "qubitB2.resonator.readout.iw3",
            },
        },
        "qubitB2.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitB2.resonator.const.wf.I",
                "Q": "qubitB2.resonator.const.wf.Q",
            },
        },
        "qubitB4.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "qubitB4.z.const.wf",
            },
        },
        "qubitB4.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitB4.resonator.readout.wf.I",
                "Q": "qubitB4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitB4.resonator.readout.iw1",
                "iw2": "qubitB4.resonator.readout.iw2",
                "iw3": "qubitB4.resonator.readout.iw3",
            },
        },
        "qubitB4.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitB4.resonator.const.wf.I",
                "Q": "qubitB4.resonator.const.wf.Q",
            },
        },
        "qubitB5.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "qubitB5.z.const.wf",
            },
        },
        "qubitB5.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1500,
            "digital_marker": "ON",
            "waveforms": {
                "I": "qubitB5.resonator.readout.wf.I",
                "Q": "qubitB5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitB5.resonator.readout.iw1",
                "iw2": "qubitB5.resonator.readout.iw2",
                "iw3": "qubitB5.resonator.readout.iw3",
            },
        },
        "qubitB5.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "qubitB5.resonator.const.wf.I",
                "Q": "qubitB5.resonator.const.wf.Q",
            },
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse": {
            "operation": "control",
            "length": 52,
            "waveforms": {
                "single": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.wf",
            },
        },
        "qubitC2.z.Cz.CZ_snz_qubitC4.pulse": {
            "operation": "control",
            "length": 60,
            "waveforms": {
                "single": "qubitC2.z.Cz.CZ_snz_qubitC4.wf",
            },
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse": {
            "operation": "control",
            "length": 52,
            "waveforms": {
                "single": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.wf",
            },
        },
        "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.pulse": {
            "operation": "control",
            "length": 60,
            "waveforms": {
                "single": "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.wf",
            },
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse": {
            "operation": "control",
            "length": 44,
            "waveforms": {
                "single": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.wf",
            },
        },
        "qubitC3.z.Cz.CZ_snz_qubitC4.pulse": {
            "operation": "control",
            "length": 52,
            "waveforms": {
                "single": "qubitC3.z.Cz.CZ_snz_qubitC4.wf",
            },
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse": {
            "operation": "control",
            "length": 48,
            "waveforms": {
                "single": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.wf",
            },
        },
        "qubitC3.z.Cz.CZ_snz_qubitC1.pulse": {
            "operation": "control",
            "length": 56,
            "waveforms": {
                "single": "qubitC3.z.Cz.CZ_snz_qubitC1.wf",
            },
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "qubitC1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0012096453125145397), np.float64(0.004789058228836524), np.float64(0.010591697163608568), np.float64(0.018380001397059975), np.float64(0.027835116838786374), np.float64(0.038569949951674565), np.float64(0.050145015406793586), np.float64(0.06208642866439842), np.float64(0.07390530686266622), np.float64(0.08511778373950074), np.float64(0.09526481917415419), np.float64(0.10393099234370592), np.float64(0.1107615091003623), np.float64(0.11547672728555419)] + [0.11788360531276522] * 2 + [np.float64(0.11547672728555419), np.float64(0.1107615091003623), np.float64(0.10393099234370594), np.float64(0.0952648191741542), np.float64(0.08511778373950073), np.float64(0.07390530686266628), np.float64(0.06208642866439847), np.float64(0.050145015406793586), np.float64(0.038569949951674565), np.float64(0.027835116838786374), np.float64(0.018380001397059982), np.float64(0.010591697163608568), np.float64(0.004789058228836524), np.float64(0.0012096453125145527), np.float64(0.0)],
        },
        "qubitC1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0013361039447842313), np.float64(-0.0026175076370834487), np.float64(-0.0037917502591763813), np.float64(-0.004810758180146893), np.float64(-0.005632813095982107), np.float64(-0.006224259981840376), np.float64(-0.006560884932722606), np.float64(-0.0066289064835879236), np.float64(-0.006425539824151712), np.float64(-0.005959110809345857), np.float64(-0.005248715097839595), np.float64(-0.004323436373529937), np.float64(-0.0032211556561057616), np.float64(-0.0019870004476508973), np.float64(-0.000671497207406568), np.float64(0.0006714972074065635), np.float64(0.001987000447650898), np.float64(0.00322115565610576), np.float64(0.004323436373529936), np.float64(0.005248715097839593), np.float64(0.005959110809345858), np.float64(0.006425539824151711), np.float64(0.0066289064835879236), np.float64(0.006560884932722606), np.float64(0.006224259981840376), np.float64(0.005632813095982108), np.float64(0.004810758180146894), np.float64(0.003791750259176382), np.float64(0.0026175076370834496), np.float64(0.0013361039447842384), np.float64(1.6257003961951508e-18)],
        },
        "qubitC1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006027057769603683), np.float64(0.0023861482625177934), np.float64(0.00527731311176796), np.float64(0.009157835696085116), np.float64(0.013868846964925286), np.float64(0.019217477563421816), np.float64(0.024984753926434858), np.float64(0.030934563082036456), np.float64(0.03682331914432338), np.float64(0.042409935748206166), np.float64(0.04746569615352224), np.float64(0.05178361693525138), np.float64(0.055186921909255414), np.float64(0.05753627937002727)] + [0.05873550634708516] * 2 + [np.float64(0.05753627937002727), np.float64(0.055186921909255414), np.float64(0.05178361693525139), np.float64(0.047465696153522245), np.float64(0.04240993574820616), np.float64(0.03682331914432341), np.float64(0.030934563082036484), np.float64(0.024984753926434858), np.float64(0.019217477563421816), np.float64(0.013868846964925286), np.float64(0.00915783569608512), np.float64(0.00527731311176796), np.float64(0.0023861482625177934), np.float64(0.0006027057769603749), np.float64(0.0)],
        },
        "qubitC1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0006657137904887421), np.float64(-0.001304173180176826), np.float64(-0.0018892395666346286), np.float64(-0.0023969602632581853), np.float64(-0.0028065491250730797), np.float64(-0.0031012375359519622), np.float64(-0.0032689609177290325), np.float64(-0.0033028526554476773), np.float64(-0.0032015252173835853), np.float64(-0.0029691269607565684), np.float64(-0.0026151722974985733), np.float64(-0.0021541521731112873), np.float64(-0.0016049408056546927), np.float64(-0.0009900229730420576), np.float64(-0.00033457348359032193), np.float64(0.00033457348359031965), np.float64(0.0009900229730420583), np.float64(0.0016049408056546923), np.float64(0.002154152173111287), np.float64(0.002615172297498573), np.float64(0.0029691269607565684), np.float64(0.0032015252173835844), np.float64(0.0033028526554476773), np.float64(0.0032689609177290325), np.float64(0.0031012375359519622), np.float64(0.00280654912507308), np.float64(0.002396960263258186), np.float64(0.001889239566634629), np.float64(0.0013041731801768266), np.float64(0.0006657137904887456), np.float64(8.100052224042324e-19)],
        },
        "qubitC1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0006027057769603682), np.float64(-0.0023861482625177934), np.float64(-0.00527731311176796), np.float64(-0.009157835696085116), np.float64(-0.013868846964925286), np.float64(-0.019217477563421816), np.float64(-0.024984753926434858), np.float64(-0.030934563082036456), np.float64(-0.03682331914432338), np.float64(-0.042409935748206166), np.float64(-0.04746569615352224), np.float64(-0.05178361693525138), np.float64(-0.055186921909255414), np.float64(-0.05753627937002727)] + [-0.05873550634708516] * 2 + [np.float64(-0.05753627937002727), np.float64(-0.055186921909255414), np.float64(-0.05178361693525139), np.float64(-0.047465696153522245), np.float64(-0.04240993574820616), np.float64(-0.03682331914432341), np.float64(-0.030934563082036484), np.float64(-0.024984753926434858), np.float64(-0.019217477563421816), np.float64(-0.013868846964925286), np.float64(-0.00915783569608512), np.float64(-0.00527731311176796), np.float64(-0.0023861482625177934), np.float64(-0.000602705776960375), np.float64(-9.919703029099832e-35)],
        },
        "qubitC1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006657137904887422), np.float64(0.0013041731801768262), np.float64(0.0018892395666346292), np.float64(0.0023969602632581866), np.float64(0.0028065491250730814), np.float64(0.0031012375359519644), np.float64(0.0032689609177290356), np.float64(0.003302852655447681), np.float64(0.0032015252173835896), np.float64(0.0029691269607565736), np.float64(0.002615172297498579), np.float64(0.002154152173111294), np.float64(0.0016049408056546994), np.float64(0.0009900229730420646), np.float64(0.00033457348359032914), np.float64(-0.00033457348359031244), np.float64(-0.0009900229730420514), np.float64(-0.0016049408056546856), np.float64(-0.0021541521731112804), np.float64(-0.0026151722974985673), np.float64(-0.002969126960756563), np.float64(-0.00320152521738358), np.float64(-0.0033028526554476734), np.float64(-0.0032689609177290295), np.float64(-0.00310123753595196), np.float64(-0.0028065491250730784), np.float64(-0.002396960263258185), np.float64(-0.0018892395666346283), np.float64(-0.0013041731801768264), np.float64(-0.0006657137904887455), np.float64(-8.100052224042324e-19)],
        },
        "qubitC1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0013361039447842313), np.float64(0.002617507637083449), np.float64(0.0037917502591763817), np.float64(0.004810758180146894), np.float64(0.005632813095982109), np.float64(0.006224259981840378), np.float64(0.0065608849327226096), np.float64(0.006628906483587927), np.float64(0.006425539824151717), np.float64(0.005959110809345862), np.float64(0.005248715097839601), np.float64(0.004323436373529943), np.float64(0.0032211556561057685), np.float64(0.0019870004476509042), np.float64(0.0006714972074065753), np.float64(-0.0006714972074065562), np.float64(-0.0019870004476508912), np.float64(-0.003221155656105753), np.float64(-0.00432343637352993), np.float64(-0.005248715097839587), np.float64(-0.005959110809345853), np.float64(-0.006425539824151707), np.float64(-0.00662890648358792), np.float64(-0.006560884932722603), np.float64(-0.006224259981840373), np.float64(-0.005632813095982106), np.float64(-0.004810758180146893), np.float64(-0.0037917502591763817), np.float64(-0.002617507637083449), np.float64(-0.0013361039447842384), np.float64(-1.6257003961951508e-18)],
        },
        "qubitC1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0012096453125145397), np.float64(0.004789058228836524), np.float64(0.010591697163608568), np.float64(0.018380001397059975), np.float64(0.027835116838786374), np.float64(0.038569949951674565), np.float64(0.050145015406793586), np.float64(0.06208642866439842), np.float64(0.07390530686266622), np.float64(0.08511778373950074), np.float64(0.09526481917415419), np.float64(0.10393099234370592), np.float64(0.1107615091003623), np.float64(0.11547672728555419)] + [0.11788360531276522] * 2 + [np.float64(0.11547672728555419), np.float64(0.1107615091003623), np.float64(0.10393099234370594), np.float64(0.0952648191741542), np.float64(0.08511778373950073), np.float64(0.07390530686266628), np.float64(0.06208642866439847), np.float64(0.050145015406793586), np.float64(0.038569949951674565), np.float64(0.027835116838786374), np.float64(0.018380001397059982), np.float64(0.010591697163608568), np.float64(0.004789058228836524), np.float64(0.0012096453125145527), np.float64(9.954543932864876e-35)],
        },
        "qubitC1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006657137904887421), np.float64(0.0013041731801768262), np.float64(0.0018892395666346288), np.float64(0.0023969602632581857), np.float64(0.0028065491250730806), np.float64(0.0031012375359519635), np.float64(0.0032689609177290343), np.float64(0.003302852655447679), np.float64(0.0032015252173835875), np.float64(0.002969126960756571), np.float64(0.0026151722974985764), np.float64(0.0021541521731112904), np.float64(0.0016049408056546962), np.float64(0.0009900229730420611), np.float64(0.0003345734835903255), np.float64(-0.0003345734835903161), np.float64(-0.0009900229730420548), np.float64(-0.0016049408056546888), np.float64(-0.002154152173111284), np.float64(-0.00261517229749857), np.float64(-0.0029691269607565658), np.float64(-0.0032015252173835823), np.float64(-0.0033028526554476755), np.float64(-0.003268960917729031), np.float64(-0.003101237535951961), np.float64(-0.0028065491250730793), np.float64(-0.0023969602632581857), np.float64(-0.0018892395666346288), np.float64(-0.0013041731801768264), np.float64(-0.0006657137904887456), np.float64(-8.100052224042324e-19)],
        },
        "qubitC1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006027057769603683), np.float64(0.0023861482625177934), np.float64(0.00527731311176796), np.float64(0.009157835696085116), np.float64(0.013868846964925286), np.float64(0.019217477563421816), np.float64(0.024984753926434858), np.float64(0.030934563082036456), np.float64(0.03682331914432338), np.float64(0.042409935748206166), np.float64(0.04746569615352224), np.float64(0.05178361693525138), np.float64(0.055186921909255414), np.float64(0.05753627937002727)] + [0.05873550634708516] * 2 + [np.float64(0.05753627937002727), np.float64(0.055186921909255414), np.float64(0.05178361693525139), np.float64(0.047465696153522245), np.float64(0.04240993574820616), np.float64(0.03682331914432341), np.float64(0.030934563082036484), np.float64(0.024984753926434858), np.float64(0.019217477563421816), np.float64(0.013868846964925286), np.float64(0.00915783569608512), np.float64(0.00527731311176796), np.float64(0.0023861482625177934), np.float64(0.0006027057769603749), np.float64(4.959851514549916e-35)],
        },
        "qubitC1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0006657137904887421), np.float64(-0.0013041731801768258), np.float64(-0.0018892395666346283), np.float64(-0.002396960263258185), np.float64(-0.002806549125073079), np.float64(-0.003101237535951961), np.float64(-0.003268960917729031), np.float64(-0.0033028526554476755), np.float64(-0.003201525217383583), np.float64(-0.0029691269607565658), np.float64(-0.0026151722974985703), np.float64(-0.0021541521731112843), np.float64(-0.0016049408056546892), np.float64(-0.0009900229730420542), np.float64(-0.00033457348359031835), np.float64(0.00033457348359032323), np.float64(0.0009900229730420618), np.float64(0.0016049408056546957), np.float64(0.00215415217311129), np.float64(0.002615172297498576), np.float64(0.002969126960756571), np.float64(0.0032015252173835866), np.float64(0.003302852655447679), np.float64(0.0032689609177290343), np.float64(0.0031012375359519635), np.float64(0.002806549125073081), np.float64(0.0023969602632581866), np.float64(0.0018892395666346292), np.float64(0.0013041731801768269), np.float64(0.0006657137904887456), np.float64(8.100052224042324e-19)],
        },
        "qubitC1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0006027057769603683), np.float64(-0.0023861482625177934), np.float64(-0.00527731311176796), np.float64(-0.009157835696085116), np.float64(-0.013868846964925286), np.float64(-0.019217477563421816), np.float64(-0.024984753926434858), np.float64(-0.030934563082036456), np.float64(-0.03682331914432338), np.float64(-0.042409935748206166), np.float64(-0.04746569615352224), np.float64(-0.05178361693525138), np.float64(-0.055186921909255414), np.float64(-0.05753627937002727)] + [-0.05873550634708516] * 2 + [np.float64(-0.05753627937002727), np.float64(-0.055186921909255414), np.float64(-0.05178361693525139), np.float64(-0.047465696153522245), np.float64(-0.04240993574820616), np.float64(-0.03682331914432341), np.float64(-0.030934563082036484), np.float64(-0.024984753926434858), np.float64(-0.019217477563421816), np.float64(-0.013868846964925286), np.float64(-0.00915783569608512), np.float64(-0.00527731311176796), np.float64(-0.0023861482625177934), np.float64(-0.0006027057769603749), np.float64(4.959851514549916e-35)],
        },
        "qubitC1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.03066287395078562,
        },
        "qubitC1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0008152112067846226), np.float64(0.0032274699845494777), np.float64(0.0071380181755045), np.float64(0.012386757477242083), np.float64(0.018758803886048423), np.float64(0.02599328507326331), np.float64(0.03379402052906672), np.float64(0.04184164722735058), np.float64(0.04980669438553113), np.float64(0.05736307203709921), np.float64(0.06420142119316287), np.float64(0.07004177903580103), np.float64(0.07464504062872312), np.float64(0.07782274790145467)] + [0.07944480514488529] * 2 + [np.float64(0.07782274790145467), np.float64(0.07464504062872312), np.float64(0.07004177903580104), np.float64(0.06420142119316288), np.float64(0.0573630720370992), np.float64(0.04980669438553117), np.float64(0.04184164722735061), np.float64(0.03379402052906672), np.float64(0.02599328507326331), np.float64(0.018758803886048423), np.float64(0.012386757477242088), np.float64(0.0071380181755045), np.float64(0.0032274699845494777), np.float64(0.0008152112067846313), np.float64(0.0)],
        },
        "qubitC1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0006000019617158705), np.float64(-0.0011754397726218564), np.float64(-0.0017027549411283943), np.float64(-0.002160359122279925), np.float64(-0.0025295179471338967), np.float64(-0.002795118010026468), np.float64(-0.002946285613176134), np.float64(-0.0029768319371485086), np.float64(-0.0028855064118511226), np.float64(-0.002676047915019943), np.float64(-0.0023570317021255694), np.float64(-0.0019415183344067071), np.float64(-0.0014465189779585168), np.float64(-0.0008922989645939421), np.float64(-0.00030154812677821575), np.float64(0.00030154812677821375), np.float64(0.0008922989645939426), np.float64(0.0014465189779585162), np.float64(0.0019415183344067065), np.float64(0.0023570317021255685), np.float64(0.002676047915019943), np.float64(0.0028855064118511217), np.float64(0.0029768319371485086), np.float64(0.002946285613176134), np.float64(0.002795118010026468), np.float64(0.002529517947133897), np.float64(0.0021603591222799255), np.float64(0.0017027549411283945), np.float64(0.0011754397726218568), np.float64(0.0006000019617158738), np.float64(7.300505553382528e-19)],
        },
        "qubitC1.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC1.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC1.z.z90.wf": {
            "type": "constant",
            "sample": 0.018880207703291515,
        },
        "qubitC1.z.z180.wf": {
            "type": "constant",
            "sample": 0.026590269797742615,
        },
        "qubitC1.z.-z90.wf": {
            "type": "constant",
            "sample": 0.032704782390436345,
        },
        "qubitC1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.018244444444444445,
        },
        "qubitC1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0011363473587025924), np.float64(0.00449886724042994), np.float64(0.009949897686144812), np.float64(0.017266272868930756), np.float64(0.026148459529166952), np.float64(0.036232819901370686), np.float64(0.04710649908704191), np.float64(0.0583243273827162), np.float64(0.06942704557992978), np.float64(0.07996010709117475), np.float64(0.08949228714059851), np.float64(0.09763333715699413), np.float64(0.10404996159615755), np.float64(0.10847946309962808)] + [0.1107404973554093] * 2 + [np.float64(0.10847946309962808), np.float64(0.10404996159615755), np.float64(0.09763333715699414), np.float64(0.08949228714059852), np.float64(0.07996010709117474), np.float64(0.06942704557992982), np.float64(0.05832432738271626), np.float64(0.04710649908704191), np.float64(0.036232819901370686), np.float64(0.026148459529166952), np.float64(0.017266272868930763), np.float64(0.009949897686144812), np.float64(0.00449886724042994), np.float64(0.0011363473587026047), np.float64(0.0)],
        },
        "qubitC2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.002210342435467431), np.float64(-0.004330193191922616), np.float64(-0.006272765330324393), np.float64(-0.007958529719084057), np.float64(-0.009318470966056883), np.float64(-0.010296912916805493), np.float64(-0.010853798042904188), np.float64(-0.010966327401859264), np.float64(-0.010629894028493465), np.float64(-0.009858271544641627), np.float64(-0.008683050265454439), np.float64(-0.007152343888184455), np.float64(-0.005328819711767478), np.float64(-0.003287133030241011), np.float64(-0.001110870736234749), np.float64(0.0011108707362347417), np.float64(0.0032871330302410126), np.float64(0.005328819711767476), np.float64(0.0071523438881844524), np.float64(0.008683050265454437), np.float64(0.009858271544641627), np.float64(0.010629894028493464), np.float64(0.010966327401859264), np.float64(0.010853798042904188), np.float64(0.010296912916805493), np.float64(0.009318470966056886), np.float64(0.007958529719084059), np.float64(0.006272765330324395), np.float64(0.004330193191922618), np.float64(0.0022103424354674426), np.float64(2.6894274110137822e-18)],
        },
        "qubitC2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0005656168977942143), np.float64(0.0022393111689239984), np.float64(0.004952561573278572), np.float64(0.008594287320510268), np.float64(0.013015395730642827), np.float64(0.018034886105907224), np.float64(0.02344725992057507), np.float64(0.02903093395474694), np.float64(0.03455731193740998), np.float64(0.03980014330463216), np.float64(0.044544785924232826), np.float64(0.04859699356989374), np.float64(0.05179086838448733), np.float64(0.05399565275783978)] + [0.05512108255865488] * 2 + [np.float64(0.05399565275783978), np.float64(0.05179086838448733), np.float64(0.04859699356989375), np.float64(0.04454478592423283), np.float64(0.039800143304632156), np.float64(0.03455731193741001), np.float64(0.029030933954746964), np.float64(0.02344725992057507), np.float64(0.018034886105907224), np.float64(0.013015395730642827), np.float64(0.008594287320510272), np.float64(0.004952561573278572), np.float64(0.0022393111689239984), np.float64(0.0005656168977942205), np.float64(0.0)],
        },
        "qubitC2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0011001979472539116), np.float64(-0.002155353661279478), np.float64(-0.003122268943168961), np.float64(-0.003961358167674082), np.float64(-0.004638268923354805), np.float64(-0.005125288404339925), np.float64(-0.005402477975855549), np.float64(-0.005458489464275439), np.float64(-0.0052910297526826125), np.float64(-0.00490695466134536), np.float64(-0.004321988269629939), np.float64(-0.003560079170343806), np.float64(-0.0026524200115322573), np.float64(-0.0016361704658024601), np.float64(-0.0005529359089608455), np.float64(0.0005529359089608417), np.float64(0.001636170465802461), np.float64(0.002652420011532256), np.float64(0.003560079170343805), np.float64(0.004321988269629938), np.float64(0.004906954661345361), np.float64(0.005291029752682612), np.float64(0.005458489464275439), np.float64(0.005402477975855549), np.float64(0.005125288404339925), np.float64(0.004638268923354806), np.float64(0.003961358167674083), np.float64(0.0031222689431689614), np.float64(0.0021553536612794787), np.float64(0.0011001979472539175), np.float64(1.3386624938321075e-18)],
        },
        "qubitC2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0005656168977942142), np.float64(-0.002239311168923998), np.float64(-0.004952561573278572), np.float64(-0.008594287320510268), np.float64(-0.013015395730642827), np.float64(-0.018034886105907224), np.float64(-0.02344725992057507), np.float64(-0.02903093395474694), np.float64(-0.03455731193740998), np.float64(-0.03980014330463216), np.float64(-0.044544785924232826), np.float64(-0.04859699356989374), np.float64(-0.05179086838448733), np.float64(-0.05399565275783978)] + [-0.05512108255865488] * 2 + [np.float64(-0.05399565275783978), np.float64(-0.05179086838448733), np.float64(-0.04859699356989375), np.float64(-0.04454478592423283), np.float64(-0.039800143304632156), np.float64(-0.03455731193741001), np.float64(-0.029030933954746964), np.float64(-0.02344725992057507), np.float64(-0.018034886105907224), np.float64(-0.013015395730642827), np.float64(-0.008594287320510272), np.float64(-0.004952561573278572), np.float64(-0.002239311168923999), np.float64(-0.0005656168977942206), np.float64(-1.639388738210104e-34)],
        },
        "qubitC2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0011001979472539116), np.float64(0.0021553536612794783), np.float64(0.0031222689431689614), np.float64(0.003961358167674083), np.float64(0.004638268923354807), np.float64(0.005125288404339927), np.float64(0.005402477975855552), np.float64(0.005458489464275442), np.float64(0.005291029752682617), np.float64(0.0049069546613453655), np.float64(0.004321988269629944), np.float64(0.0035600791703438123), np.float64(0.002652420011532264), np.float64(0.0016361704658024666), np.float64(0.0005529359089608522), np.float64(-0.0005529359089608349), np.float64(-0.0016361704658024545), np.float64(-0.0026524200115322495), np.float64(-0.003560079170343799), np.float64(-0.004321988269629933), np.float64(-0.004906954661345356), np.float64(-0.005291029752682607), np.float64(-0.005458489464275435), np.float64(-0.005402477975855547), np.float64(-0.005125288404339922), np.float64(-0.004638268923354804), np.float64(-0.003961358167674082), np.float64(-0.003122268943168961), np.float64(-0.0021553536612794783), np.float64(-0.0011001979472539175), np.float64(-1.3386624938321075e-18)],
        },
        "qubitC2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.002210342435467431), np.float64(0.004330193191922616), np.float64(0.006272765330324394), np.float64(0.007958529719084059), np.float64(0.009318470966056884), np.float64(0.010296912916805494), np.float64(0.010853798042904192), np.float64(0.010966327401859268), np.float64(0.010629894028493469), np.float64(0.009858271544641632), np.float64(0.008683050265454444), np.float64(0.007152343888184461), np.float64(0.005328819711767484), np.float64(0.0032871330302410174), np.float64(0.0011108707362347558), np.float64(-0.001110870736234735), np.float64(-0.003287133030241006), np.float64(-0.00532881971176747), np.float64(-0.007152343888184446), np.float64(-0.008683050265454432), np.float64(-0.009858271544641622), np.float64(-0.01062989402849346), np.float64(-0.01096632740185926), np.float64(-0.010853798042904185), np.float64(-0.010296912916805491), np.float64(-0.009318470966056884), np.float64(-0.007958529719084057), np.float64(-0.006272765330324394), np.float64(-0.004330193191922618), np.float64(-0.0022103424354674426), np.float64(-2.6894274110137822e-18)],
        },
        "qubitC2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0011363473587025922), np.float64(0.00449886724042994), np.float64(0.009949897686144812), np.float64(0.017266272868930756), np.float64(0.026148459529166952), np.float64(0.036232819901370686), np.float64(0.04710649908704191), np.float64(0.0583243273827162), np.float64(0.06942704557992978), np.float64(0.07996010709117475), np.float64(0.08949228714059851), np.float64(0.09763333715699413), np.float64(0.10404996159615755), np.float64(0.10847946309962808)] + [0.1107404973554093] * 2 + [np.float64(0.10847946309962808), np.float64(0.10404996159615755), np.float64(0.09763333715699414), np.float64(0.08949228714059852), np.float64(0.07996010709117474), np.float64(0.06942704557992982), np.float64(0.05832432738271626), np.float64(0.04710649908704191), np.float64(0.036232819901370686), np.float64(0.026148459529166952), np.float64(0.017266272868930763), np.float64(0.009949897686144812), np.float64(0.00449886724042994), np.float64(0.001136347358702605), np.float64(1.6467993352185907e-34)],
        },
        "qubitC2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0011001979472539116), np.float64(0.002155353661279478), np.float64(0.0031222689431689614), np.float64(0.003961358167674083), np.float64(0.004638268923354806), np.float64(0.005125288404339926), np.float64(0.005402477975855551), np.float64(0.0054584894642754405), np.float64(0.005291029752682614), np.float64(0.004906954661345363), np.float64(0.004321988269629941), np.float64(0.0035600791703438092), np.float64(0.0026524200115322603), np.float64(0.0016361704658024634), np.float64(0.0005529359089608488), np.float64(-0.0005529359089608383), np.float64(-0.0016361704658024577), np.float64(-0.002652420011532253), np.float64(-0.003560079170343802), np.float64(-0.004321988269629935), np.float64(-0.0049069546613453585), np.float64(-0.00529102975268261), np.float64(-0.005458489464275437), np.float64(-0.0054024779758555476), np.float64(-0.005125288404339924), np.float64(-0.004638268923354805), np.float64(-0.003961358167674082), np.float64(-0.003122268943168961), np.float64(-0.0021553536612794787), np.float64(-0.0011001979472539175), np.float64(-1.3386624938321075e-18)],
        },
        "qubitC2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0005656168977942142), np.float64(0.0022393111689239984), np.float64(0.004952561573278572), np.float64(0.008594287320510268), np.float64(0.013015395730642827), np.float64(0.018034886105907224), np.float64(0.02344725992057507), np.float64(0.02903093395474694), np.float64(0.03455731193740998), np.float64(0.03980014330463216), np.float64(0.044544785924232826), np.float64(0.04859699356989374), np.float64(0.05179086838448733), np.float64(0.05399565275783978)] + [0.05512108255865488] * 2 + [np.float64(0.05399565275783978), np.float64(0.05179086838448733), np.float64(0.04859699356989375), np.float64(0.04454478592423283), np.float64(0.039800143304632156), np.float64(0.03455731193741001), np.float64(0.029030933954746964), np.float64(0.02344725992057507), np.float64(0.018034886105907224), np.float64(0.013015395730642827), np.float64(0.008594287320510272), np.float64(0.004952561573278572), np.float64(0.0022393111689239984), np.float64(0.0005656168977942206), np.float64(8.19694369105052e-35)],
        },
        "qubitC2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0011001979472539116), np.float64(-0.002155353661279478), np.float64(-0.0031222689431689606), np.float64(-0.003961358167674081), np.float64(-0.004638268923354804), np.float64(-0.005125288404339924), np.float64(-0.0054024779758555476), np.float64(-0.005458489464275437), np.float64(-0.005291029752682611), np.float64(-0.004906954661345358), np.float64(-0.004321988269629936), np.float64(-0.003560079170343803), np.float64(-0.0026524200115322543), np.float64(-0.0016361704658024569), np.float64(-0.0005529359089608421), np.float64(0.000552935908960845), np.float64(0.0016361704658024643), np.float64(0.002652420011532259), np.float64(0.003560079170343808), np.float64(0.0043219882696299405), np.float64(0.004906954661345364), np.float64(0.005291029752682613), np.float64(0.0054584894642754405), np.float64(0.005402477975855551), np.float64(0.005125288404339926), np.float64(0.004638268923354807), np.float64(0.003961358167674084), np.float64(0.003122268943168962), np.float64(0.0021553536612794787), np.float64(0.0011001979472539175), np.float64(1.3386624938321075e-18)],
        },
        "qubitC2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0005656168977942144), np.float64(-0.0022393111689239984), np.float64(-0.004952561573278572), np.float64(-0.008594287320510268), np.float64(-0.013015395730642827), np.float64(-0.018034886105907224), np.float64(-0.02344725992057507), np.float64(-0.02903093395474694), np.float64(-0.03455731193740998), np.float64(-0.03980014330463216), np.float64(-0.044544785924232826), np.float64(-0.04859699356989374), np.float64(-0.05179086838448733), np.float64(-0.05399565275783978)] + [-0.05512108255865488] * 2 + [np.float64(-0.05399565275783978), np.float64(-0.05179086838448733), np.float64(-0.04859699356989375), np.float64(-0.04454478592423283), np.float64(-0.039800143304632156), np.float64(-0.03455731193741001), np.float64(-0.029030933954746964), np.float64(-0.02344725992057507), np.float64(-0.018034886105907224), np.float64(-0.013015395730642827), np.float64(-0.008594287320510272), np.float64(-0.004952561573278572), np.float64(-0.0022393111689239984), np.float64(-0.0005656168977942204), np.float64(8.19694369105052e-35)],
        },
        "qubitC2.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC2.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC2.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC2.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.02976981820061825,
        },
        "qubitC2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0010294361571335725), np.float64(0.004075599391307629), np.float64(0.0090137794218061), np.float64(0.015641806577968854), np.float64(0.023688328649262486), np.float64(0.03282392007666201), np.float64(0.04267456867374328), np.float64(0.052836987729539736), np.float64(0.06289512661387352), np.float64(0.07243720393903888), np.float64(0.08107256593819703), np.float64(0.08844767987649055), np.float64(0.09426060772274655), np.float64(0.09827336752794931)] + [0.10032167643419457] * 2 + [np.float64(0.09827336752794931), np.float64(0.09426060772274655), np.float64(0.08844767987649056), np.float64(0.08107256593819705), np.float64(0.07243720393903888), np.float64(0.06289512661387356), np.float64(0.052836987729539785), np.float64(0.04267456867374328), np.float64(0.03282392007666201), np.float64(0.023688328649262486), np.float64(0.015641806577968858), np.float64(0.0090137794218061), np.float64(0.004075599391307629), np.float64(0.0010294361571335838), np.float64(0.0)],
        },
        "qubitC2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0014928327546150184), np.float64(-0.002924548760855697), np.float64(-0.004236533397207055), np.float64(-0.005375073858505217), np.float64(-0.006293558164491814), np.float64(-0.00695438345976243), np.float64(-0.0073304954790850115), np.float64(-0.007406496152197205), np.float64(-0.007179274002612097), np.float64(-0.006658131531831172), np.float64(-0.0058644043738398455), np.float64(-0.004830587811745254), np.float64(-0.003599006417067845), np.float64(-0.002220081276810381), np.float64(-0.0007502657482318286), np.float64(0.0007502657482318235), np.float64(0.0022200812768103824), np.float64(0.003599006417067844), np.float64(0.0048305878117452525), np.float64(0.005864404373839844), np.float64(0.006658131531831174), np.float64(0.0071792740026120965), np.float64(0.007406496152197205), np.float64(0.0073304954790850115), np.float64(0.00695438345976243), np.float64(0.006293558164491814), np.float64(0.005375073858505217), np.float64(0.004236533397207055), np.float64(0.0029245487608556982), np.float64(0.0014928327546150262), np.float64(1.816399697122858e-18)],
        },
        "qubitC2.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC2.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC2.z.z90.wf": {
            "type": "constant",
            "sample": 0.01729078010516483,
        },
        "qubitC2.z.z180.wf": {
            "type": "constant",
            "sample": 0.02457771966234662,
        },
        "qubitC2.z.-z90.wf": {
            "type": "constant",
            "sample": 0.03028389262492936,
        },
        "qubitC2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.018244444444444445,
        },
        "qubitC2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.002128256352676728), np.float64(0.008425894345568822), np.float64(0.01863508793930623), np.float64(0.032337871548632026), np.float64(0.04897325160257173), np.float64(0.06786017368714284), np.float64(0.08822540499308393), np.float64(0.1092351905579167), np.float64(0.13002938729207184), np.float64(0.1497566783380388), np.float64(0.16760942608253726), np.float64(0.1828567369352988), np.float64(0.19487438419852832), np.float64(0.20317036398125563)] + [0.2074050378964396] * 2 + [np.float64(0.20317036398125563), np.float64(0.19487438419852832), np.float64(0.18285673693529883), np.float64(0.1676094260825373), np.float64(0.1497566783380388), np.float64(0.13002938729207195), np.float64(0.10923519055791678), np.float64(0.08822540499308393), np.float64(0.06786017368714284), np.float64(0.04897325160257173), np.float64(0.03233787154863204), np.float64(0.01863508793930623), np.float64(0.008425894345568822), np.float64(0.002128256352676751), np.float64(0.0)],
        },
        "qubitC3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0013626632003288573), np.float64(0.0026695388091301235), np.float64(0.003867123185428111), np.float64(0.00490638708414699), np.float64(0.005744782919164887), np.float64(0.006347986666488841), np.float64(0.0066913030938299785), np.float64(0.0067606767863149855), np.float64(0.006553267576822464), np.float64(0.006077566822758586), np.float64(0.005353049768887191), np.float64(0.00440937822852091), np.float64(0.0032851862253990404), np.float64(0.0020264983122163438), np.float64(0.0006848453200280706), np.float64(-0.0006848453200280659), np.float64(-0.0020264983122163446), np.float64(-0.0032851862253990387), np.float64(-0.004409378228520908), np.float64(-0.0053530497688871895), np.float64(-0.006077566822758586), np.float64(-0.006553267576822463), np.float64(-0.0067606767863149855), np.float64(-0.0066913030938299785), np.float64(-0.006347986666488841), np.float64(-0.005744782919164888), np.float64(-0.0049063870841469905), np.float64(-0.003867123185428112), np.float64(-0.0026695388091301244), np.float64(-0.0013626632003288645), np.float64(-1.6580162893036912e-18)],
        },
        "qubitC3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.001052422766398641), np.float64(0.004166604753883778), np.float64(0.009215050985986923), np.float64(0.015991077480798522), np.float64(0.024217272917471697), np.float64(0.0335568558882921), np.float64(0.043627462769079965), np.float64(0.054016801730889755), np.float64(0.06429953201592947), np.float64(0.07405467743816012), np.float64(0.0828828611978146), np.float64(0.09042265641450517), np.float64(0.09636538298617217), np.float64(0.10046774498873082)] + [0.1025617912397893] * 2 + [np.float64(0.10046774498873082), np.float64(0.09636538298617217), np.float64(0.09042265641450518), np.float64(0.08288286119781461), np.float64(0.0740546774381601), np.float64(0.06429953201592951), np.float64(0.0540168017308898), np.float64(0.043627462769079965), np.float64(0.0335568558882921), np.float64(0.024217272917471697), np.float64(0.01599107748079853), np.float64(0.009215050985986923), np.float64(0.004166604753883778), np.float64(0.0010524227663986523), np.float64(0.0)],
        },
        "qubitC3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0005379556773534572), np.float64(0.0010538873860689555), np.float64(0.0015266728213722807), np.float64(0.0019369560919921848), np.float64(0.0022679401526232517), np.float64(0.002506074476934269), np.float64(0.0026416098177083494), np.float64(0.0026689973421694757), np.float64(0.0025871158018483113), np.float64(0.002399317436626277), np.float64(0.0021132907336409356), np.float64(0.0017407456597192143), np.float64(0.001296934253959659), np.float64(0.0008000262074596908), np.float64(0.000270364994027236), np.float64(-0.00027036499402723416), np.float64(-0.0008000262074596914), np.float64(-0.0012969342539596582), np.float64(-0.0017407456597192139), np.float64(-0.0021132907336409348), np.float64(-0.0023993174366262774), np.float64(-0.0025871158018483105), np.float64(-0.0026689973421694757), np.float64(-0.0026416098177083494), np.float64(-0.002506074476934269), np.float64(-0.0022679401526232517), np.float64(-0.001936956091992185), np.float64(-0.0015266728213722814), np.float64(-0.0010538873860689558), np.float64(-0.00053795567735346), np.float64(-6.545559282441745e-19)],
        },
        "qubitC3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.001052422766398641), np.float64(-0.004166604753883778), np.float64(-0.009215050985986923), np.float64(-0.015991077480798522), np.float64(-0.024217272917471697), np.float64(-0.0335568558882921), np.float64(-0.043627462769079965), np.float64(-0.054016801730889755), np.float64(-0.06429953201592947), np.float64(-0.07405467743816012), np.float64(-0.0828828611978146), np.float64(-0.09042265641450517), np.float64(-0.09636538298617217), np.float64(-0.10046774498873082)] + [-0.1025617912397893] * 2 + [np.float64(-0.10046774498873082), np.float64(-0.09636538298617217), np.float64(-0.09042265641450518), np.float64(-0.08288286119781461), np.float64(-0.0740546774381601), np.float64(-0.06429953201592951), np.float64(-0.0540168017308898), np.float64(-0.043627462769079965), np.float64(-0.0335568558882921), np.float64(-0.024217272917471697), np.float64(-0.01599107748079853), np.float64(-0.009215050985986923), np.float64(-0.004166604753883778), np.float64(-0.0010524227663986523), np.float64(8.01599822387153e-35)],
        },
        "qubitC3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0005379556773534571), np.float64(-0.0010538873860689551), np.float64(-0.0015266728213722796), np.float64(-0.0019369560919921828), np.float64(-0.0022679401526232486), np.float64(-0.002506074476934265), np.float64(-0.0026416098177083442), np.float64(-0.002668997342169469), np.float64(-0.0025871158018483035), np.float64(-0.002399317436626268), np.float64(-0.0021132907336409257), np.float64(-0.0017407456597192032), np.float64(-0.0012969342539596474), np.float64(-0.0008000262074596786), np.float64(-0.0002703649940272234), np.float64(0.00027036499402724673), np.float64(0.0008000262074597036), np.float64(0.00129693425395967), np.float64(0.001740745659719225), np.float64(0.0021132907336409447), np.float64(0.0023993174366262865), np.float64(0.0025871158018483183), np.float64(0.002668997342169482), np.float64(0.0026416098177083546), np.float64(0.002506074476934273), np.float64(0.0022679401526232547), np.float64(0.001936956091992187), np.float64(0.0015266728213722824), np.float64(0.0010538873860689562), np.float64(0.0005379556773534601), np.float64(6.545559282441745e-19)],
        },
        "qubitC3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.001362663200328857), np.float64(-0.002669538809130123), np.float64(-0.0038671231854281096), np.float64(-0.004906387084146988), np.float64(-0.005744782919164884), np.float64(-0.006347986666488836), np.float64(-0.006691303093829973), np.float64(-0.006760676786314979), np.float64(-0.006553267576822456), np.float64(-0.006077566822758576), np.float64(-0.005353049768887181), np.float64(-0.004409378228520899), np.float64(-0.0032851862253990283), np.float64(-0.002026498312216331), np.float64(-0.0006848453200280579), np.float64(0.0006848453200280786), np.float64(0.002026498312216357), np.float64(0.003285186225399051), np.float64(0.0044093782285209195), np.float64(0.0053530497688872), np.float64(0.006077566822758595), np.float64(0.006553267576822471), np.float64(0.0067606767863149925), np.float64(0.006691303093829984), np.float64(0.006347986666488845), np.float64(0.00574478291916489), np.float64(0.004906387084146992), np.float64(0.0038671231854281135), np.float64(0.002669538809130125), np.float64(0.0013626632003288647), np.float64(1.6580162893036912e-18)],
        },
        "qubitC3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.002128256352676728), np.float64(0.008425894345568822), np.float64(0.01863508793930623), np.float64(0.032337871548632026), np.float64(0.04897325160257173), np.float64(0.06786017368714284), np.float64(0.08822540499308393), np.float64(0.1092351905579167), np.float64(0.13002938729207184), np.float64(0.1497566783380388), np.float64(0.16760942608253726), np.float64(0.1828567369352988), np.float64(0.19487438419852832), np.float64(0.20317036398125563)] + [0.2074050378964396] * 2 + [np.float64(0.20317036398125563), np.float64(0.19487438419852832), np.float64(0.18285673693529883), np.float64(0.1676094260825373), np.float64(0.1497566783380388), np.float64(0.13002938729207195), np.float64(0.10923519055791678), np.float64(0.08822540499308393), np.float64(0.06786017368714284), np.float64(0.04897325160257173), np.float64(0.03233787154863204), np.float64(0.01863508793930623), np.float64(0.008425894345568822), np.float64(0.002128256352676751), np.float64(-1.0152421708149686e-34)],
        },
        "qubitC3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0005379556773534571), np.float64(-0.0010538873860689553), np.float64(-0.00152667282137228), np.float64(-0.0019369560919921837), np.float64(-0.0022679401526232504), np.float64(-0.002506074476934267), np.float64(-0.002641609817708347), np.float64(-0.0026689973421694722), np.float64(-0.0025871158018483074), np.float64(-0.0023993174366262727), np.float64(-0.0021132907336409304), np.float64(-0.0017407456597192087), np.float64(-0.0012969342539596532), np.float64(-0.0008000262074596846), np.float64(-0.0002703649940272297), np.float64(0.00027036499402724045), np.float64(0.0008000262074596975), np.float64(0.001296934253959664), np.float64(0.0017407456597192195), np.float64(0.00211329073364094), np.float64(0.0023993174366262818), np.float64(0.0025871158018483144), np.float64(0.002668997342169479), np.float64(0.002641609817708352), np.float64(0.0025060744769342712), np.float64(0.002267940152623253), np.float64(0.001936956091992186), np.float64(0.001526672821372282), np.float64(0.001053887386068956), np.float64(0.0005379556773534601), np.float64(6.545559282441745e-19)],
        },
        "qubitC3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.001052422766398641), np.float64(0.004166604753883778), np.float64(0.009215050985986923), np.float64(0.015991077480798522), np.float64(0.024217272917471697), np.float64(0.0335568558882921), np.float64(0.043627462769079965), np.float64(0.054016801730889755), np.float64(0.06429953201592947), np.float64(0.07405467743816012), np.float64(0.0828828611978146), np.float64(0.09042265641450517), np.float64(0.09636538298617217), np.float64(0.10046774498873082)] + [0.1025617912397893] * 2 + [np.float64(0.10046774498873082), np.float64(0.09636538298617217), np.float64(0.09042265641450518), np.float64(0.08288286119781461), np.float64(0.0740546774381601), np.float64(0.06429953201592951), np.float64(0.0540168017308898), np.float64(0.043627462769079965), np.float64(0.0335568558882921), np.float64(0.024217272917471697), np.float64(0.01599107748079853), np.float64(0.009215050985986923), np.float64(0.004166604753883778), np.float64(0.0010524227663986523), np.float64(-4.007999111935765e-35)],
        },
        "qubitC3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0005379556773534573), np.float64(0.0010538873860689558), np.float64(0.0015266728213722814), np.float64(0.0019369560919921859), np.float64(0.002267940152623253), np.float64(0.0025060744769342712), np.float64(0.002641609817708352), np.float64(0.002668997342169479), np.float64(0.0025871158018483153), np.float64(0.0023993174366262813), np.float64(0.002113290733640941), np.float64(0.00174074565971922), np.float64(0.001296934253959665), np.float64(0.000800026207459697), np.float64(0.0002703649940272423), np.float64(-0.00027036499402722787), np.float64(-0.0008000262074596852), np.float64(-0.0012969342539596524), np.float64(-0.0017407456597192082), np.float64(-0.0021132907336409296), np.float64(-0.002399317436626273), np.float64(-0.0025871158018483066), np.float64(-0.0026689973421694722), np.float64(-0.002641609817708347), np.float64(-0.002506074476934267), np.float64(-0.0022679401526232504), np.float64(-0.001936956091992184), np.float64(-0.0015266728213722807), np.float64(-0.0010538873860689555), np.float64(-0.0005379556773534599), np.float64(-6.545559282441745e-19)],
        },
        "qubitC3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.001052422766398641), np.float64(-0.004166604753883778), np.float64(-0.009215050985986923), np.float64(-0.015991077480798522), np.float64(-0.024217272917471697), np.float64(-0.0335568558882921), np.float64(-0.043627462769079965), np.float64(-0.054016801730889755), np.float64(-0.06429953201592947), np.float64(-0.07405467743816012), np.float64(-0.0828828611978146), np.float64(-0.09042265641450517), np.float64(-0.09636538298617217), np.float64(-0.10046774498873082)] + [-0.1025617912397893] * 2 + [np.float64(-0.10046774498873082), np.float64(-0.09636538298617217), np.float64(-0.09042265641450518), np.float64(-0.08288286119781461), np.float64(-0.0740546774381601), np.float64(-0.06429953201592951), np.float64(-0.0540168017308898), np.float64(-0.043627462769079965), np.float64(-0.0335568558882921), np.float64(-0.024217272917471697), np.float64(-0.01599107748079853), np.float64(-0.009215050985986923), np.float64(-0.004166604753883778), np.float64(-0.0010524227663986523), np.float64(-4.007999111935765e-35)],
        },
        "qubitC3.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC3.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC3.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC3.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.047085973003043784,
        },
        "qubitC3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.001612649397959325), np.float64(0.0063845755360065854), np.float64(0.014120415197372612), np.float64(0.024503462197398504), np.float64(0.03710863337194701), np.float64(0.051419871532087345), np.float64(0.06685127290604431), np.float64(0.08277107410845543), np.float64(0.09852751660759673), np.float64(0.11347552979625009), np.float64(0.12700314025815493), np.float64(0.13855652603157423), np.float64(0.14766269014547898), np.float64(0.15394882517111705)] + [0.15715757600194932] * 2 + [np.float64(0.15394882517111705), np.float64(0.14766269014547898), np.float64(0.13855652603157426), np.float64(0.12700314025815493), np.float64(0.11347552979625007), np.float64(0.09852751660759682), np.float64(0.0827710741084555), np.float64(0.06685127290604431), np.float64(0.051419871532087345), np.float64(0.03710863337194701), np.float64(0.02450346219739851), np.float64(0.014120415197372612), np.float64(0.0063845755360065854), np.float64(0.0016126493979593424), np.float64(0.0)],
        },
        "qubitC3.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.000678852362866962), np.float64(0.0013299124302363849), np.float64(0.0019265257264538544), np.float64(0.0024442668330731418), np.float64(0.002861939368357256), np.float64(0.003162443769637227), np.float64(0.0033334773514168844), np.float64(0.0033680379787625757), np.float64(0.0032647107355300684), np.float64(0.003027725851197736), np.float64(0.0026667855147746874), np.float64(0.0021966666660427677), np.float64(0.0016366160259056837), np.float64(0.0010095621333738032), np.float64(0.0003411766484830867), np.float64(-0.0003411766484830844), np.float64(-0.0010095621333738036), np.float64(-0.0016366160259056828), np.float64(-0.002196666666042767), np.float64(-0.0026667855147746865), np.float64(-0.0030277258511977366), np.float64(-0.003264710735530068), np.float64(-0.0033680379787625757), np.float64(-0.0033334773514168844), np.float64(-0.003162443769637227), np.float64(-0.0028619393683572566), np.float64(-0.0024442668330731418), np.float64(-0.0019265257264538549), np.float64(-0.0013299124302363855), np.float64(-0.0006788523628669656), np.float64(-8.25991540238329e-19)],
        },
        "qubitC3.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC3.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC3.z.z90.wf": {
            "type": "constant",
            "sample": 0.018092149544688144,
        },
        "qubitC3.z.z180.wf": {
            "type": "constant",
            "sample": 0.025534508275873505,
        },
        "qubitC3.z.-z90.wf": {
            "type": "constant",
            "sample": 0.03139992138505721,
        },
        "qubitC3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01658888888888889,
        },
        "qubitC3.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC3.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0013318728729940627), np.float64(0.005272964460067455), np.float64(0.011661926008586794), np.float64(0.02023719268208458), np.float64(0.03064769205539679), np.float64(0.04246721706099824), np.float64(0.05521187495642443), np.float64(0.06835989795000286), np.float64(0.08137300443554933), np.float64(0.09371843630457051), np.float64(0.10489077012582194), np.float64(0.1144326092399682), np.float64(0.1219533096325576), np.float64(0.12714497294587274)] + [0.129795051873872] * 2 + [np.float64(0.12714497294587274), np.float64(0.1219533096325576), np.float64(0.11443260923996823), np.float64(0.10489077012582196), np.float64(0.0937184363045705), np.float64(0.08137300443554939), np.float64(0.06835989795000293), np.float64(0.05521187495642443), np.float64(0.04246721706099824), np.float64(0.03064769205539679), np.float64(0.020237192682084588), np.float64(0.011661926008586794), np.float64(0.005272964460067455), np.float64(0.0013318728729940772), np.float64(0.0)],
        },
        "qubitC4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0017975143813591585), np.float64(0.0035214383127465014), np.float64(0.005101194145858569), np.float64(0.006472106491274323), np.float64(0.007578050036497085), np.float64(0.008373747322842583), np.float64(0.008826622409917372), np.float64(0.008918134538446046), np.float64(0.008644537191134423), np.float64(0.008017031475527766), np.float64(0.007061307549351801), np.float64(0.005816492862436962), np.float64(0.004333550274325061), np.float64(0.002673191628811757), np.float64(0.000903392203928223), np.float64(-0.000903392203928217), np.float64(-0.0026731916288117584), np.float64(-0.0043335502743250594), np.float64(-0.005816492862436959), np.float64(-0.0070613075493517995), np.float64(-0.008017031475527766), np.float64(-0.008644537191134422), np.float64(-0.008918134538446046), np.float64(-0.008826622409917372), np.float64(-0.008373747322842583), np.float64(-0.007578050036497087), np.float64(-0.006472106491274324), np.float64(-0.005101194145858571), np.float64(-0.0035214383127465027), np.float64(-0.0017975143813591683), np.float64(-2.187120136385779e-18)],
        },
        "qubitC4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006609544945816668), np.float64(0.0026167584236596376), np.float64(0.005787340944580256), np.float64(0.0100428980364094), np.float64(0.015209206691800868), np.float64(0.02107475762738743), np.float64(0.0273994144986932), np.float64(0.0339242451102913), np.float64(0.04038212213030785), np.float64(0.046508659314775054), np.float64(0.052053035511523245), np.float64(0.056788263307621045), np.float64(0.06052048192073054), np.float64(0.06309689388231397)] + [0.06441202058394856] * 2 + [np.float64(0.06309689388231397), np.float64(0.06052048192073054), np.float64(0.05678826330762105), np.float64(0.05205303551152325), np.float64(0.04650865931477505), np.float64(0.040382122130307875), np.float64(0.03392424511029133), np.float64(0.0273994144986932), np.float64(0.02107475762738743), np.float64(0.015209206691800868), np.float64(0.010042898036409403), np.float64(0.005787340944580256), np.float64(0.0026167584236596376), np.float64(0.000660954494581674), np.float64(0.0)],
        },
        "qubitC4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0008920334917278672), np.float64(0.0017475470274949108), np.float64(0.0025315157826282554), np.float64(0.0032118439841803243), np.float64(0.0037606789156444066), np.float64(0.004155551210440996), np.float64(0.0043802947504255965), np.float64(0.0044257085086649925), np.float64(0.004289933240560974), np.float64(0.003978527601542636), np.float64(0.003504240575059999), np.float64(0.002886489527703618), np.float64(0.002150565259934934), np.float64(0.0013265965977438251), np.float64(0.00044831691497260374), np.float64(-0.00044831691497260065), np.float64(-0.0013265965977438258), np.float64(-0.002150565259934933), np.float64(-0.0028864895277036177), np.float64(-0.0035042405750599984), np.float64(-0.003978527601542636), np.float64(-0.004289933240560972), np.float64(-0.0044257085086649925), np.float64(-0.0043802947504255965), np.float64(-0.004155551210440996), np.float64(-0.0037606789156444066), np.float64(-0.0032118439841803243), np.float64(-0.002531515782628256), np.float64(-0.0017475470274949112), np.float64(-0.0008920334917278719), np.float64(-1.0853790280183091e-18)],
        },
        "qubitC4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0006609544945816669), np.float64(-0.0026167584236596376), np.float64(-0.005787340944580256), np.float64(-0.0100428980364094), np.float64(-0.015209206691800868), np.float64(-0.02107475762738743), np.float64(-0.0273994144986932), np.float64(-0.0339242451102913), np.float64(-0.04038212213030785), np.float64(-0.046508659314775054), np.float64(-0.052053035511523245), np.float64(-0.056788263307621045), np.float64(-0.06052048192073054), np.float64(-0.06309689388231397)] + [-0.06441202058394856] * 2 + [np.float64(-0.06309689388231397), np.float64(-0.06052048192073054), np.float64(-0.05678826330762105), np.float64(-0.05205303551152325), np.float64(-0.04650865931477505), np.float64(-0.040382122130307875), np.float64(-0.03392424511029133), np.float64(-0.0273994144986932), np.float64(-0.02107475762738743), np.float64(-0.015209206691800868), np.float64(-0.010042898036409403), np.float64(-0.005787340944580256), np.float64(-0.0026167584236596376), np.float64(-0.0006609544945816738), np.float64(1.3292059525242876e-34)],
        },
        "qubitC4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0008920334917278671), np.float64(-0.0017475470274949105), np.float64(-0.0025315157826282545), np.float64(-0.003211843984180323), np.float64(-0.003760678915644405), np.float64(-0.004155551210440993), np.float64(-0.004380294750425593), np.float64(-0.004425708508664988), np.float64(-0.004289933240560969), np.float64(-0.00397852760154263), np.float64(-0.0035042405750599923), np.float64(-0.002886489527703611), np.float64(-0.0021505652599349266), np.float64(-0.0013265965977438173), np.float64(-0.0004483169149725958), np.float64(0.00044831691497260857), np.float64(0.0013265965977438336), np.float64(0.0021505652599349404), np.float64(0.0028864895277036246), np.float64(0.003504240575060005), np.float64(0.003978527601542642), np.float64(0.0042899332405609775), np.float64(0.004425708508664997), np.float64(0.0043802947504256), np.float64(0.004155551210440998), np.float64(0.0037606789156444083), np.float64(0.0032118439841803256), np.float64(0.0025315157826282567), np.float64(0.0017475470274949114), np.float64(0.000892033491727872), np.float64(1.0853790280183091e-18)],
        },
        "qubitC4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0017975143813591585), np.float64(-0.003521438312746501), np.float64(-0.005101194145858568), np.float64(-0.0064721064912743225), np.float64(-0.007578050036497083), np.float64(-0.008373747322842582), np.float64(-0.008826622409917368), np.float64(-0.008918134538446042), np.float64(-0.008644537191134418), np.float64(-0.00801703147552776), np.float64(-0.007061307549351795), np.float64(-0.005816492862436955), np.float64(-0.004333550274325053), np.float64(-0.0026731916288117493), np.float64(-0.0009033922039282151), np.float64(0.0009033922039282249), np.float64(0.0026731916288117662), np.float64(0.004333550274325067), np.float64(0.005816492862436966), np.float64(0.0070613075493518055), np.float64(0.00801703147552777), np.float64(0.008644537191134427), np.float64(0.00891813453844605), np.float64(0.008826622409917375), np.float64(0.008373747322842585), np.float64(0.007578050036497089), np.float64(0.006472106491274325), np.float64(0.005101194145858572), np.float64(0.003521438312746503), np.float64(0.0017975143813591683), np.float64(2.187120136385779e-18)],
        },
        "qubitC4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.001331872872994063), np.float64(0.005272964460067455), np.float64(0.011661926008586794), np.float64(0.02023719268208458), np.float64(0.03064769205539679), np.float64(0.04246721706099824), np.float64(0.05521187495642443), np.float64(0.06835989795000286), np.float64(0.08137300443554933), np.float64(0.09371843630457051), np.float64(0.10489077012582194), np.float64(0.1144326092399682), np.float64(0.1219533096325576), np.float64(0.12714497294587274)] + [0.129795051873872] * 2 + [np.float64(0.12714497294587274), np.float64(0.1219533096325576), np.float64(0.11443260923996823), np.float64(0.10489077012582196), np.float64(0.0937184363045705), np.float64(0.08137300443554939), np.float64(0.06835989795000293), np.float64(0.05521187495642443), np.float64(0.04246721706099824), np.float64(0.03064769205539679), np.float64(0.020237192682084588), np.float64(0.011661926008586794), np.float64(0.005272964460067455), np.float64(0.001331872872994077), np.float64(-1.3392248371877835e-34)],
        },
        "qubitC4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0008920334917278672), np.float64(-0.0017475470274949105), np.float64(-0.002531515782628255), np.float64(-0.003211843984180324), np.float64(-0.0037606789156444057), np.float64(-0.004155551210440995), np.float64(-0.004380294750425595), np.float64(-0.004425708508664991), np.float64(-0.004289933240560971), np.float64(-0.003978527601542633), np.float64(-0.0035042405750599958), np.float64(-0.0028864895277036146), np.float64(-0.00215056525993493), np.float64(-0.0013265965977438212), np.float64(-0.0004483169149725998), np.float64(0.0004483169149726046), np.float64(0.0013265965977438297), np.float64(0.002150565259934937), np.float64(0.002886489527703621), np.float64(0.0035042405750600014), np.float64(0.0039785276015426385), np.float64(0.004289933240560975), np.float64(0.004425708508664994), np.float64(0.004380294750425598), np.float64(0.004155551210440997), np.float64(0.0037606789156444075), np.float64(0.0032118439841803247), np.float64(0.0025315157826282563), np.float64(0.0017475470274949114), np.float64(0.0008920334917278719), np.float64(1.0853790280183091e-18)],
        },
        "qubitC4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0006609544945816669), np.float64(0.0026167584236596376), np.float64(0.005787340944580256), np.float64(0.0100428980364094), np.float64(0.015209206691800868), np.float64(0.02107475762738743), np.float64(0.0273994144986932), np.float64(0.0339242451102913), np.float64(0.04038212213030785), np.float64(0.046508659314775054), np.float64(0.052053035511523245), np.float64(0.056788263307621045), np.float64(0.06052048192073054), np.float64(0.06309689388231397)] + [0.06441202058394856] * 2 + [np.float64(0.06309689388231397), np.float64(0.06052048192073054), np.float64(0.05678826330762105), np.float64(0.05205303551152325), np.float64(0.04650865931477505), np.float64(0.040382122130307875), np.float64(0.03392424511029133), np.float64(0.0273994144986932), np.float64(0.02107475762738743), np.float64(0.015209206691800868), np.float64(0.010042898036409403), np.float64(0.005787340944580256), np.float64(0.0026167584236596376), np.float64(0.0006609544945816738), np.float64(-6.646029762621438e-35)],
        },
        "qubitC4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0008920334917278672), np.float64(0.001747547027494911), np.float64(0.002531515782628256), np.float64(0.0032118439841803247), np.float64(0.0037606789156444075), np.float64(0.004155551210440997), np.float64(0.004380294750425598), np.float64(0.004425708508664994), np.float64(0.004289933240560977), np.float64(0.0039785276015426385), np.float64(0.003504240575060002), np.float64(0.0028864895277036216), np.float64(0.002150565259934938), np.float64(0.001326596597743829), np.float64(0.0004483169149726077), np.float64(-0.0004483169149725967), np.float64(-0.0013265965977438219), np.float64(-0.002150565259934929), np.float64(-0.002886489527703614), np.float64(-0.0035042405750599953), np.float64(-0.003978527601542633), np.float64(-0.00428993324056097), np.float64(-0.004425708508664991), np.float64(-0.004380294750425595), np.float64(-0.004155551210440995), np.float64(-0.0037606789156444057), np.float64(-0.003211843984180324), np.float64(-0.0025315157826282554), np.float64(-0.001747547027494911), np.float64(-0.0008920334917278719), np.float64(-1.0853790280183091e-18)],
        },
        "qubitC4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(-0.0006609544945816667), np.float64(-0.0026167584236596376), np.float64(-0.005787340944580256), np.float64(-0.0100428980364094), np.float64(-0.015209206691800868), np.float64(-0.02107475762738743), np.float64(-0.0273994144986932), np.float64(-0.0339242451102913), np.float64(-0.04038212213030785), np.float64(-0.046508659314775054), np.float64(-0.052053035511523245), np.float64(-0.056788263307621045), np.float64(-0.06052048192073054), np.float64(-0.06309689388231397)] + [-0.06441202058394856] * 2 + [np.float64(-0.06309689388231397), np.float64(-0.06052048192073054), np.float64(-0.05678826330762105), np.float64(-0.05205303551152325), np.float64(-0.04650865931477505), np.float64(-0.040382122130307875), np.float64(-0.03392424511029133), np.float64(-0.0273994144986932), np.float64(-0.02107475762738743), np.float64(-0.015209206691800868), np.float64(-0.010042898036409403), np.float64(-0.005787340944580256), np.float64(-0.0026167584236596376), np.float64(-0.0006609544945816741), np.float64(-6.646029762621438e-35)],
        },
        "qubitC4.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC4.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC4.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC4.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.02836710646226952,
        },
        "qubitC4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0010091967820947487), np.float64(0.0039954704935444485), np.float64(0.008836562737730086), np.float64(0.015334278629370507), np.float64(0.023222600916412775), np.float64(0.03217858075758684), np.float64(0.041835559285912094), np.float64(0.05179817866579211), np.float64(0.0616585680892565), np.float64(0.07101304205476212), np.float64(0.0794786272990571), np.float64(0.086708741767576), np.float64(0.09240738372447525), np.float64(0.09634125009847742)] + [0.09834928793802118] * 2 + [np.float64(0.09634125009847742), np.float64(0.09240738372447525), np.float64(0.08670874176757601), np.float64(0.0794786272990571), np.float64(0.07101304205476212), np.float64(0.06165856808925655), np.float64(0.05179817866579215), np.float64(0.041835559285912094), np.float64(0.03217858075758684), np.float64(0.023222600916412775), np.float64(0.015334278629370514), np.float64(0.008836562737730086), np.float64(0.0039954704935444485), np.float64(0.0010091967820947598), np.float64(0.0)],
        },
        "qubitC4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.000900158013173167), np.float64(0.0017634634515229489), np.float64(0.002554572488969224), np.float64(0.0032410970285675755), np.float64(0.00379493067500364), np.float64(0.004193399413719634), np.float64(0.004420189887734435), np.float64(0.00446601726839513), np.float64(0.00432900537735298), np.float64(0.004014763497525464), np.float64(0.0035361567283945755), np.float64(0.0029127792873224374), np.float64(0.002170152319990285), np.float64(0.0013386790616956603), np.float64(0.00045240012532710546), np.float64(-0.00045240012532710237), np.float64(-0.001338679061695661), np.float64(-0.0021701523199902837), np.float64(-0.0029127792873224366), np.float64(-0.0035361567283945746), np.float64(-0.004014763497525464), np.float64(-0.004329005377352979), np.float64(-0.00446601726839513), np.float64(-0.004420189887734435), np.float64(-0.004193399413719634), np.float64(-0.00379493067500364), np.float64(-0.003241097028567576), np.float64(-0.0025545724889692242), np.float64(-0.0017634634515229497), np.float64(-0.0009001580131731717), np.float64(-1.0952645146857798e-18)],
        },
        "qubitC4.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC4.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC4.z.z90.wf": {
            "type": "constant",
            "sample": 0.018890467518712446,
        },
        "qubitC4.z.z180.wf": {
            "type": "constant",
            "sample": 0.02634963499864954,
        },
        "qubitC4.z.-z90.wf": {
            "type": "constant",
            "sample": 0.03253175743797823,
        },
        "qubitC4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.014933333333333335,
        },
        "qubitC4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC5.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitC5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB2.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitB2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitB2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitB2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB4.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitB4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitB4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitB4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB5.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitB5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitB5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitB5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.2033748528790328] * 50 + [0.0] * 2,
        },
        "qubitC2.z.Cz.CZ_snz_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.20232766962830684] * 26 + [0.08760032829676206] + [0] * 2 + [-0.08760032829676206] + [-0.20232766962830684] * 26 + [0.0] * 4,
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.210600443040884] * 51 + [np.float64(0.0)],
        },
        "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.21143597111202225] * 26 + [0.1912462623254267] + [0] * 2 + [-0.1912462623254267] + [-0.21143597111202225] * 26 + [0.0] * 4,
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.13426639835122103] * 42 + [0.0] * 2,
        },
        "qubitC3.z.Cz.CZ_snz_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.13345224540011472] * 22 + [0.03987712031031265] + [0] * 2 + [-0.03987712031031265] + [-0.13345224540011472] * 22 + [0.0] * 4,
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.1405818855635322] * 48,
        },
        "qubitC3.z.Cz.CZ_snz_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.14128141093535543] * 24 + [0.04242761306307403] + [0] * 2 + [-0.04242761306307403] + [-0.14128141093535543] * 24 + [0.0] * 4,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [[1, 0]],
        },
    },
    "integration_weights": {
        "qubitC1.resonator.readout.iw1": {
            "cosine": [(np.float64(0.9346129235434444), 1500)],
            "sine": [(np.float64(-0.35566653363168105), 1500)],
        },
        "qubitC1.resonator.readout.iw2": {
            "cosine": [(np.float64(0.35566653363168105), 1500)],
            "sine": [(np.float64(0.9346129235434444), 1500)],
        },
        "qubitC1.resonator.readout.iw3": {
            "cosine": [(np.float64(-0.35566653363168105), 1500)],
            "sine": [(np.float64(-0.9346129235434444), 1500)],
        },
        "qubitC2.resonator.readout.iw1": {
            "cosine": [(np.float64(-0.1561117302631273), 1500)],
            "sine": [(np.float64(0.9877394027142243), 1500)],
        },
        "qubitC2.resonator.readout.iw2": {
            "cosine": [(np.float64(-0.9877394027142243), 1500)],
            "sine": [(np.float64(-0.1561117302631273), 1500)],
        },
        "qubitC2.resonator.readout.iw3": {
            "cosine": [(np.float64(0.9877394027142243), 1500)],
            "sine": [(np.float64(0.1561117302631273), 1500)],
        },
        "qubitC3.resonator.readout.iw1": {
            "cosine": [(np.float64(0.3164120553070932), 1500)],
            "sine": [(np.float64(0.9486218483971055), 1500)],
        },
        "qubitC3.resonator.readout.iw2": {
            "cosine": [(np.float64(-0.9486218483971055), 1500)],
            "sine": [(np.float64(0.3164120553070932), 1500)],
        },
        "qubitC3.resonator.readout.iw3": {
            "cosine": [(np.float64(0.9486218483971055), 1500)],
            "sine": [(np.float64(-0.3164120553070932), 1500)],
        },
        "qubitC4.resonator.readout.iw1": {
            "cosine": [(np.float64(0.025991300810657807), 1500)],
            "sine": [(np.float64(0.9996621690762184), 1500)],
        },
        "qubitC4.resonator.readout.iw2": {
            "cosine": [(np.float64(-0.9996621690762184), 1500)],
            "sine": [(np.float64(0.025991300810657807), 1500)],
        },
        "qubitC4.resonator.readout.iw3": {
            "cosine": [(np.float64(0.9996621690762184), 1500)],
            "sine": [(np.float64(-0.025991300810657807), 1500)],
        },
        "qubitC5.resonator.readout.iw1": {
            "cosine": [(np.float64(1.0), 1500)],
            "sine": [(np.float64(-0.0), 1500)],
        },
        "qubitC5.resonator.readout.iw2": {
            "cosine": [(np.float64(0.0), 1500)],
            "sine": [(np.float64(1.0), 1500)],
        },
        "qubitC5.resonator.readout.iw3": {
            "cosine": [(np.float64(-0.0), 1500)],
            "sine": [(np.float64(-1.0), 1500)],
        },
        "qubitB2.resonator.readout.iw1": {
            "cosine": [(np.float64(1.0), 1500)],
            "sine": [(np.float64(-0.0), 1500)],
        },
        "qubitB2.resonator.readout.iw2": {
            "cosine": [(np.float64(0.0), 1500)],
            "sine": [(np.float64(1.0), 1500)],
        },
        "qubitB2.resonator.readout.iw3": {
            "cosine": [(np.float64(-0.0), 1500)],
            "sine": [(np.float64(-1.0), 1500)],
        },
        "qubitB4.resonator.readout.iw1": {
            "cosine": [(np.float64(1.0), 1500)],
            "sine": [(np.float64(-0.0), 1500)],
        },
        "qubitB4.resonator.readout.iw2": {
            "cosine": [(np.float64(0.0), 1500)],
            "sine": [(np.float64(1.0), 1500)],
        },
        "qubitB4.resonator.readout.iw3": {
            "cosine": [(np.float64(-0.0), 1500)],
            "sine": [(np.float64(-1.0), 1500)],
        },
        "qubitB5.resonator.readout.iw1": {
            "cosine": [(np.float64(1.0), 1500)],
            "sine": [(np.float64(-0.0), 1500)],
        },
        "qubitB5.resonator.readout.iw2": {
            "cosine": [(np.float64(0.0), 1500)],
            "sine": [(np.float64(1.0), 1500)],
        },
        "qubitB5.resonator.readout.iw3": {
            "cosine": [(np.float64(-0.0), 1500)],
            "sine": [(np.float64(-1.0), 1500)],
        },
    },
    "mixers": {},
    "oscillators": {},
    "octaves": {
        "oct1": {
            "RF_outputs": {
                "1": {
                    "LO_frequency": 7550000000,
                    "LO_source": "internal",
                    "gain": -20,
                    "output_mode": "always_on",
                    "input_attenuators": "off",
                    "I_connection": ('con1', 1),
                    "Q_connection": ('con1', 2),
                },
                "2": {
                    "LO_frequency": 5100000000.0,
                    "LO_source": "internal",
                    "gain": 0,
                    "output_mode": "always_on",
                    "input_attenuators": "off",
                    "I_connection": ('con1', 3),
                    "Q_connection": ('con1', 4),
                },
                "3": {
                    "LO_frequency": 5100000000.0,
                    "LO_source": "internal",
                    "gain": 0,
                    "output_mode": "always_on",
                    "input_attenuators": "off",
                    "I_connection": ('con1', 5),
                    "Q_connection": ('con1', 6),
                },
                "4": {
                    "LO_frequency": 5850000000.0,
                    "LO_source": "internal",
                    "gain": 0,
                    "output_mode": "always_on",
                    "input_attenuators": "off",
                    "I_connection": ('con1', 7),
                    "Q_connection": ('con1', 8),
                },
                "5": {
                    "LO_frequency": 5850000000.0,
                    "LO_source": "internal",
                    "gain": 0,
                    "output_mode": "always_on",
                    "input_attenuators": "off",
                    "I_connection": ('con1', 9),
                    "Q_connection": ('con1', 10),
                },
            },
            "IF_outputs": {
                "IF_out1": {
                    "port": ('con1', 1),
                    "name": "out1",
                },
                "IF_out2": {
                    "port": ('con1', 2),
                    "name": "out2",
                },
            },
            "RF_inputs": {
                "1": {
                    "RF_source": "RF_in",
                    "LO_frequency": 7550000000,
                    "LO_source": "internal",
                    "IF_mode_I": "direct",
                    "IF_mode_Q": "direct",
                },
            },
            "loopbacks": [],
        },
        "oct2": {
            "RF_outputs": {
                "1": {
                    "LO_frequency": 7300000000,
                    "LO_source": "internal",
                    "gain": -20,
                    "output_mode": "always_on",
                    "input_attenuators": "off",
                    "I_connection": ('con3', 1),
                    "Q_connection": ('con3', 2),
                },
            },
            "IF_outputs": {
                "IF_out1": {
                    "port": ('con3', 1),
                    "name": "out1",
                },
                "IF_out2": {
                    "port": ('con3', 2),
                    "name": "out2",
                },
            },
            "RF_inputs": {
                "1": {
                    "RF_source": "RF_in",
                    "LO_frequency": 7300000000,
                    "LO_source": "internal",
                    "IF_mode_I": "direct",
                    "IF_mode_Q": "direct",
                },
            },
            "loopbacks": [],
        },
    },
}

loaded_config = {
    "version": 1,
    "controllers": {
        "con3": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "2": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
            },
            "analog_inputs": {
                "1": {
                    "offset": 0.018978428548177082,
                    "gain_db": 0,
                    "shareable": False,
                    "sampling_rate": 1000000000.0,
                },
                "2": {
                    "offset": 0.01666322998046875,
                    "gain_db": 0,
                    "shareable": False,
                    "sampling_rate": 1000000000.0,
                },
            },
            "digital_outputs": {},
            "digital_inputs": {},
        },
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                "3": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "4": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "7": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "8": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "9": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "10": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "5": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "6": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "2": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
            },
            "analog_inputs": {
                "1": {
                    "offset": 0.018978428548177082,
                    "gain_db": 0,
                    "shareable": False,
                    "sampling_rate": 1000000000.0,
                },
                "2": {
                    "offset": 0.01666322998046875,
                    "gain_db": 0,
                    "shareable": False,
                    "sampling_rate": 1000000000.0,
                },
            },
            "digital_outputs": {},
            "digital_inputs": {},
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.7437646298079738, -1.5474800067682553, 0.9638397736653462, -0.09863832640526682, -0.1525128963855616, 0.16087758901132937, -0.09051889113973684, 0.010077582254409289, 0.027090116685094576, 0.00033456171915884417, -0.023190809959393446, -0.006128571097164163, 0.0360283330805724, -0.03864695005187116, 0.01870973771548932, 0.032648599368585274, -0.07137166532104003, 0.03161425556665239, 0.020460680461686618, -0.012764200827653933, 0.003101094270150914, -0.02041290517345084, 0.01920022493232309, -0.002568138962594708, 0.008915921192951131, -0.01062843563049461, -0.0226370653075193, 0.045729720758981615, -0.02947007539168625, 0.006815779492935853],
                        "feedback": [0.9763347431904402, 0.8106993503581621],
                    },
                    "crosstalk": {},
                },
                "2": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.7516121544766298, -1.9900000000000002, 1.9352042705802737, -0.9391889804531905, 0.2560644771647802, 0.0622433374241618, -0.13781447384058654, 0.07253115151545006, 0.007170716680246397, -0.020358308641562968, -0.009300247208301833, 0.02580573838921365, -0.02819064606791724, 0.02233397470094777, -0.006095696153540007, -0.0026071488754044417, 0.0036733424820162637, -0.025602299378631815, 0.036947053112735856, -0.007752460082591457, -0.017146474427837174, 0.01542945596577327, -0.00021144883384914472, -0.013632865939604102, 0.017421689234526017, -0.012362467979236172, -0.0016104040917614966, 0.01622023982253356, -0.015885057774454616, 0.0051438745487504215],
                        "feedback": [0.9967005270060615, 0.9633584230394913],
                    },
                    "crosstalk": {},
                },
                "3": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.4615102116871844, -0.9844153841725033, 0.8617726427003987, -0.7865602449145748, 0.6966284135425775, -0.38845876334383916, 0.29793158268536934, -0.29409701936781474, 0.20869968861210988, -0.10254049942400192, 0.04557950632739816, -0.0189023102296481, 0.016472666176946954, -0.06339552097679621, 0.07537183271094809, 0.023932502404315128, -0.10935043620716622, 0.07656930841518865, -0.01922165930378962, 0.007908023395297123, 0.018280825322080675, -0.09818188557009226, 0.1396180303557303, -0.07665883440172865, 0.025040434050241707, -0.07035734178615066, 0.11050746321359928, -0.07044311107294293, 0.017851054460900014, -0.0010647441292200319],
                        "feedback": [0.9979573425801123, 0.9730675990337674],
                    },
                    "crosstalk": {},
                },
                "4": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [0.7282743233046902, -1.471131009932805, 0.8496962507290762, -0.022380144883349558, -0.1844587784789683, 0.18097801315018458, -0.10569973585271662, 0.01139040408765742, 0.03910721019202143, -0.005248985025044176, -0.05110632779021178, 0.05371329375151368, -0.02381837653035794, -0.004868089050569556, 0.037269884282889076, -0.062231724409859704, 0.051949994736028314, -0.023745429810177455, -0.005616786010352695, 0.0280981651278796, -0.03261722284368266, 0.025263321836419175, -0.006843487643150184, -0.0038878100767963373, -0.006896053134760072, 0.011929893011419383, 0.002618162155459523, -0.010401845109661835, 0.003455397632138337, 0.00045243533493845956],
                        "feedback": [0.9662288085546595, 0.8077992061763967],
                    },
                    "crosstalk": {},
                },
                "5": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "8": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "9": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
                "10": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
            },
            "analog_inputs": {},
            "digital_outputs": {},
            "digital_inputs": {},
        },
    },
    "oscillators": {},
    "elements": {
        "qubitC1.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "qubitC1.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC1.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC1.xy.x180_Square.pulse",
                "x90_Square": "qubitC1.xy.x90_Square.pulse",
                "-x90_Square": "qubitC1.xy.-x90_Square.pulse",
                "y180_Square": "qubitC1.xy.y180_Square.pulse",
                "y90_Square": "qubitC1.xy.y90_Square.pulse",
                "-y90_Square": "qubitC1.xy.-y90_Square.pulse",
                "x180": "qubitC1.xy.x180_DragCosine.pulse",
                "x90": "qubitC1.xy.x90_DragCosine.pulse",
                "-x90": "qubitC1.xy.-x90_DragCosine.pulse",
                "y180": "qubitC1.xy.y180_DragCosine.pulse",
                "y90": "qubitC1.xy.y90_DragCosine.pulse",
                "-y90": "qubitC1.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC1.xy.saturation.pulse",
                "EF_x180": "qubitC1.xy.EF_x180.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 3),
                "Q": ('con1', 1, 4),
                "mixer": "qubitC1.xy_mixer_5f3",
                "lo_frequency": 5100000000.0,
            },
            "intermediate_frequency": -161579893.20423436,
        },
        "qubitC1.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitC1.z.const.pulse",
                "z0": "qubitC1.z.z0.pulse",
                "z90": "qubitC1.z.z90.pulse",
                "z180": "qubitC1.z.z180.pulse",
                "-z90": "qubitC1.z.-z90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 1),
            },
        },
        "qubitC1.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con1', 1, 1),
                "out2": ('con1', 1, 2),
            },
            "operations": {
                "readout": "qubitC1.resonator.readout.pulse",
                "const": "qubitC1.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 1),
                "Q": ('con1', 1, 2),
                "mixer": "qubitC1.resonator_mixer_256",
                "lo_frequency": 7550000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": -193703946.0,
        },
        "qubitC2.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "qubitC2.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC2.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC2.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC2.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC2.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC2.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC2.xy.x180_Square.pulse",
                "x90_Square": "qubitC2.xy.x90_Square.pulse",
                "-x90_Square": "qubitC2.xy.-x90_Square.pulse",
                "y180_Square": "qubitC2.xy.y180_Square.pulse",
                "y90_Square": "qubitC2.xy.y90_Square.pulse",
                "-y90_Square": "qubitC2.xy.-y90_Square.pulse",
                "x180": "qubitC2.xy.x180_Square.pulse",
                "x90": "qubitC2.xy.x90_Square.pulse",
                "-x90": "qubitC2.xy.-x90_DragCosine.pulse",
                "y180": "qubitC2.xy.y180_DragCosine.pulse",
                "y90": "qubitC2.xy.y90_DragCosine.pulse",
                "-y90": "qubitC2.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC2.xy.saturation.pulse",
                "EF_x180": "qubitC2.xy.EF_x180.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 7),
                "Q": ('con1', 1, 8),
                "mixer": "qubitC2.xy_mixer_ddb",
                "lo_frequency": 5850000000.0,
            },
            "intermediate_frequency": -102724629.50200598,
        },
        "qubitC2.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitC2.z.const.pulse",
                "z0": "qubitC2.z.z0.pulse",
                "z90": "qubitC2.z.z90.pulse",
                "z180": "qubitC2.z.z180.pulse",
                "-z90": "qubitC2.z.-z90.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC4": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse",
                "Cz.CZ_snz_qubitC4": "qubitC2.z.Cz.CZ_snz_qubitC4.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC1": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse",
                "Cz_SNZ.CZ_snz_qubitC1": "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 2),
            },
        },
        "qubitC2.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con1', 1, 1),
                "out2": ('con1', 1, 2),
            },
            "operations": {
                "readout": "qubitC2.resonator.readout.pulse",
                "const": "qubitC2.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 1),
                "Q": ('con1', 1, 2),
                "mixer": "qubitC2.resonator_mixer_66f",
                "lo_frequency": 7550000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": 66883846.0,
        },
        "qubitC3.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "qubitC3.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC3.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC3.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC3.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC3.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC3.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC3.xy.x180_Square.pulse",
                "x90_Square": "qubitC3.xy.x90_Square.pulse",
                "-x90_Square": "qubitC3.xy.-x90_Square.pulse",
                "y180_Square": "qubitC3.xy.y180_Square.pulse",
                "y90_Square": "qubitC3.xy.y90_Square.pulse",
                "-y90_Square": "qubitC3.xy.-y90_Square.pulse",
                "x180": "qubitC3.xy.x180_DragCosine.pulse",
                "x90": "qubitC3.xy.x90_DragCosine.pulse",
                "-x90": "qubitC3.xy.-x90_DragCosine.pulse",
                "y180": "qubitC3.xy.y180_DragCosine.pulse",
                "y90": "qubitC3.xy.y90_DragCosine.pulse",
                "-y90": "qubitC3.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC3.xy.saturation.pulse",
                "EF_x180": "qubitC3.xy.EF_x180.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 9),
                "Q": ('con1', 1, 10),
                "mixer": "qubitC3.xy_mixer_fa2",
                "lo_frequency": 5850000000.0,
            },
            "intermediate_frequency": -235169505.05035493,
        },
        "qubitC3.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitC3.z.const.pulse",
                "z0": "qubitC3.z.z0.pulse",
                "z90": "qubitC3.z.z90.pulse",
                "z180": "qubitC3.z.z180.pulse",
                "-z90": "qubitC3.z.-z90.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC4": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse",
                "Cz.CZ_snz_qubitC4": "qubitC3.z.Cz.CZ_snz_qubitC4.pulse",
                "Cz_unipolar.flux_pulse_control_qubitC1": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse",
                "Cz.CZ_snz_qubitC1": "qubitC3.z.Cz.CZ_snz_qubitC1.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 3),
            },
        },
        "qubitC3.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con3', 1, 1),
                "out2": ('con3', 1, 2),
            },
            "operations": {
                "readout": "qubitC3.resonator.readout.pulse",
                "const": "qubitC3.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con3', 1, 1),
                "Q": ('con3', 1, 2),
                "mixer": "qubitC3.resonator_mixer_293",
                "lo_frequency": 7300000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": 173982913.0,
        },
        "qubitC4.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "qubitC4.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "qubitC4.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "qubitC4.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "qubitC4.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "qubitC4.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "qubitC4.xy.-y90_DragCosine.pulse",
                "x180_Square": "qubitC4.xy.x180_Square.pulse",
                "x90_Square": "qubitC4.xy.x90_Square.pulse",
                "-x90_Square": "qubitC4.xy.-x90_Square.pulse",
                "y180_Square": "qubitC4.xy.y180_Square.pulse",
                "y90_Square": "qubitC4.xy.y90_Square.pulse",
                "-y90_Square": "qubitC4.xy.-y90_Square.pulse",
                "x180": "qubitC4.xy.x180_DragCosine.pulse",
                "x90": "qubitC4.xy.x90_DragCosine.pulse",
                "-x90": "qubitC4.xy.-x90_DragCosine.pulse",
                "y180": "qubitC4.xy.y180_DragCosine.pulse",
                "y90": "qubitC4.xy.y90_DragCosine.pulse",
                "-y90": "qubitC4.xy.-y90_DragCosine.pulse",
                "saturation": "qubitC4.xy.saturation.pulse",
                "EF_x180": "qubitC4.xy.EF_x180.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 5),
                "Q": ('con1', 1, 6),
                "mixer": "qubitC4.xy_mixer_868",
                "lo_frequency": 5100000000.0,
            },
            "intermediate_frequency": -115634512.22316144,
        },
        "qubitC4.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitC4.z.const.pulse",
                "z0": "qubitC4.z.z0.pulse",
                "z90": "qubitC4.z.z90.pulse",
                "z180": "qubitC4.z.z180.pulse",
                "-z90": "qubitC4.z.-z90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 4),
            },
        },
        "qubitC4.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con3', 1, 1),
                "out2": ('con3', 1, 2),
            },
            "operations": {
                "readout": "qubitC4.resonator.readout.pulse",
                "const": "qubitC4.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con3', 1, 1),
                "Q": ('con3', 1, 2),
                "mixer": "qubitC4.resonator_mixer_b16",
                "lo_frequency": 7300000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": -85478390.0,
        },
        "qubitC5.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitC5.z.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 5),
            },
        },
        "qubitC5.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con3', 1, 1),
                "out2": ('con3', 1, 2),
            },
            "operations": {
                "readout": "qubitC5.resonator.readout.pulse",
                "const": "qubitC5.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con3', 1, 1),
                "Q": ('con3', 1, 2),
                "mixer": "qubitC5.resonator_mixer_11c",
                "lo_frequency": 7300000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": 315375522.0,
        },
        "qubitB2.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitB2.z.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 2),
            },
        },
        "qubitB2.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con1', 1, 1),
                "out2": ('con1', 1, 2),
            },
            "operations": {
                "readout": "qubitB2.resonator.readout.pulse",
                "const": "qubitB2.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 1),
                "Q": ('con1', 1, 2),
                "mixer": "qubitB2.resonator_mixer_aab",
                "lo_frequency": 7550000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": -176200000.0,
        },
        "qubitB4.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitB4.z.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 9),
            },
        },
        "qubitB4.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con1', 1, 1),
                "out2": ('con1', 1, 2),
            },
            "operations": {
                "readout": "qubitB4.resonator.readout.pulse",
                "const": "qubitB4.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 1),
                "Q": ('con1', 1, 2),
                "mixer": "qubitB4.resonator_mixer_bc5",
                "lo_frequency": 7550000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": 137955078.0,
        },
        "qubitB5.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "qubitB5.z.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "singleInput": {
                "port": ('con2', 1, 10),
            },
        },
        "qubitB5.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con1', 1, 1),
                "out2": ('con1', 1, 2),
            },
            "operations": {
                "readout": "qubitB5.resonator.readout.pulse",
                "const": "qubitB5.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "",
            "mixInputs": {
                "I": ('con1', 1, 1),
                "Q": ('con1', 1, 2),
                "mixer": "qubitB5.resonator_mixer_6ab",
                "lo_frequency": 7550000000.0,
            },
            "smearing": 0,
            "time_of_flight": 264,
            "intermediate_frequency": 64300000.0,
        },
    },
    "pulses": {
        "const_pulse": {
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC1.xy.x180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC1.xy.x180_DragCosine.wf.I",
                "Q": "qubitC1.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC1.xy.x90_DragCosine.wf.I",
                "Q": "qubitC1.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.-x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC1.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC1.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.y180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC1.xy.y180_DragCosine.wf.I",
                "Q": "qubitC1.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC1.xy.y90_DragCosine.wf.I",
                "Q": "qubitC1.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.-y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC1.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC1.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.x180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC1.xy.x180_Square.wf.I",
                "Q": "qubitC1.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC1.xy.x90_Square.wf.I",
                "Q": "qubitC1.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.-x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC1.xy.-x90_Square.wf.I",
                "Q": "qubitC1.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.y180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC1.xy.y180_Square.wf.I",
                "Q": "qubitC1.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC1.xy.y90_Square.wf.I",
                "Q": "qubitC1.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.-y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC1.xy.-y90_Square.wf.I",
                "Q": "qubitC1.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "qubitC1.xy.saturation.wf.I",
                "Q": "qubitC1.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.xy.EF_x180.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC1.xy.EF_x180.wf.I",
                "Q": "qubitC1.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC1.z.const.pulse": {
            "length": 60,
            "waveforms": {
                "single": "qubitC1.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC1.z.z0.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.z0.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC1.z.z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC1.z.z180.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.z180.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC1.z.-z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC1.z.-z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC1.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitC1.resonator.readout.wf.I",
                "Q": "qubitC1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC1.resonator.readout.iw1",
                "iw2": "qubitC1.resonator.readout.iw2",
                "iw3": "qubitC1.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitC1.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC1.resonator.const.wf.I",
                "Q": "qubitC1.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.xy.x180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC2.xy.x180_DragCosine.wf.I",
                "Q": "qubitC2.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC2.xy.x90_DragCosine.wf.I",
                "Q": "qubitC2.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.-x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC2.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC2.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.y180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC2.xy.y180_DragCosine.wf.I",
                "Q": "qubitC2.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC2.xy.y90_DragCosine.wf.I",
                "Q": "qubitC2.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.-y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC2.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC2.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.x180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC2.xy.x180_Square.wf.I",
                "Q": "qubitC2.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC2.xy.x90_Square.wf.I",
                "Q": "qubitC2.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.-x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC2.xy.-x90_Square.wf.I",
                "Q": "qubitC2.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.y180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC2.xy.y180_Square.wf.I",
                "Q": "qubitC2.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC2.xy.y90_Square.wf.I",
                "Q": "qubitC2.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.-y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC2.xy.-y90_Square.wf.I",
                "Q": "qubitC2.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "qubitC2.xy.saturation.wf.I",
                "Q": "qubitC2.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.xy.EF_x180.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC2.xy.EF_x180.wf.I",
                "Q": "qubitC2.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC2.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "qubitC2.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.z0.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.z0.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.z180.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.z180.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.-z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC2.z.-z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitC2.resonator.readout.wf.I",
                "Q": "qubitC2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC2.resonator.readout.iw1",
                "iw2": "qubitC2.resonator.readout.iw2",
                "iw3": "qubitC2.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitC2.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC2.resonator.const.wf.I",
                "Q": "qubitC2.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.xy.x180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC3.xy.x180_DragCosine.wf.I",
                "Q": "qubitC3.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC3.xy.x90_DragCosine.wf.I",
                "Q": "qubitC3.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.-x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC3.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC3.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.y180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC3.xy.y180_DragCosine.wf.I",
                "Q": "qubitC3.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC3.xy.y90_DragCosine.wf.I",
                "Q": "qubitC3.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.-y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC3.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC3.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.x180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC3.xy.x180_Square.wf.I",
                "Q": "qubitC3.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC3.xy.x90_Square.wf.I",
                "Q": "qubitC3.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.-x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC3.xy.-x90_Square.wf.I",
                "Q": "qubitC3.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.y180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC3.xy.y180_Square.wf.I",
                "Q": "qubitC3.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC3.xy.y90_Square.wf.I",
                "Q": "qubitC3.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.-y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC3.xy.-y90_Square.wf.I",
                "Q": "qubitC3.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "qubitC3.xy.saturation.wf.I",
                "Q": "qubitC3.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.xy.EF_x180.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC3.xy.EF_x180.wf.I",
                "Q": "qubitC3.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC3.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "qubitC3.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.z0.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.z0.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.z180.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.z180.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.-z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC3.z.-z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitC3.resonator.readout.wf.I",
                "Q": "qubitC3.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC3.resonator.readout.iw1",
                "iw2": "qubitC3.resonator.readout.iw2",
                "iw3": "qubitC3.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitC3.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC3.resonator.const.wf.I",
                "Q": "qubitC3.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC4.xy.x180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC4.xy.x180_DragCosine.wf.I",
                "Q": "qubitC4.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC4.xy.x90_DragCosine.wf.I",
                "Q": "qubitC4.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.-x90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC4.xy.-x90_DragCosine.wf.I",
                "Q": "qubitC4.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.y180_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC4.xy.y180_DragCosine.wf.I",
                "Q": "qubitC4.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC4.xy.y90_DragCosine.wf.I",
                "Q": "qubitC4.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.-y90_DragCosine.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC4.xy.-y90_DragCosine.wf.I",
                "Q": "qubitC4.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.x180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC4.xy.x180_Square.wf.I",
                "Q": "qubitC4.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC4.xy.x90_Square.wf.I",
                "Q": "qubitC4.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.-x90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC4.xy.-x90_Square.wf.I",
                "Q": "qubitC4.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.y180_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC4.xy.y180_Square.wf.I",
                "Q": "qubitC4.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC4.xy.y90_Square.wf.I",
                "Q": "qubitC4.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.-y90_Square.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC4.xy.-y90_Square.wf.I",
                "Q": "qubitC4.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "qubitC4.xy.saturation.wf.I",
                "Q": "qubitC4.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.xy.EF_x180.pulse": {
            "length": 32,
            "waveforms": {
                "I": "qubitC4.xy.EF_x180.wf.I",
                "Q": "qubitC4.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "qubitC4.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "qubitC4.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC4.z.z0.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.z0.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC4.z.z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC4.z.z180.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.z180.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC4.z.-z90.pulse": {
            "length": 32,
            "waveforms": {
                "single": "qubitC4.z.-z90.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC4.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitC4.resonator.readout.wf.I",
                "Q": "qubitC4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC4.resonator.readout.iw1",
                "iw2": "qubitC4.resonator.readout.iw2",
                "iw3": "qubitC4.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitC4.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC4.resonator.const.wf.I",
                "Q": "qubitC4.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC5.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "qubitC5.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC5.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitC5.resonator.readout.wf.I",
                "Q": "qubitC5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitC5.resonator.readout.iw1",
                "iw2": "qubitC5.resonator.readout.iw2",
                "iw3": "qubitC5.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitC5.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitC5.resonator.const.wf.I",
                "Q": "qubitC5.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitB2.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "qubitB2.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitB2.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitB2.resonator.readout.wf.I",
                "Q": "qubitB2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitB2.resonator.readout.iw1",
                "iw2": "qubitB2.resonator.readout.iw2",
                "iw3": "qubitB2.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitB2.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitB2.resonator.const.wf.I",
                "Q": "qubitB2.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitB4.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "qubitB4.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitB4.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitB4.resonator.readout.wf.I",
                "Q": "qubitB4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitB4.resonator.readout.iw1",
                "iw2": "qubitB4.resonator.readout.iw2",
                "iw3": "qubitB4.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitB4.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitB4.resonator.const.wf.I",
                "Q": "qubitB4.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitB5.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "qubitB5.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitB5.resonator.readout.pulse": {
            "length": 1500,
            "waveforms": {
                "I": "qubitB5.resonator.readout.wf.I",
                "Q": "qubitB5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "qubitB5.resonator.readout.iw1",
                "iw2": "qubitB5.resonator.readout.iw2",
                "iw3": "qubitB5.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "qubitB5.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "qubitB5.resonator.const.wf.I",
                "Q": "qubitB5.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse": {
            "length": 52,
            "waveforms": {
                "single": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.Cz.CZ_snz_qubitC4.pulse": {
            "length": 60,
            "waveforms": {
                "single": "qubitC2.z.Cz.CZ_snz_qubitC4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse": {
            "length": 52,
            "waveforms": {
                "single": "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.pulse": {
            "length": 60,
            "waveforms": {
                "single": "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.pulse": {
            "length": 44,
            "waveforms": {
                "single": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.Cz.CZ_snz_qubitC4.pulse": {
            "length": 52,
            "waveforms": {
                "single": "qubitC3.z.Cz.CZ_snz_qubitC4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.pulse": {
            "length": 48,
            "waveforms": {
                "single": "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "qubitC3.z.Cz.CZ_snz_qubitC1.pulse": {
            "length": 56,
            "waveforms": {
                "single": "qubitC3.z.Cz.CZ_snz_qubitC1.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "qubitC1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0012096453125145397, 0.004789058228836524, 0.010591697163608568, 0.018380001397059975, 0.027835116838786374, 0.038569949951674565, 0.050145015406793586, 0.06208642866439842, 0.07390530686266622, 0.08511778373950074, 0.09526481917415419, 0.10393099234370592, 0.1107615091003623, 0.11547672728555419] + [0.11788360531276522] * 2 + [0.11547672728555419, 0.1107615091003623, 0.10393099234370594, 0.0952648191741542, 0.08511778373950073, 0.07390530686266628, 0.06208642866439847, 0.050145015406793586, 0.038569949951674565, 0.027835116838786374, 0.018380001397059982, 0.010591697163608568, 0.004789058228836524, 0.0012096453125145527, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0013361039447842313, -0.0026175076370834487, -0.0037917502591763813, -0.004810758180146893, -0.005632813095982107, -0.006224259981840376, -0.006560884932722606, -0.0066289064835879236, -0.006425539824151712, -0.005959110809345857, -0.005248715097839595, -0.004323436373529937, -0.0032211556561057616, -0.0019870004476508973, -0.000671497207406568, 0.0006714972074065635, 0.001987000447650898, 0.00322115565610576, 0.004323436373529936, 0.005248715097839593, 0.005959110809345858, 0.006425539824151711, 0.0066289064835879236, 0.006560884932722606, 0.006224259981840376, 0.005632813095982108, 0.004810758180146894, 0.003791750259176382, 0.0026175076370834496, 0.0013361039447842384, 1.6257003961951508e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006027057769603683, 0.0023861482625177934, 0.00527731311176796, 0.009157835696085116, 0.013868846964925286, 0.019217477563421816, 0.024984753926434858, 0.030934563082036456, 0.03682331914432338, 0.042409935748206166, 0.04746569615352224, 0.05178361693525138, 0.055186921909255414, 0.05753627937002727] + [0.05873550634708516] * 2 + [0.05753627937002727, 0.055186921909255414, 0.05178361693525139, 0.047465696153522245, 0.04240993574820616, 0.03682331914432341, 0.030934563082036484, 0.024984753926434858, 0.019217477563421816, 0.013868846964925286, 0.00915783569608512, 0.00527731311176796, 0.0023861482625177934, 0.0006027057769603749, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0006657137904887421, -0.001304173180176826, -0.0018892395666346286, -0.0023969602632581853, -0.0028065491250730797, -0.0031012375359519622, -0.0032689609177290325, -0.0033028526554476773, -0.0032015252173835853, -0.0029691269607565684, -0.0026151722974985733, -0.0021541521731112873, -0.0016049408056546927, -0.0009900229730420576, -0.00033457348359032193, 0.00033457348359031965, 0.0009900229730420583, 0.0016049408056546923, 0.002154152173111287, 0.002615172297498573, 0.0029691269607565684, 0.0032015252173835844, 0.0033028526554476773, 0.0032689609177290325, 0.0031012375359519622, 0.00280654912507308, 0.002396960263258186, 0.001889239566634629, 0.0013041731801768266, 0.0006657137904887456, 8.100052224042324e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0006027057769603682, -0.0023861482625177934, -0.00527731311176796, -0.009157835696085116, -0.013868846964925286, -0.019217477563421816, -0.024984753926434858, -0.030934563082036456, -0.03682331914432338, -0.042409935748206166, -0.04746569615352224, -0.05178361693525138, -0.055186921909255414, -0.05753627937002727] + [-0.05873550634708516] * 2 + [-0.05753627937002727, -0.055186921909255414, -0.05178361693525139, -0.047465696153522245, -0.04240993574820616, -0.03682331914432341, -0.030934563082036484, -0.024984753926434858, -0.019217477563421816, -0.013868846964925286, -0.00915783569608512, -0.00527731311176796, -0.0023861482625177934, -0.000602705776960375, -9.919703029099832e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006657137904887422, 0.0013041731801768262, 0.0018892395666346292, 0.0023969602632581866, 0.0028065491250730814, 0.0031012375359519644, 0.0032689609177290356, 0.003302852655447681, 0.0032015252173835896, 0.0029691269607565736, 0.002615172297498579, 0.002154152173111294, 0.0016049408056546994, 0.0009900229730420646, 0.00033457348359032914, -0.00033457348359031244, -0.0009900229730420514, -0.0016049408056546856, -0.0021541521731112804, -0.0026151722974985673, -0.002969126960756563, -0.00320152521738358, -0.0033028526554476734, -0.0032689609177290295, -0.00310123753595196, -0.0028065491250730784, -0.002396960263258185, -0.0018892395666346283, -0.0013041731801768264, -0.0006657137904887455, -8.100052224042324e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0013361039447842313, 0.002617507637083449, 0.0037917502591763817, 0.004810758180146894, 0.005632813095982109, 0.006224259981840378, 0.0065608849327226096, 0.006628906483587927, 0.006425539824151717, 0.005959110809345862, 0.005248715097839601, 0.004323436373529943, 0.0032211556561057685, 0.0019870004476509042, 0.0006714972074065753, -0.0006714972074065562, -0.0019870004476508912, -0.003221155656105753, -0.00432343637352993, -0.005248715097839587, -0.005959110809345853, -0.006425539824151707, -0.00662890648358792, -0.006560884932722603, -0.006224259981840373, -0.005632813095982106, -0.004810758180146893, -0.0037917502591763817, -0.002617507637083449, -0.0013361039447842384, -1.6257003961951508e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0012096453125145397, 0.004789058228836524, 0.010591697163608568, 0.018380001397059975, 0.027835116838786374, 0.038569949951674565, 0.050145015406793586, 0.06208642866439842, 0.07390530686266622, 0.08511778373950074, 0.09526481917415419, 0.10393099234370592, 0.1107615091003623, 0.11547672728555419] + [0.11788360531276522] * 2 + [0.11547672728555419, 0.1107615091003623, 0.10393099234370594, 0.0952648191741542, 0.08511778373950073, 0.07390530686266628, 0.06208642866439847, 0.050145015406793586, 0.038569949951674565, 0.027835116838786374, 0.018380001397059982, 0.010591697163608568, 0.004789058228836524, 0.0012096453125145527, 9.954543932864876e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006657137904887421, 0.0013041731801768262, 0.0018892395666346288, 0.0023969602632581857, 0.0028065491250730806, 0.0031012375359519635, 0.0032689609177290343, 0.003302852655447679, 0.0032015252173835875, 0.002969126960756571, 0.0026151722974985764, 0.0021541521731112904, 0.0016049408056546962, 0.0009900229730420611, 0.0003345734835903255, -0.0003345734835903161, -0.0009900229730420548, -0.0016049408056546888, -0.002154152173111284, -0.00261517229749857, -0.0029691269607565658, -0.0032015252173835823, -0.0033028526554476755, -0.003268960917729031, -0.003101237535951961, -0.0028065491250730793, -0.0023969602632581857, -0.0018892395666346288, -0.0013041731801768264, -0.0006657137904887456, -8.100052224042324e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006027057769603683, 0.0023861482625177934, 0.00527731311176796, 0.009157835696085116, 0.013868846964925286, 0.019217477563421816, 0.024984753926434858, 0.030934563082036456, 0.03682331914432338, 0.042409935748206166, 0.04746569615352224, 0.05178361693525138, 0.055186921909255414, 0.05753627937002727] + [0.05873550634708516] * 2 + [0.05753627937002727, 0.055186921909255414, 0.05178361693525139, 0.047465696153522245, 0.04240993574820616, 0.03682331914432341, 0.030934563082036484, 0.024984753926434858, 0.019217477563421816, 0.013868846964925286, 0.00915783569608512, 0.00527731311176796, 0.0023861482625177934, 0.0006027057769603749, 4.959851514549916e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0006657137904887421, -0.0013041731801768258, -0.0018892395666346283, -0.002396960263258185, -0.002806549125073079, -0.003101237535951961, -0.003268960917729031, -0.0033028526554476755, -0.003201525217383583, -0.0029691269607565658, -0.0026151722974985703, -0.0021541521731112843, -0.0016049408056546892, -0.0009900229730420542, -0.00033457348359031835, 0.00033457348359032323, 0.0009900229730420618, 0.0016049408056546957, 0.00215415217311129, 0.002615172297498576, 0.002969126960756571, 0.0032015252173835866, 0.003302852655447679, 0.0032689609177290343, 0.0031012375359519635, 0.002806549125073081, 0.0023969602632581866, 0.0018892395666346292, 0.0013041731801768269, 0.0006657137904887456, 8.100052224042324e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0006027057769603683, -0.0023861482625177934, -0.00527731311176796, -0.009157835696085116, -0.013868846964925286, -0.019217477563421816, -0.024984753926434858, -0.030934563082036456, -0.03682331914432338, -0.042409935748206166, -0.04746569615352224, -0.05178361693525138, -0.055186921909255414, -0.05753627937002727] + [-0.05873550634708516] * 2 + [-0.05753627937002727, -0.055186921909255414, -0.05178361693525139, -0.047465696153522245, -0.04240993574820616, -0.03682331914432341, -0.030934563082036484, -0.024984753926434858, -0.019217477563421816, -0.013868846964925286, -0.00915783569608512, -0.00527731311176796, -0.0023861482625177934, -0.0006027057769603749, 4.959851514549916e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.03066287395078562,
        },
        "qubitC1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0008152112067846226, 0.0032274699845494777, 0.0071380181755045, 0.012386757477242083, 0.018758803886048423, 0.02599328507326331, 0.03379402052906672, 0.04184164722735058, 0.04980669438553113, 0.05736307203709921, 0.06420142119316287, 0.07004177903580103, 0.07464504062872312, 0.07782274790145467] + [0.07944480514488529] * 2 + [0.07782274790145467, 0.07464504062872312, 0.07004177903580104, 0.06420142119316288, 0.0573630720370992, 0.04980669438553117, 0.04184164722735061, 0.03379402052906672, 0.02599328507326331, 0.018758803886048423, 0.012386757477242088, 0.0071380181755045, 0.0032274699845494777, 0.0008152112067846313, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0006000019617158705, -0.0011754397726218564, -0.0017027549411283943, -0.002160359122279925, -0.0025295179471338967, -0.002795118010026468, -0.002946285613176134, -0.0029768319371485086, -0.0028855064118511226, -0.002676047915019943, -0.0023570317021255694, -0.0019415183344067071, -0.0014465189779585168, -0.0008922989645939421, -0.00030154812677821575, 0.00030154812677821375, 0.0008922989645939426, 0.0014465189779585162, 0.0019415183344067065, 0.0023570317021255685, 0.002676047915019943, 0.0028855064118511217, 0.0029768319371485086, 0.002946285613176134, 0.002795118010026468, 0.002529517947133897, 0.0021603591222799255, 0.0017027549411283945, 0.0011754397726218568, 0.0006000019617158738, 7.300505553382528e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC1.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC1.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC1.z.z90.wf": {
            "type": "constant",
            "sample": 0.018880207703291515,
        },
        "qubitC1.z.z180.wf": {
            "type": "constant",
            "sample": 0.026590269797742615,
        },
        "qubitC1.z.-z90.wf": {
            "type": "constant",
            "sample": 0.032704782390436345,
        },
        "qubitC1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.018244444444444445,
        },
        "qubitC1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0011363473587025924, 0.00449886724042994, 0.009949897686144812, 0.017266272868930756, 0.026148459529166952, 0.036232819901370686, 0.04710649908704191, 0.0583243273827162, 0.06942704557992978, 0.07996010709117475, 0.08949228714059851, 0.09763333715699413, 0.10404996159615755, 0.10847946309962808] + [0.1107404973554093] * 2 + [0.10847946309962808, 0.10404996159615755, 0.09763333715699414, 0.08949228714059852, 0.07996010709117474, 0.06942704557992982, 0.05832432738271626, 0.04710649908704191, 0.036232819901370686, 0.026148459529166952, 0.017266272868930763, 0.009949897686144812, 0.00449886724042994, 0.0011363473587026047, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.002210342435467431, -0.004330193191922616, -0.006272765330324393, -0.007958529719084057, -0.009318470966056883, -0.010296912916805493, -0.010853798042904188, -0.010966327401859264, -0.010629894028493465, -0.009858271544641627, -0.008683050265454439, -0.007152343888184455, -0.005328819711767478, -0.003287133030241011, -0.001110870736234749, 0.0011108707362347417, 0.0032871330302410126, 0.005328819711767476, 0.0071523438881844524, 0.008683050265454437, 0.009858271544641627, 0.010629894028493464, 0.010966327401859264, 0.010853798042904188, 0.010296912916805493, 0.009318470966056886, 0.007958529719084059, 0.006272765330324395, 0.004330193191922618, 0.0022103424354674426, 2.6894274110137822e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005656168977942143, 0.0022393111689239984, 0.004952561573278572, 0.008594287320510268, 0.013015395730642827, 0.018034886105907224, 0.02344725992057507, 0.02903093395474694, 0.03455731193740998, 0.03980014330463216, 0.044544785924232826, 0.04859699356989374, 0.05179086838448733, 0.05399565275783978] + [0.05512108255865488] * 2 + [0.05399565275783978, 0.05179086838448733, 0.04859699356989375, 0.04454478592423283, 0.039800143304632156, 0.03455731193741001, 0.029030933954746964, 0.02344725992057507, 0.018034886105907224, 0.013015395730642827, 0.008594287320510272, 0.004952561573278572, 0.0022393111689239984, 0.0005656168977942205, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0011001979472539116, -0.002155353661279478, -0.003122268943168961, -0.003961358167674082, -0.004638268923354805, -0.005125288404339925, -0.005402477975855549, -0.005458489464275439, -0.0052910297526826125, -0.00490695466134536, -0.004321988269629939, -0.003560079170343806, -0.0026524200115322573, -0.0016361704658024601, -0.0005529359089608455, 0.0005529359089608417, 0.001636170465802461, 0.002652420011532256, 0.003560079170343805, 0.004321988269629938, 0.004906954661345361, 0.005291029752682612, 0.005458489464275439, 0.005402477975855549, 0.005125288404339925, 0.004638268923354806, 0.003961358167674083, 0.0031222689431689614, 0.0021553536612794787, 0.0011001979472539175, 1.3386624938321075e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005656168977942142, -0.002239311168923998, -0.004952561573278572, -0.008594287320510268, -0.013015395730642827, -0.018034886105907224, -0.02344725992057507, -0.02903093395474694, -0.03455731193740998, -0.03980014330463216, -0.044544785924232826, -0.04859699356989374, -0.05179086838448733, -0.05399565275783978] + [-0.05512108255865488] * 2 + [-0.05399565275783978, -0.05179086838448733, -0.04859699356989375, -0.04454478592423283, -0.039800143304632156, -0.03455731193741001, -0.029030933954746964, -0.02344725992057507, -0.018034886105907224, -0.013015395730642827, -0.008594287320510272, -0.004952561573278572, -0.002239311168923999, -0.0005656168977942206, -1.639388738210104e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0011001979472539116, 0.0021553536612794783, 0.0031222689431689614, 0.003961358167674083, 0.004638268923354807, 0.005125288404339927, 0.005402477975855552, 0.005458489464275442, 0.005291029752682617, 0.0049069546613453655, 0.004321988269629944, 0.0035600791703438123, 0.002652420011532264, 0.0016361704658024666, 0.0005529359089608522, -0.0005529359089608349, -0.0016361704658024545, -0.0026524200115322495, -0.003560079170343799, -0.004321988269629933, -0.004906954661345356, -0.005291029752682607, -0.005458489464275435, -0.005402477975855547, -0.005125288404339922, -0.004638268923354804, -0.003961358167674082, -0.003122268943168961, -0.0021553536612794783, -0.0011001979472539175, -1.3386624938321075e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002210342435467431, 0.004330193191922616, 0.006272765330324394, 0.007958529719084059, 0.009318470966056884, 0.010296912916805494, 0.010853798042904192, 0.010966327401859268, 0.010629894028493469, 0.009858271544641632, 0.008683050265454444, 0.007152343888184461, 0.005328819711767484, 0.0032871330302410174, 0.0011108707362347558, -0.001110870736234735, -0.003287133030241006, -0.00532881971176747, -0.007152343888184446, -0.008683050265454432, -0.009858271544641622, -0.01062989402849346, -0.01096632740185926, -0.010853798042904185, -0.010296912916805491, -0.009318470966056884, -0.007958529719084057, -0.006272765330324394, -0.004330193191922618, -0.0022103424354674426, -2.6894274110137822e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0011363473587025922, 0.00449886724042994, 0.009949897686144812, 0.017266272868930756, 0.026148459529166952, 0.036232819901370686, 0.04710649908704191, 0.0583243273827162, 0.06942704557992978, 0.07996010709117475, 0.08949228714059851, 0.09763333715699413, 0.10404996159615755, 0.10847946309962808] + [0.1107404973554093] * 2 + [0.10847946309962808, 0.10404996159615755, 0.09763333715699414, 0.08949228714059852, 0.07996010709117474, 0.06942704557992982, 0.05832432738271626, 0.04710649908704191, 0.036232819901370686, 0.026148459529166952, 0.017266272868930763, 0.009949897686144812, 0.00449886724042994, 0.001136347358702605, 1.6467993352185907e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0011001979472539116, 0.002155353661279478, 0.0031222689431689614, 0.003961358167674083, 0.004638268923354806, 0.005125288404339926, 0.005402477975855551, 0.0054584894642754405, 0.005291029752682614, 0.004906954661345363, 0.004321988269629941, 0.0035600791703438092, 0.0026524200115322603, 0.0016361704658024634, 0.0005529359089608488, -0.0005529359089608383, -0.0016361704658024577, -0.002652420011532253, -0.003560079170343802, -0.004321988269629935, -0.0049069546613453585, -0.00529102975268261, -0.005458489464275437, -0.0054024779758555476, -0.005125288404339924, -0.004638268923354805, -0.003961358167674082, -0.003122268943168961, -0.0021553536612794787, -0.0011001979472539175, -1.3386624938321075e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005656168977942142, 0.0022393111689239984, 0.004952561573278572, 0.008594287320510268, 0.013015395730642827, 0.018034886105907224, 0.02344725992057507, 0.02903093395474694, 0.03455731193740998, 0.03980014330463216, 0.044544785924232826, 0.04859699356989374, 0.05179086838448733, 0.05399565275783978] + [0.05512108255865488] * 2 + [0.05399565275783978, 0.05179086838448733, 0.04859699356989375, 0.04454478592423283, 0.039800143304632156, 0.03455731193741001, 0.029030933954746964, 0.02344725992057507, 0.018034886105907224, 0.013015395730642827, 0.008594287320510272, 0.004952561573278572, 0.0022393111689239984, 0.0005656168977942206, 8.19694369105052e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0011001979472539116, -0.002155353661279478, -0.0031222689431689606, -0.003961358167674081, -0.004638268923354804, -0.005125288404339924, -0.0054024779758555476, -0.005458489464275437, -0.005291029752682611, -0.004906954661345358, -0.004321988269629936, -0.003560079170343803, -0.0026524200115322543, -0.0016361704658024569, -0.0005529359089608421, 0.000552935908960845, 0.0016361704658024643, 0.002652420011532259, 0.003560079170343808, 0.0043219882696299405, 0.004906954661345364, 0.005291029752682613, 0.0054584894642754405, 0.005402477975855551, 0.005125288404339926, 0.004638268923354807, 0.003961358167674084, 0.003122268943168962, 0.0021553536612794787, 0.0011001979472539175, 1.3386624938321075e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005656168977942144, -0.0022393111689239984, -0.004952561573278572, -0.008594287320510268, -0.013015395730642827, -0.018034886105907224, -0.02344725992057507, -0.02903093395474694, -0.03455731193740998, -0.03980014330463216, -0.044544785924232826, -0.04859699356989374, -0.05179086838448733, -0.05399565275783978] + [-0.05512108255865488] * 2 + [-0.05399565275783978, -0.05179086838448733, -0.04859699356989375, -0.04454478592423283, -0.039800143304632156, -0.03455731193741001, -0.029030933954746964, -0.02344725992057507, -0.018034886105907224, -0.013015395730642827, -0.008594287320510272, -0.004952561573278572, -0.0022393111689239984, -0.0005656168977942204, 8.19694369105052e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC2.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC2.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC2.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.02976981820061825,
        },
        "qubitC2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0010294361571335725, 0.004075599391307629, 0.0090137794218061, 0.015641806577968854, 0.023688328649262486, 0.03282392007666201, 0.04267456867374328, 0.052836987729539736, 0.06289512661387352, 0.07243720393903888, 0.08107256593819703, 0.08844767987649055, 0.09426060772274655, 0.09827336752794931] + [0.10032167643419457] * 2 + [0.09827336752794931, 0.09426060772274655, 0.08844767987649056, 0.08107256593819705, 0.07243720393903888, 0.06289512661387356, 0.052836987729539785, 0.04267456867374328, 0.03282392007666201, 0.023688328649262486, 0.015641806577968858, 0.0090137794218061, 0.004075599391307629, 0.0010294361571335838, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0014928327546150184, -0.002924548760855697, -0.004236533397207055, -0.005375073858505217, -0.006293558164491814, -0.00695438345976243, -0.0073304954790850115, -0.007406496152197205, -0.007179274002612097, -0.006658131531831172, -0.0058644043738398455, -0.004830587811745254, -0.003599006417067845, -0.002220081276810381, -0.0007502657482318286, 0.0007502657482318235, 0.0022200812768103824, 0.003599006417067844, 0.0048305878117452525, 0.005864404373839844, 0.006658131531831174, 0.0071792740026120965, 0.007406496152197205, 0.0073304954790850115, 0.00695438345976243, 0.006293558164491814, 0.005375073858505217, 0.004236533397207055, 0.0029245487608556982, 0.0014928327546150262, 1.816399697122858e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC2.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC2.z.z90.wf": {
            "type": "constant",
            "sample": 0.01729078010516483,
        },
        "qubitC2.z.z180.wf": {
            "type": "constant",
            "sample": 0.02457771966234662,
        },
        "qubitC2.z.-z90.wf": {
            "type": "constant",
            "sample": 0.03028389262492936,
        },
        "qubitC2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.018244444444444445,
        },
        "qubitC2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.002128256352676728, 0.008425894345568822, 0.01863508793930623, 0.032337871548632026, 0.04897325160257173, 0.06786017368714284, 0.08822540499308393, 0.1092351905579167, 0.13002938729207184, 0.1497566783380388, 0.16760942608253726, 0.1828567369352988, 0.19487438419852832, 0.20317036398125563] + [0.2074050378964396] * 2 + [0.20317036398125563, 0.19487438419852832, 0.18285673693529883, 0.1676094260825373, 0.1497566783380388, 0.13002938729207195, 0.10923519055791678, 0.08822540499308393, 0.06786017368714284, 0.04897325160257173, 0.03233787154863204, 0.01863508793930623, 0.008425894345568822, 0.002128256352676751, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0013626632003288573, 0.0026695388091301235, 0.003867123185428111, 0.00490638708414699, 0.005744782919164887, 0.006347986666488841, 0.0066913030938299785, 0.0067606767863149855, 0.006553267576822464, 0.006077566822758586, 0.005353049768887191, 0.00440937822852091, 0.0032851862253990404, 0.0020264983122163438, 0.0006848453200280706, -0.0006848453200280659, -0.0020264983122163446, -0.0032851862253990387, -0.004409378228520908, -0.0053530497688871895, -0.006077566822758586, -0.006553267576822463, -0.0067606767863149855, -0.0066913030938299785, -0.006347986666488841, -0.005744782919164888, -0.0049063870841469905, -0.003867123185428112, -0.0026695388091301244, -0.0013626632003288645, -1.6580162893036912e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.001052422766398641, 0.004166604753883778, 0.009215050985986923, 0.015991077480798522, 0.024217272917471697, 0.0335568558882921, 0.043627462769079965, 0.054016801730889755, 0.06429953201592947, 0.07405467743816012, 0.0828828611978146, 0.09042265641450517, 0.09636538298617217, 0.10046774498873082] + [0.1025617912397893] * 2 + [0.10046774498873082, 0.09636538298617217, 0.09042265641450518, 0.08288286119781461, 0.0740546774381601, 0.06429953201592951, 0.0540168017308898, 0.043627462769079965, 0.0335568558882921, 0.024217272917471697, 0.01599107748079853, 0.009215050985986923, 0.004166604753883778, 0.0010524227663986523, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005379556773534572, 0.0010538873860689555, 0.0015266728213722807, 0.0019369560919921848, 0.0022679401526232517, 0.002506074476934269, 0.0026416098177083494, 0.0026689973421694757, 0.0025871158018483113, 0.002399317436626277, 0.0021132907336409356, 0.0017407456597192143, 0.001296934253959659, 0.0008000262074596908, 0.000270364994027236, -0.00027036499402723416, -0.0008000262074596914, -0.0012969342539596582, -0.0017407456597192139, -0.0021132907336409348, -0.0023993174366262774, -0.0025871158018483105, -0.0026689973421694757, -0.0026416098177083494, -0.002506074476934269, -0.0022679401526232517, -0.001936956091992185, -0.0015266728213722814, -0.0010538873860689558, -0.00053795567735346, -6.545559282441745e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.001052422766398641, -0.004166604753883778, -0.009215050985986923, -0.015991077480798522, -0.024217272917471697, -0.0335568558882921, -0.043627462769079965, -0.054016801730889755, -0.06429953201592947, -0.07405467743816012, -0.0828828611978146, -0.09042265641450517, -0.09636538298617217, -0.10046774498873082] + [-0.1025617912397893] * 2 + [-0.10046774498873082, -0.09636538298617217, -0.09042265641450518, -0.08288286119781461, -0.0740546774381601, -0.06429953201592951, -0.0540168017308898, -0.043627462769079965, -0.0335568558882921, -0.024217272917471697, -0.01599107748079853, -0.009215050985986923, -0.004166604753883778, -0.0010524227663986523, 8.01599822387153e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005379556773534571, -0.0010538873860689551, -0.0015266728213722796, -0.0019369560919921828, -0.0022679401526232486, -0.002506074476934265, -0.0026416098177083442, -0.002668997342169469, -0.0025871158018483035, -0.002399317436626268, -0.0021132907336409257, -0.0017407456597192032, -0.0012969342539596474, -0.0008000262074596786, -0.0002703649940272234, 0.00027036499402724673, 0.0008000262074597036, 0.00129693425395967, 0.001740745659719225, 0.0021132907336409447, 0.0023993174366262865, 0.0025871158018483183, 0.002668997342169482, 0.0026416098177083546, 0.002506074476934273, 0.0022679401526232547, 0.001936956091992187, 0.0015266728213722824, 0.0010538873860689562, 0.0005379556773534601, 6.545559282441745e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.001362663200328857, -0.002669538809130123, -0.0038671231854281096, -0.004906387084146988, -0.005744782919164884, -0.006347986666488836, -0.006691303093829973, -0.006760676786314979, -0.006553267576822456, -0.006077566822758576, -0.005353049768887181, -0.004409378228520899, -0.0032851862253990283, -0.002026498312216331, -0.0006848453200280579, 0.0006848453200280786, 0.002026498312216357, 0.003285186225399051, 0.0044093782285209195, 0.0053530497688872, 0.006077566822758595, 0.006553267576822471, 0.0067606767863149925, 0.006691303093829984, 0.006347986666488845, 0.00574478291916489, 0.004906387084146992, 0.0038671231854281135, 0.002669538809130125, 0.0013626632003288647, 1.6580162893036912e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.002128256352676728, 0.008425894345568822, 0.01863508793930623, 0.032337871548632026, 0.04897325160257173, 0.06786017368714284, 0.08822540499308393, 0.1092351905579167, 0.13002938729207184, 0.1497566783380388, 0.16760942608253726, 0.1828567369352988, 0.19487438419852832, 0.20317036398125563] + [0.2074050378964396] * 2 + [0.20317036398125563, 0.19487438419852832, 0.18285673693529883, 0.1676094260825373, 0.1497566783380388, 0.13002938729207195, 0.10923519055791678, 0.08822540499308393, 0.06786017368714284, 0.04897325160257173, 0.03233787154863204, 0.01863508793930623, 0.008425894345568822, 0.002128256352676751, -1.0152421708149686e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0005379556773534571, -0.0010538873860689553, -0.00152667282137228, -0.0019369560919921837, -0.0022679401526232504, -0.002506074476934267, -0.002641609817708347, -0.0026689973421694722, -0.0025871158018483074, -0.0023993174366262727, -0.0021132907336409304, -0.0017407456597192087, -0.0012969342539596532, -0.0008000262074596846, -0.0002703649940272297, 0.00027036499402724045, 0.0008000262074596975, 0.001296934253959664, 0.0017407456597192195, 0.00211329073364094, 0.0023993174366262818, 0.0025871158018483144, 0.002668997342169479, 0.002641609817708352, 0.0025060744769342712, 0.002267940152623253, 0.001936956091992186, 0.001526672821372282, 0.001053887386068956, 0.0005379556773534601, 6.545559282441745e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.001052422766398641, 0.004166604753883778, 0.009215050985986923, 0.015991077480798522, 0.024217272917471697, 0.0335568558882921, 0.043627462769079965, 0.054016801730889755, 0.06429953201592947, 0.07405467743816012, 0.0828828611978146, 0.09042265641450517, 0.09636538298617217, 0.10046774498873082] + [0.1025617912397893] * 2 + [0.10046774498873082, 0.09636538298617217, 0.09042265641450518, 0.08288286119781461, 0.0740546774381601, 0.06429953201592951, 0.0540168017308898, 0.043627462769079965, 0.0335568558882921, 0.024217272917471697, 0.01599107748079853, 0.009215050985986923, 0.004166604753883778, 0.0010524227663986523, -4.007999111935765e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0005379556773534573, 0.0010538873860689558, 0.0015266728213722814, 0.0019369560919921859, 0.002267940152623253, 0.0025060744769342712, 0.002641609817708352, 0.002668997342169479, 0.0025871158018483153, 0.0023993174366262813, 0.002113290733640941, 0.00174074565971922, 0.001296934253959665, 0.000800026207459697, 0.0002703649940272423, -0.00027036499402722787, -0.0008000262074596852, -0.0012969342539596524, -0.0017407456597192082, -0.0021132907336409296, -0.002399317436626273, -0.0025871158018483066, -0.0026689973421694722, -0.002641609817708347, -0.002506074476934267, -0.0022679401526232504, -0.001936956091992184, -0.0015266728213722807, -0.0010538873860689555, -0.0005379556773534599, -6.545559282441745e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.001052422766398641, -0.004166604753883778, -0.009215050985986923, -0.015991077480798522, -0.024217272917471697, -0.0335568558882921, -0.043627462769079965, -0.054016801730889755, -0.06429953201592947, -0.07405467743816012, -0.0828828611978146, -0.09042265641450517, -0.09636538298617217, -0.10046774498873082] + [-0.1025617912397893] * 2 + [-0.10046774498873082, -0.09636538298617217, -0.09042265641450518, -0.08288286119781461, -0.0740546774381601, -0.06429953201592951, -0.0540168017308898, -0.043627462769079965, -0.0335568558882921, -0.024217272917471697, -0.01599107748079853, -0.009215050985986923, -0.004166604753883778, -0.0010524227663986523, -4.007999111935765e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC3.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC3.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC3.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.047085973003043784,
        },
        "qubitC3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.001612649397959325, 0.0063845755360065854, 0.014120415197372612, 0.024503462197398504, 0.03710863337194701, 0.051419871532087345, 0.06685127290604431, 0.08277107410845543, 0.09852751660759673, 0.11347552979625009, 0.12700314025815493, 0.13855652603157423, 0.14766269014547898, 0.15394882517111705] + [0.15715757600194932] * 2 + [0.15394882517111705, 0.14766269014547898, 0.13855652603157426, 0.12700314025815493, 0.11347552979625007, 0.09852751660759682, 0.0827710741084555, 0.06685127290604431, 0.051419871532087345, 0.03710863337194701, 0.02450346219739851, 0.014120415197372612, 0.0063845755360065854, 0.0016126493979593424, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.000678852362866962, 0.0013299124302363849, 0.0019265257264538544, 0.0024442668330731418, 0.002861939368357256, 0.003162443769637227, 0.0033334773514168844, 0.0033680379787625757, 0.0032647107355300684, 0.003027725851197736, 0.0026667855147746874, 0.0021966666660427677, 0.0016366160259056837, 0.0010095621333738032, 0.0003411766484830867, -0.0003411766484830844, -0.0010095621333738036, -0.0016366160259056828, -0.002196666666042767, -0.0026667855147746865, -0.0030277258511977366, -0.003264710735530068, -0.0033680379787625757, -0.0033334773514168844, -0.003162443769637227, -0.0028619393683572566, -0.0024442668330731418, -0.0019265257264538549, -0.0013299124302363855, -0.0006788523628669656, -8.25991540238329e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC3.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC3.z.z90.wf": {
            "type": "constant",
            "sample": 0.018092149544688144,
        },
        "qubitC3.z.z180.wf": {
            "type": "constant",
            "sample": 0.025534508275873505,
        },
        "qubitC3.z.-z90.wf": {
            "type": "constant",
            "sample": 0.03139992138505721,
        },
        "qubitC3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01658888888888889,
        },
        "qubitC3.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC3.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC3.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0013318728729940627, 0.005272964460067455, 0.011661926008586794, 0.02023719268208458, 0.03064769205539679, 0.04246721706099824, 0.05521187495642443, 0.06835989795000286, 0.08137300443554933, 0.09371843630457051, 0.10489077012582194, 0.1144326092399682, 0.1219533096325576, 0.12714497294587274] + [0.129795051873872] * 2 + [0.12714497294587274, 0.1219533096325576, 0.11443260923996823, 0.10489077012582196, 0.0937184363045705, 0.08137300443554939, 0.06835989795000293, 0.05521187495642443, 0.04246721706099824, 0.03064769205539679, 0.020237192682084588, 0.011661926008586794, 0.005272964460067455, 0.0013318728729940772, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0017975143813591585, 0.0035214383127465014, 0.005101194145858569, 0.006472106491274323, 0.007578050036497085, 0.008373747322842583, 0.008826622409917372, 0.008918134538446046, 0.008644537191134423, 0.008017031475527766, 0.007061307549351801, 0.005816492862436962, 0.004333550274325061, 0.002673191628811757, 0.000903392203928223, -0.000903392203928217, -0.0026731916288117584, -0.0043335502743250594, -0.005816492862436959, -0.0070613075493517995, -0.008017031475527766, -0.008644537191134422, -0.008918134538446046, -0.008826622409917372, -0.008373747322842583, -0.007578050036497087, -0.006472106491274324, -0.005101194145858571, -0.0035214383127465027, -0.0017975143813591683, -2.187120136385779e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006609544945816668, 0.0026167584236596376, 0.005787340944580256, 0.0100428980364094, 0.015209206691800868, 0.02107475762738743, 0.0273994144986932, 0.0339242451102913, 0.04038212213030785, 0.046508659314775054, 0.052053035511523245, 0.056788263307621045, 0.06052048192073054, 0.06309689388231397] + [0.06441202058394856] * 2 + [0.06309689388231397, 0.06052048192073054, 0.05678826330762105, 0.05205303551152325, 0.04650865931477505, 0.040382122130307875, 0.03392424511029133, 0.0273994144986932, 0.02107475762738743, 0.015209206691800868, 0.010042898036409403, 0.005787340944580256, 0.0026167584236596376, 0.000660954494581674, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0008920334917278672, 0.0017475470274949108, 0.0025315157826282554, 0.0032118439841803243, 0.0037606789156444066, 0.004155551210440996, 0.0043802947504255965, 0.0044257085086649925, 0.004289933240560974, 0.003978527601542636, 0.003504240575059999, 0.002886489527703618, 0.002150565259934934, 0.0013265965977438251, 0.00044831691497260374, -0.00044831691497260065, -0.0013265965977438258, -0.002150565259934933, -0.0028864895277036177, -0.0035042405750599984, -0.003978527601542636, -0.004289933240560972, -0.0044257085086649925, -0.0043802947504255965, -0.004155551210440996, -0.0037606789156444066, -0.0032118439841803243, -0.002531515782628256, -0.0017475470274949112, -0.0008920334917278719, -1.0853790280183091e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0006609544945816669, -0.0026167584236596376, -0.005787340944580256, -0.0100428980364094, -0.015209206691800868, -0.02107475762738743, -0.0273994144986932, -0.0339242451102913, -0.04038212213030785, -0.046508659314775054, -0.052053035511523245, -0.056788263307621045, -0.06052048192073054, -0.06309689388231397] + [-0.06441202058394856] * 2 + [-0.06309689388231397, -0.06052048192073054, -0.05678826330762105, -0.05205303551152325, -0.04650865931477505, -0.040382122130307875, -0.03392424511029133, -0.0273994144986932, -0.02107475762738743, -0.015209206691800868, -0.010042898036409403, -0.005787340944580256, -0.0026167584236596376, -0.0006609544945816738, 1.3292059525242876e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008920334917278671, -0.0017475470274949105, -0.0025315157826282545, -0.003211843984180323, -0.003760678915644405, -0.004155551210440993, -0.004380294750425593, -0.004425708508664988, -0.004289933240560969, -0.00397852760154263, -0.0035042405750599923, -0.002886489527703611, -0.0021505652599349266, -0.0013265965977438173, -0.0004483169149725958, 0.00044831691497260857, 0.0013265965977438336, 0.0021505652599349404, 0.0028864895277036246, 0.003504240575060005, 0.003978527601542642, 0.0042899332405609775, 0.004425708508664997, 0.0043802947504256, 0.004155551210440998, 0.0037606789156444083, 0.0032118439841803256, 0.0025315157826282567, 0.0017475470274949114, 0.000892033491727872, 1.0853790280183091e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0017975143813591585, -0.003521438312746501, -0.005101194145858568, -0.0064721064912743225, -0.007578050036497083, -0.008373747322842582, -0.008826622409917368, -0.008918134538446042, -0.008644537191134418, -0.00801703147552776, -0.007061307549351795, -0.005816492862436955, -0.004333550274325053, -0.0026731916288117493, -0.0009033922039282151, 0.0009033922039282249, 0.0026731916288117662, 0.004333550274325067, 0.005816492862436966, 0.0070613075493518055, 0.00801703147552777, 0.008644537191134427, 0.00891813453844605, 0.008826622409917375, 0.008373747322842585, 0.007578050036497089, 0.006472106491274325, 0.005101194145858572, 0.003521438312746503, 0.0017975143813591683, 2.187120136385779e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.001331872872994063, 0.005272964460067455, 0.011661926008586794, 0.02023719268208458, 0.03064769205539679, 0.04246721706099824, 0.05521187495642443, 0.06835989795000286, 0.08137300443554933, 0.09371843630457051, 0.10489077012582194, 0.1144326092399682, 0.1219533096325576, 0.12714497294587274] + [0.129795051873872] * 2 + [0.12714497294587274, 0.1219533096325576, 0.11443260923996823, 0.10489077012582196, 0.0937184363045705, 0.08137300443554939, 0.06835989795000293, 0.05521187495642443, 0.04246721706099824, 0.03064769205539679, 0.020237192682084588, 0.011661926008586794, 0.005272964460067455, 0.001331872872994077, -1.3392248371877835e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008920334917278672, -0.0017475470274949105, -0.002531515782628255, -0.003211843984180324, -0.0037606789156444057, -0.004155551210440995, -0.004380294750425595, -0.004425708508664991, -0.004289933240560971, -0.003978527601542633, -0.0035042405750599958, -0.0028864895277036146, -0.00215056525993493, -0.0013265965977438212, -0.0004483169149725998, 0.0004483169149726046, 0.0013265965977438297, 0.002150565259934937, 0.002886489527703621, 0.0035042405750600014, 0.0039785276015426385, 0.004289933240560975, 0.004425708508664994, 0.004380294750425598, 0.004155551210440997, 0.0037606789156444075, 0.0032118439841803247, 0.0025315157826282563, 0.0017475470274949114, 0.0008920334917278719, 1.0853790280183091e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0006609544945816669, 0.0026167584236596376, 0.005787340944580256, 0.0100428980364094, 0.015209206691800868, 0.02107475762738743, 0.0273994144986932, 0.0339242451102913, 0.04038212213030785, 0.046508659314775054, 0.052053035511523245, 0.056788263307621045, 0.06052048192073054, 0.06309689388231397] + [0.06441202058394856] * 2 + [0.06309689388231397, 0.06052048192073054, 0.05678826330762105, 0.05205303551152325, 0.04650865931477505, 0.040382122130307875, 0.03392424511029133, 0.0273994144986932, 0.02107475762738743, 0.015209206691800868, 0.010042898036409403, 0.005787340944580256, 0.0026167584236596376, 0.0006609544945816738, -6.646029762621438e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0008920334917278672, 0.001747547027494911, 0.002531515782628256, 0.0032118439841803247, 0.0037606789156444075, 0.004155551210440997, 0.004380294750425598, 0.004425708508664994, 0.004289933240560977, 0.0039785276015426385, 0.003504240575060002, 0.0028864895277036216, 0.002150565259934938, 0.001326596597743829, 0.0004483169149726077, -0.0004483169149725967, -0.0013265965977438219, -0.002150565259934929, -0.002886489527703614, -0.0035042405750599953, -0.003978527601542633, -0.00428993324056097, -0.004425708508664991, -0.004380294750425595, -0.004155551210440995, -0.0037606789156444057, -0.003211843984180324, -0.0025315157826282554, -0.001747547027494911, -0.0008920334917278719, -1.0853790280183091e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0006609544945816667, -0.0026167584236596376, -0.005787340944580256, -0.0100428980364094, -0.015209206691800868, -0.02107475762738743, -0.0273994144986932, -0.0339242451102913, -0.04038212213030785, -0.046508659314775054, -0.052053035511523245, -0.056788263307621045, -0.06052048192073054, -0.06309689388231397] + [-0.06441202058394856] * 2 + [-0.06309689388231397, -0.06052048192073054, -0.05678826330762105, -0.05205303551152325, -0.04650865931477505, -0.040382122130307875, -0.03392424511029133, -0.0273994144986932, -0.02107475762738743, -0.015209206691800868, -0.010042898036409403, -0.005787340944580256, -0.0026167584236596376, -0.0006609544945816741, -6.646029762621438e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC4.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC4.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "qubitC4.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "qubitC4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "qubitC4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "qubitC4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "qubitC4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "qubitC4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "qubitC4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.02836710646226952,
        },
        "qubitC4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0010091967820947487, 0.0039954704935444485, 0.008836562737730086, 0.015334278629370507, 0.023222600916412775, 0.03217858075758684, 0.041835559285912094, 0.05179817866579211, 0.0616585680892565, 0.07101304205476212, 0.0794786272990571, 0.086708741767576, 0.09240738372447525, 0.09634125009847742] + [0.09834928793802118] * 2 + [0.09634125009847742, 0.09240738372447525, 0.08670874176757601, 0.0794786272990571, 0.07101304205476212, 0.06165856808925655, 0.05179817866579215, 0.041835559285912094, 0.03217858075758684, 0.023222600916412775, 0.015334278629370514, 0.008836562737730086, 0.0039954704935444485, 0.0010091967820947598, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.000900158013173167, 0.0017634634515229489, 0.002554572488969224, 0.0032410970285675755, 0.00379493067500364, 0.004193399413719634, 0.004420189887734435, 0.00446601726839513, 0.00432900537735298, 0.004014763497525464, 0.0035361567283945755, 0.0029127792873224374, 0.002170152319990285, 0.0013386790616956603, 0.00045240012532710546, -0.00045240012532710237, -0.001338679061695661, -0.0021701523199902837, -0.0029127792873224366, -0.0035361567283945746, -0.004014763497525464, -0.004329005377352979, -0.00446601726839513, -0.004420189887734435, -0.004193399413719634, -0.00379493067500364, -0.003241097028567576, -0.0025545724889692242, -0.0017634634515229497, -0.0009001580131731717, -1.0952645146857798e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC4.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC4.z.z0.wf": {
            "type": "constant",
            "sample": -0.0,
        },
        "qubitC4.z.z90.wf": {
            "type": "constant",
            "sample": 0.018890467518712446,
        },
        "qubitC4.z.z180.wf": {
            "type": "constant",
            "sample": 0.02634963499864954,
        },
        "qubitC4.z.-z90.wf": {
            "type": "constant",
            "sample": 0.03253175743797823,
        },
        "qubitC4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.014933333333333335,
        },
        "qubitC4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC5.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitC5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitC5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitC5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB2.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitB2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitB2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitB2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB4.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitB4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitB4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitB4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB5.z.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "qubitB5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubitB5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitB5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "qubitB5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.2033748528790328] * 50 + [0.0] * 2,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.z.Cz.CZ_snz_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.20232766962830684] * 26 + [0.08760032829676206] + [0.0] * 2 + [-0.08760032829676206] + [-0.20232766962830684] * 26 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.z.Cz_unipolar.flux_pulse_control_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.210600443040884] * 51 + [0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC2.z.Cz_SNZ.CZ_snz_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.21143597111202225] * 26 + [0.1912462623254267] + [0.0] * 2 + [-0.1912462623254267] + [-0.21143597111202225] * 26 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.13426639835122103] * 42 + [0.0] * 2,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.z.Cz.CZ_snz_qubitC4.wf": {
            "type": "arbitrary",
            "samples": [0.13345224540011472] * 22 + [0.03987712031031265] + [0.0] * 2 + [-0.03987712031031265] + [-0.13345224540011472] * 22 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.z.Cz_unipolar.flux_pulse_control_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.1405818855635322] * 48,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubitC3.z.Cz.CZ_snz_qubitC1.wf": {
            "type": "arbitrary",
            "samples": [0.14128141093535543] * 24 + [0.04242761306307403] + [0.0] * 2 + [-0.04242761306307403] + [-0.14128141093535543] * 24 + [0.0] * 4,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "qubitC1.resonator.readout.iw1": {
            "cosine": [(0.9346129235434444, 1500)],
            "sine": [(-0.35566653363168105, 1500)],
        },
        "qubitC1.resonator.readout.iw2": {
            "cosine": [(0.35566653363168105, 1500)],
            "sine": [(0.9346129235434444, 1500)],
        },
        "qubitC1.resonator.readout.iw3": {
            "cosine": [(-0.35566653363168105, 1500)],
            "sine": [(-0.9346129235434444, 1500)],
        },
        "qubitC2.resonator.readout.iw1": {
            "cosine": [(-0.1561117302631273, 1500)],
            "sine": [(0.9877394027142243, 1500)],
        },
        "qubitC2.resonator.readout.iw2": {
            "cosine": [(-0.9877394027142243, 1500)],
            "sine": [(-0.1561117302631273, 1500)],
        },
        "qubitC2.resonator.readout.iw3": {
            "cosine": [(0.9877394027142243, 1500)],
            "sine": [(0.1561117302631273, 1500)],
        },
        "qubitC3.resonator.readout.iw1": {
            "cosine": [(0.3164120553070932, 1500)],
            "sine": [(0.9486218483971055, 1500)],
        },
        "qubitC3.resonator.readout.iw2": {
            "cosine": [(-0.9486218483971055, 1500)],
            "sine": [(0.3164120553070932, 1500)],
        },
        "qubitC3.resonator.readout.iw3": {
            "cosine": [(0.9486218483971055, 1500)],
            "sine": [(-0.3164120553070932, 1500)],
        },
        "qubitC4.resonator.readout.iw1": {
            "cosine": [(0.025991300810657807, 1500)],
            "sine": [(0.9996621690762184, 1500)],
        },
        "qubitC4.resonator.readout.iw2": {
            "cosine": [(-0.9996621690762184, 1500)],
            "sine": [(0.025991300810657807, 1500)],
        },
        "qubitC4.resonator.readout.iw3": {
            "cosine": [(0.9996621690762184, 1500)],
            "sine": [(-0.025991300810657807, 1500)],
        },
        "qubitC5.resonator.readout.iw1": {
            "cosine": [(1.0, 1500)],
            "sine": [(-0.0, 1500)],
        },
        "qubitC5.resonator.readout.iw2": {
            "cosine": [(0.0, 1500)],
            "sine": [(1.0, 1500)],
        },
        "qubitC5.resonator.readout.iw3": {
            "cosine": [(-0.0, 1500)],
            "sine": [(-1.0, 1500)],
        },
        "qubitB2.resonator.readout.iw1": {
            "cosine": [(1.0, 1500)],
            "sine": [(-0.0, 1500)],
        },
        "qubitB2.resonator.readout.iw2": {
            "cosine": [(0.0, 1500)],
            "sine": [(1.0, 1500)],
        },
        "qubitB2.resonator.readout.iw3": {
            "cosine": [(-0.0, 1500)],
            "sine": [(-1.0, 1500)],
        },
        "qubitB4.resonator.readout.iw1": {
            "cosine": [(1.0, 1500)],
            "sine": [(-0.0, 1500)],
        },
        "qubitB4.resonator.readout.iw2": {
            "cosine": [(0.0, 1500)],
            "sine": [(1.0, 1500)],
        },
        "qubitB4.resonator.readout.iw3": {
            "cosine": [(-0.0, 1500)],
            "sine": [(-1.0, 1500)],
        },
        "qubitB5.resonator.readout.iw1": {
            "cosine": [(1.0, 1500)],
            "sine": [(-0.0, 1500)],
        },
        "qubitB5.resonator.readout.iw2": {
            "cosine": [(0.0, 1500)],
            "sine": [(1.0, 1500)],
        },
        "qubitB5.resonator.readout.iw3": {
            "cosine": [(-0.0, 1500)],
            "sine": [(-1.0, 1500)],
        },
    },
    "mixers": {
        "qubitC1.xy_mixer_5f3": [{'intermediate_frequency': -161579893.20423436, 'lo_frequency': 5100000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC1.resonator_mixer_256": [{'intermediate_frequency': -193703946.0, 'lo_frequency': 7550000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC2.xy_mixer_ddb": [{'intermediate_frequency': -102724629.50200598, 'lo_frequency': 5850000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC2.resonator_mixer_66f": [{'intermediate_frequency': 66883846.0, 'lo_frequency': 7550000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC3.xy_mixer_fa2": [{'intermediate_frequency': -235169505.05035493, 'lo_frequency': 5850000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC3.resonator_mixer_293": [{'intermediate_frequency': 173982913.0, 'lo_frequency': 7300000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC4.xy_mixer_868": [{'intermediate_frequency': -115634512.22316144, 'lo_frequency': 5100000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC4.resonator_mixer_b16": [{'intermediate_frequency': -85478390.0, 'lo_frequency': 7300000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitC5.resonator_mixer_11c": [{'intermediate_frequency': 315375522.0, 'lo_frequency': 7300000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitB2.resonator_mixer_aab": [{'intermediate_frequency': -176200000.0, 'lo_frequency': 7550000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitB4.resonator_mixer_bc5": [{'intermediate_frequency': 137955078.0, 'lo_frequency': 7550000000.0, 'correction': (1, 0, 0, 1)}],
        "qubitB5.resonator_mixer_6ab": [{'intermediate_frequency': 64300000.0, 'lo_frequency': 7550000000.0, 'correction': (1, 0, 0, 1)}],
    },
}


