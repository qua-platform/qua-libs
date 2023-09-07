
# Single QUA script generated at 2023-09-06 16:57:18.504750
# QUA library version: 1.1.3

from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(int, )
    v3 = declare(int, )
    v4 = declare(fixed, )
    v5 = declare(fixed, )
    v6 = declare(fixed, )
    v7 = declare(fixed, )
    v8 = declare(fixed, )
    v9 = declare(fixed, )
    v10 = declare(fixed, )
    v11 = declare(fixed, )
    v12 = declare(bool, )
    v13 = declare(bool, )
    v14 = declare(int, )
    v15 = declare(int, )
    v16 = declare(int, )
    v17 = declare(int, )
    v18 = declare(fixed, )
    v19 = declare(fixed, )
    v20 = declare(fixed, )
    v21 = declare(fixed, )
    assign(v15, 0)
    with for_(v1,0,(v1<10000),(v1+1)):
        r1 = declare_stream()
        save(v15, r1)
        assign(v15, (v15+1))
        assign(v16, 0)
        with for_(v2,0,(v2<3),(v2+1)):
            assign(v16, (v16+1))
            assign(v17, 0)
            with for_(v3,0,(v3<3),(v3+1)):
                assign(v17, (v17+1))
                with if_((v2==0), unsafe=True):
                    assign(v4, 0.0)
                    assign(v6, 0.0)
                    assign(v8, 0.0)
                    assign(v10, 0.0)
                with elif_((v2==1)):
                    assign(v4, 0.0)
                    assign(v6, -0.5)
                    assign(v8, 0.5)
                    assign(v10, 0.0)
                with elif_((v2==2)):
                    assign(v4, 0.5)
                    assign(v6, 0.0)
                    assign(v8, 0.0)
                    assign(v10, 0.5)
                with if_((v3==0), unsafe=True):
                    assign(v5, 0.0)
                    assign(v7, 0.0)
                    assign(v9, 0.0)
                    assign(v11, 0.0)
                with elif_((v3==1)):
                    assign(v5, 0.0)
                    assign(v7, -0.5)
                    assign(v9, 0.5)
                    assign(v11, 0.0)
                with elif_((v3==2)):
                    assign(v5, 0.5)
                    assign(v7, 0.0)
                    assign(v9, 0.0)
                    assign(v11, 0.5)
                play("x$drag"*amp(v4, v6, v8, v10), "qubit0_xy")
                wait(20, "qubit0_xy")
                play("x$drag"*amp(v5, v7, v9, v11), "qubit1_xy")
                wait(20, "qubit1_xy")
                align("qubit0_xy", "qubit0_rr", "qubit0_z")
                measure("readout$rect$rotation", "qubit0_rr", None, dual_demod.full("w1", "out1", "w2", "out2", v18), dual_demod.full("w3", "out1", "w1", "out2", v19))
                align("qubit0_xy", "qubit0_rr", "qubit0_z")
                assign(v12, (v18>0.0))
                align("qubit1_xy", "qubit1_rr", "qubit1_z")
                measure("readout$rect$rotation", "qubit1_rr", None, dual_demod.full("w1", "out1", "w2", "out2", v20), dual_demod.full("w3", "out1", "w1", "out2", v21))
                align("qubit1_xy", "qubit1_rr", "qubit1_z")
                assign(v13, (v20>0.0))
                assign(v14, (Cast.unsafe_cast_int(v12)+(Cast.unsafe_cast_int(v13)<<1)))
                r3 = declare_stream()
                save(v14, r3)
                wait(100, )
    save(v15, r1)
    with stream_processing():
        r3.buffer(10000, 3, 3).save("statequbit0_qubit1")
        r1.save("qubit0_qubit1")


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "2": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "3": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "4": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "5": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "6": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "7": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "8": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "9": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "10": {
                    "offset": 0.0,
                    "delay": 0,
                },
            },
            "analog_inputs": {},
            "digital_outputs": {
                "1": {},
                "2": {},
                "3": {},
                "4": {},
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                "3": {
                    "offset": 0.018766451590108237,
                    "delay": 0,
                    "crosstalk": {},
                },
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "2": {
                    "offset": 0.0,
                    "delay": 0,
                },
                "4": {
                    "offset": -0.03823046968983945,
                    "delay": 0,
                    "crosstalk": {},
                },
                "5": {
                    "offset": 0.03181726374202038,
                    "delay": 0,
                    "crosstalk": {},
                },
                "6": {
                    "offset": 0.009284224666683593,
                    "delay": 0,
                    "crosstalk": {},
                },
                "7": {
                    "offset": 0.005801026962879232,
                    "delay": 0,
                    "crosstalk": {},
                },
            },
            "analog_inputs": {
                "1": {
                    "offset": 0.013969112792968748,
                    "gain_db": 0,
                },
                "2": {
                    "offset": 0.01661838671875,
                    "gain_db": 0,
                },
            },
            "digital_outputs": {
                "5": {},
            },
        },
    },
    "elements": {
        "qubit0_xy": {
            "mixInputs": {
                "I": [con1, 1],
                "Q": [con1, 2],
                "lo_frequency": 4945000000.0,
                "mixer": "octave_Octave_2",
            },
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit0_xy$cw$rect",
                "x$drag": "qubit0_xy$x$drag",
                "x_12$drag": "qubit0_xy$x_12$drag",
                "id$drag": "qubit0_xy$id$drag",
                "y$drag_rot": "qubit0_xy$y$drag_rot",
                "sy$drag_rot": "qubit0_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit0_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit0_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit0_xy$-sx$drag_rot",
            },
        },
        "qubit0_z": {
            "singleInput": {
                "port": [con2, 3],
            },
            "intermediate_frequency": 0,
        },
        "qubit0_rr": {
            "mixInputs": {
                "I": [con2, 1],
                "Q": [con2, 2],
                "lo_frequency": 7350000000.0,
                "mixer": "octave_Octave_1",
            },
            "intermediate_frequency": -104000000.0,
            "time_of_flight": 284,
            "smearing": 0,
            "outputs": {
                "out1": [con2, 1],
                "out2": [con2, 2],
            },
            "operations": {
                "readout$rect$rotation": "qubit0_rr$readout$rect$rotation",
            },
        },
        "qubit0_blanker": {
            "digitalInputs": {
                "qubit0_blanker$input": {
                    "buffer": 0,
                    "delay": 0,
                    "port": [con1, 1],
                },
            },
            "operations": {},
        },
        "qubit1_xy": {
            "mixInputs": {
                "I": [con1, 3],
                "Q": [con1, 4],
                "lo_frequency": 5049000000.0,
                "mixer": "octave_Octave_3",
            },
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit1_xy$cw$rect",
                "x$drag": "qubit1_xy$x$drag",
                "x_12$drag": "qubit1_xy$x_12$drag",
                "id$drag": "qubit1_xy$id$drag",
                "y$drag_rot": "qubit1_xy$y$drag_rot",
                "sy$drag_rot": "qubit1_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit1_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit1_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit1_xy$-sx$drag_rot",
            },
        },
        "qubit1_z": {
            "singleInput": {
                "port": [con2, 4],
            },
            "intermediate_frequency": 0,
        },
        "qubit1_rr": {
            "mixInputs": {
                "I": [con2, 1],
                "Q": [con2, 2],
                "lo_frequency": 7350000000.0,
                "mixer": "octave_Octave_1",
            },
            "intermediate_frequency": -35000000.0,
            "time_of_flight": 284,
            "smearing": 0,
            "outputs": {
                "out1": [con2, 1],
                "out2": [con2, 2],
            },
            "operations": {
                "readout$rect$rotation": "qubit1_rr$readout$rect$rotation",
            },
        },
        "qubit1_blanker": {
            "digitalInputs": {
                "qubit1_blanker$input": {
                    "buffer": 0,
                    "delay": 0,
                    "port": [con1, 2],
                },
            },
            "operations": {},
        },
        "qubit2_xy": {
            "mixInputs": {
                "I": [con1, 5],
                "Q": [con1, 6],
                "lo_frequency": 5575000000.0,
                "mixer": "octave_Octave_2",
            },
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit2_xy$cw$rect",
                "x$drag": "qubit2_xy$x$drag",
                "x_12$drag": "qubit2_xy$x_12$drag",
                "id$drag": "qubit2_xy$id$drag",
                "y$drag_rot": "qubit2_xy$y$drag_rot",
                "sy$drag_rot": "qubit2_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit2_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit2_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit2_xy$-sx$drag_rot",
            },
        },
        "qubit2_z": {
            "singleInput": {
                "port": [con2, 5],
            },
            "intermediate_frequency": 0,
        },
        "qubit2_rr": {
            "mixInputs": {
                "I": [con2, 1],
                "Q": [con2, 2],
                "lo_frequency": 7350000000.0,
                "mixer": "octave_Octave_1",
            },
            "intermediate_frequency": 118000000.0,
            "time_of_flight": 284,
            "smearing": 0,
            "outputs": {
                "out1": [con2, 1],
                "out2": [con2, 2],
            },
            "operations": {
                "readout$rect$rotation": "qubit2_rr$readout$rect$rotation",
            },
        },
        "qubit2_blanker": {
            "digitalInputs": {
                "qubit2_blanker$input": {
                    "buffer": 0,
                    "delay": 0,
                    "port": [con1, 3],
                },
            },
            "operations": {},
        },
        "qubit3_xy": {
            "mixInputs": {
                "I": [con1, 7],
                "Q": [con1, 8],
                "lo_frequency": 6675000000.0,
                "mixer": "octave_Octave_4",
            },
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit3_xy$cw$rect",
                "x$drag": "qubit3_xy$x$drag",
                "x_12$drag": "qubit3_xy$x_12$drag",
                "id$drag": "qubit3_xy$id$drag",
                "y$drag_rot": "qubit3_xy$y$drag_rot",
                "sy$drag_rot": "qubit3_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit3_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit3_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit3_xy$-sx$drag_rot",
            },
        },
        "qubit3_z": {
            "singleInput": {
                "port": [con2, 6],
            },
            "intermediate_frequency": 0,
        },
        "qubit3_rr": {
            "mixInputs": {
                "I": [con2, 1],
                "Q": [con2, 2],
                "lo_frequency": 7350000000.0,
                "mixer": "octave_Octave_1",
            },
            "intermediate_frequency": 143000000.0,
            "time_of_flight": 284,
            "smearing": 0,
            "outputs": {
                "out1": [con2, 1],
                "out2": [con2, 2],
            },
            "operations": {
                "readout$rect$rotation": "qubit3_rr$readout$rect$rotation",
            },
        },
        "qubit3_blanker": {
            "digitalInputs": {
                "qubit3_blanker$input": {
                    "buffer": 0,
                    "delay": 0,
                    "port": [con1, 4],
                },
            },
            "operations": {},
        },
        "qubit4_xy": {
            "mixInputs": {
                "I": [con1, 9],
                "Q": [con1, 10],
                "lo_frequency": 6536000000.0,
                "mixer": "octave_Octave_5",
            },
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit4_xy$cw$rect",
                "x$drag": "qubit4_xy$x$drag",
                "x_12$drag": "qubit4_xy$x_12$drag",
                "id$drag": "qubit4_xy$id$drag",
                "y$drag_rot": "qubit4_xy$y$drag_rot",
                "sy$drag_rot": "qubit4_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit4_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit4_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit4_xy$-sx$drag_rot",
            },
        },
        "qubit4_z": {
            "singleInput": {
                "port": [con2, 7],
            },
            "intermediate_frequency": 0,
        },
        "qubit4_rr": {
            "mixInputs": {
                "I": [con2, 1],
                "Q": [con2, 2],
                "lo_frequency": 7350000000.0,
                "mixer": "octave_Octave_1",
            },
            "intermediate_frequency": 245000000.0,
            "time_of_flight": 284,
            "smearing": 0,
            "outputs": {
                "out1": [con2, 1],
                "out2": [con2, 2],
            },
            "operations": {
                "readout$rect$rotation": "qubit4_rr$readout$rect$rotation",
            },
        },
        "qubit4_blanker": {
            "digitalInputs": {
                "qubit4_blanker$input": {
                    "buffer": 0,
                    "delay": 0,
                    "port": [con2, 5],
                },
            },
            "operations": {},
        },
    },
    "pulses": {
        "qubit0_xy$cw$rect": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "qubit0_xy$cw$rect$i",
                "Q": "qubit0_xy$cw$rect$q",
            },
        },
        "qubit0_xy$x$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$x$drag$i",
                "Q": "qubit0_xy$x$drag$q",
            },
        },
        "qubit0_xy$x_12$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$x_12$drag$i",
                "Q": "qubit0_xy$x_12$drag$q",
            },
        },
        "qubit0_xy$id$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$id$drag$i",
                "Q": "qubit0_xy$id$drag$q",
            },
        },
        "qubit0_xy$y$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$y$drag_rot$i",
                "Q": "qubit0_xy$y$drag_rot$q",
            },
        },
        "qubit0_xy$sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$sy$drag_rot$i",
                "Q": "qubit0_xy$sy$drag_rot$q",
            },
        },
        "qubit0_xy$-sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$-sy$drag_rot$i",
                "Q": "qubit0_xy$-sy$drag_rot$q",
            },
        },
        "qubit0_xy$sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$sx$drag_rot$i",
                "Q": "qubit0_xy$sx$drag_rot$q",
            },
        },
        "qubit0_xy$-sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$-sx$drag_rot$i",
                "Q": "qubit0_xy$-sx$drag_rot$q",
            },
        },
        "qubit0_rr$readout$rect$rotation": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "qubit0_rr$readout$rect$rotation$i",
                "Q": "qubit0_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit0_rr$readout$rect$rotation$w1",
                "w2": "qubit0_rr$readout$rect$rotation$w2",
                "w3": "qubit0_rr$readout$rect$rotation$w3",
            },
        },
        "qubit1_xy$cw$rect": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "qubit1_xy$cw$rect$i",
                "Q": "qubit1_xy$cw$rect$q",
            },
        },
        "qubit1_xy$x$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$x$drag$i",
                "Q": "qubit1_xy$x$drag$q",
            },
        },
        "qubit1_xy$x_12$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$x_12$drag$i",
                "Q": "qubit1_xy$x_12$drag$q",
            },
        },
        "qubit1_xy$id$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$id$drag$i",
                "Q": "qubit1_xy$id$drag$q",
            },
        },
        "qubit1_xy$y$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$y$drag_rot$i",
                "Q": "qubit1_xy$y$drag_rot$q",
            },
        },
        "qubit1_xy$sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$sy$drag_rot$i",
                "Q": "qubit1_xy$sy$drag_rot$q",
            },
        },
        "qubit1_xy$-sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$-sy$drag_rot$i",
                "Q": "qubit1_xy$-sy$drag_rot$q",
            },
        },
        "qubit1_xy$sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$sx$drag_rot$i",
                "Q": "qubit1_xy$sx$drag_rot$q",
            },
        },
        "qubit1_xy$-sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$-sx$drag_rot$i",
                "Q": "qubit1_xy$-sx$drag_rot$q",
            },
        },
        "qubit1_rr$readout$rect$rotation": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "qubit1_rr$readout$rect$rotation$i",
                "Q": "qubit1_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit1_rr$readout$rect$rotation$w1",
                "w2": "qubit1_rr$readout$rect$rotation$w2",
                "w3": "qubit1_rr$readout$rect$rotation$w3",
            },
        },
        "qubit2_xy$cw$rect": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "qubit2_xy$cw$rect$i",
                "Q": "qubit2_xy$cw$rect$q",
            },
        },
        "qubit2_xy$x$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$x$drag$i",
                "Q": "qubit2_xy$x$drag$q",
            },
        },
        "qubit2_xy$x_12$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$x_12$drag$i",
                "Q": "qubit2_xy$x_12$drag$q",
            },
        },
        "qubit2_xy$id$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$id$drag$i",
                "Q": "qubit2_xy$id$drag$q",
            },
        },
        "qubit2_xy$y$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$y$drag_rot$i",
                "Q": "qubit2_xy$y$drag_rot$q",
            },
        },
        "qubit2_xy$sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$sy$drag_rot$i",
                "Q": "qubit2_xy$sy$drag_rot$q",
            },
        },
        "qubit2_xy$-sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$-sy$drag_rot$i",
                "Q": "qubit2_xy$-sy$drag_rot$q",
            },
        },
        "qubit2_xy$sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$sx$drag_rot$i",
                "Q": "qubit2_xy$sx$drag_rot$q",
            },
        },
        "qubit2_xy$-sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$-sx$drag_rot$i",
                "Q": "qubit2_xy$-sx$drag_rot$q",
            },
        },
        "qubit2_rr$readout$rect$rotation": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "qubit2_rr$readout$rect$rotation$i",
                "Q": "qubit2_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit2_rr$readout$rect$rotation$w1",
                "w2": "qubit2_rr$readout$rect$rotation$w2",
                "w3": "qubit2_rr$readout$rect$rotation$w3",
            },
        },
        "qubit3_xy$cw$rect": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "qubit3_xy$cw$rect$i",
                "Q": "qubit3_xy$cw$rect$q",
            },
        },
        "qubit3_xy$x$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$x$drag$i",
                "Q": "qubit3_xy$x$drag$q",
            },
        },
        "qubit3_xy$x_12$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$x_12$drag$i",
                "Q": "qubit3_xy$x_12$drag$q",
            },
        },
        "qubit3_xy$id$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$id$drag$i",
                "Q": "qubit3_xy$id$drag$q",
            },
        },
        "qubit3_xy$y$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$y$drag_rot$i",
                "Q": "qubit3_xy$y$drag_rot$q",
            },
        },
        "qubit3_xy$sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$sy$drag_rot$i",
                "Q": "qubit3_xy$sy$drag_rot$q",
            },
        },
        "qubit3_xy$-sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$-sy$drag_rot$i",
                "Q": "qubit3_xy$-sy$drag_rot$q",
            },
        },
        "qubit3_xy$sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$sx$drag_rot$i",
                "Q": "qubit3_xy$sx$drag_rot$q",
            },
        },
        "qubit3_xy$-sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$-sx$drag_rot$i",
                "Q": "qubit3_xy$-sx$drag_rot$q",
            },
        },
        "qubit3_rr$readout$rect$rotation": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "qubit3_rr$readout$rect$rotation$i",
                "Q": "qubit3_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit3_rr$readout$rect$rotation$w1",
                "w2": "qubit3_rr$readout$rect$rotation$w2",
                "w3": "qubit3_rr$readout$rect$rotation$w3",
            },
        },
        "qubit4_xy$cw$rect": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "qubit4_xy$cw$rect$i",
                "Q": "qubit4_xy$cw$rect$q",
            },
        },
        "qubit4_xy$x$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$x$drag$i",
                "Q": "qubit4_xy$x$drag$q",
            },
        },
        "qubit4_xy$x_12$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$x_12$drag$i",
                "Q": "qubit4_xy$x_12$drag$q",
            },
        },
        "qubit4_xy$id$drag": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$id$drag$i",
                "Q": "qubit4_xy$id$drag$q",
            },
        },
        "qubit4_xy$y$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$y$drag_rot$i",
                "Q": "qubit4_xy$y$drag_rot$q",
            },
        },
        "qubit4_xy$sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$sy$drag_rot$i",
                "Q": "qubit4_xy$sy$drag_rot$q",
            },
        },
        "qubit4_xy$-sy$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$-sy$drag_rot$i",
                "Q": "qubit4_xy$-sy$drag_rot$q",
            },
        },
        "qubit4_xy$sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$sx$drag_rot$i",
                "Q": "qubit4_xy$sx$drag_rot$q",
            },
        },
        "qubit4_xy$-sx$drag_rot": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$-sx$drag_rot$i",
                "Q": "qubit4_xy$-sx$drag_rot$q",
            },
        },
        "qubit4_rr$readout$rect$rotation": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "qubit4_rr$readout$rect$rotation$i",
                "Q": "qubit4_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit4_rr$readout$rect$rotation$w1",
                "w2": "qubit4_rr$readout$rect$rotation$w2",
                "w3": "qubit4_rr$readout$rect$rotation$w3",
            },
        },
    },
    "waveforms": {
        "qubit0_xy$cw$rect$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit0_xy$cw$rect$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit0_xy$x$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit0_xy$x$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 1.961813737422993e-05, 3.9112242567516353e-05, 5.83590675702984e-05, 7.723692775799656e-05, 9.562647122038802e-05, 0.0001134114333428111, 0.00013047937186849273, 0.00014672237779469904, 0.000162037757608194, 0.0001763286825466946, 0.00018950480078148177, 0.00020148280865075246, 0.00021218697733219069, 0.00022154963162496455, 0.0002295115778141358, 0.0002360224779123837, 0.00024104116791296874, 0.00024453591804183655, 0.0002464846333634655, 0.00024687499347216343, 0.0002457045303856384, 0.00024298064414837836, 0.0002387205560461864, 0.0002329511997276678, 0.0002257090509210314, 0.00021703989682279182, 0.000206998546616371, 0.00019564848495079342, 0.00018306147057029523, 0.00016931708263243875, 0.00015450221758305573, 0.00013871053976893965, 0.00012204188926168753, 0.00010460165063662163, 8.650008669757402e-05, 6.78516413599388e-05, 4.877421609939171e-05, 2.9388424540800346e-05, 9.816829900053113e-06, -9.816829900053054e-06, -2.9388424540800176e-05, -4.8774216099391756e-05, -6.785164135993873e-05, -8.650008669757386e-05, -0.00010460165063662167, -0.00012204188926168749, -0.00013871053976893967, -0.00015450221758305568, -0.00016931708263243862, -0.00018306147057029526, -0.00019564848495079334, -0.00020699854661637105, -0.00021703989682279182, -0.00022570905092103135, -0.00023295119972766782, -0.00023872055604618635, -0.00024298064414837828, -0.0002457045303856384, -0.00024687499347216343, -0.00024648463336346556, -0.00024453591804183655, -0.0002410411679129688, -0.0002360224779123837, -0.0002295115778141358, -0.0002215496316249646, -0.00021218697733219069, -0.00020148280865075246, -0.00018950480078148174, -0.00017632868254669463, -0.000162037757608194, -0.00014672237779469896, -0.00013047937186849278, -0.0001134114333428111, -9.562647122038798e-05, -7.723692775799665e-05, -5.835906757029844e-05, -3.9112242567516326e-05, -1.9618137374230055e-05, -6.047888898493356e-20],
        },
        "qubit0_xy$x_12$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit0_xy$x_12$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 1.961813737422993e-05, 3.9112242567516353e-05, 5.83590675702984e-05, 7.723692775799656e-05, 9.562647122038802e-05, 0.0001134114333428111, 0.00013047937186849273, 0.00014672237779469904, 0.000162037757608194, 0.0001763286825466946, 0.00018950480078148177, 0.00020148280865075246, 0.00021218697733219069, 0.00022154963162496455, 0.0002295115778141358, 0.0002360224779123837, 0.00024104116791296874, 0.00024453591804183655, 0.0002464846333634655, 0.00024687499347216343, 0.0002457045303856384, 0.00024298064414837836, 0.0002387205560461864, 0.0002329511997276678, 0.0002257090509210314, 0.00021703989682279182, 0.000206998546616371, 0.00019564848495079342, 0.00018306147057029523, 0.00016931708263243875, 0.00015450221758305573, 0.00013871053976893965, 0.00012204188926168753, 0.00010460165063662163, 8.650008669757402e-05, 6.78516413599388e-05, 4.877421609939171e-05, 2.9388424540800346e-05, 9.816829900053113e-06, -9.816829900053054e-06, -2.9388424540800176e-05, -4.8774216099391756e-05, -6.785164135993873e-05, -8.650008669757386e-05, -0.00010460165063662167, -0.00012204188926168749, -0.00013871053976893967, -0.00015450221758305568, -0.00016931708263243862, -0.00018306147057029526, -0.00019564848495079334, -0.00020699854661637105, -0.00021703989682279182, -0.00022570905092103135, -0.00023295119972766782, -0.00023872055604618635, -0.00024298064414837828, -0.0002457045303856384, -0.00024687499347216343, -0.00024648463336346556, -0.00024453591804183655, -0.0002410411679129688, -0.0002360224779123837, -0.0002295115778141358, -0.0002215496316249646, -0.00021218697733219069, -0.00020148280865075246, -0.00018950480078148174, -0.00017632868254669463, -0.000162037757608194, -0.00014672237779469896, -0.00013047937186849278, -0.0001134114333428111, -9.562647122038798e-05, -7.723692775799665e-05, -5.835906757029844e-05, -3.9112242567516326e-05, -1.9618137374230055e-05, -6.047888898493356e-20],
        },
        "qubit0_xy$id$drag$i": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit0_xy$id$drag$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit0_xy$y$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit0_xy$y$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit0_xy$sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit0_xy$sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit0_xy$-sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit0_xy$-sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit0_xy$sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit0_xy$sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit0_xy$-sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit0_xy$-sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit0_rr$readout$rect$rotation$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit0_rr$readout$rect$rotation$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit1_xy$cw$rect$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit1_xy$cw$rect$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit1_xy$x$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit1_xy$x$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.0014309468777483e-05, 3.990208202907396e-05, 5.9537580779485e-05, 7.879666377485707e-05, 9.755756889163469e-05, 0.00011570168364720961, 0.00013311429510622539, 0.00014968531513268908, 0.00016530997640261188, 0.00017988949477676375, 0.00019333169384580686, 0.00020555158769923368, 0.00021647191823365625, 0.00022602364360341045, 0.00023414637472533384, 0.000240788757077993, 0.0002459087953815025, 0.00024947411910520564, 0.00025146218712459185, 0.000251860430233543, 0.00025066633061090175, 0.00024788743773894754, 0.00024354132067313847, 0.00023765545696488672, 0.00023026705893963085, 0.00022142283842853359, 0.00021117871144124532, 0.0001995994446468871, 0.00018675824589831674, 0.00017273630138851128, 0.0001576222623653155, 0.00014151168464971047, 0.00012450642450114686, 0.00010671399464947767, 8.824688456486483e-05, 6.922184926312981e-05, 4.9759171142952236e-05, 2.998189952181845e-05, 1.0015072678617293e-05, -1.0015072678617232e-05, -2.9981899521818275e-05, -4.975917114295229e-05, -6.922184926312973e-05, -8.824688456486466e-05, -0.00010671399464947772, -0.0001245064245011468, -0.00014151168464971047, -0.00015762226236531543, -0.00017273630138851115, -0.00018675824589831677, -0.00019959944464688703, -0.00021117871144124532, -0.0002214228384285336, -0.0002302670589396308, -0.00023765545696488675, -0.00024354132067313847, -0.0002478874377389475, -0.0002506663306109018, -0.000251860430233543, -0.00025146218712459185, -0.00024947411910520564, -0.0002459087953815025, -0.000240788757077993, -0.00023414637472533384, -0.00022602364360341048, -0.00021647191823365625, -0.00020555158769923368, -0.0001933316938458068, -0.0001798894947767638, -0.00016530997640261188, -0.000149685315132689, -0.00013311429510622547, -0.00011570168364720961, -9.755756889163465e-05, -7.879666377485716e-05, -5.953758077948505e-05, -3.9902082029073934e-05, -2.0014309468777612e-05, -6.170021023822154e-20],
        },
        "qubit1_xy$x_12$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit1_xy$x_12$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.0014309468777483e-05, 3.990208202907396e-05, 5.9537580779485e-05, 7.879666377485707e-05, 9.755756889163469e-05, 0.00011570168364720961, 0.00013311429510622539, 0.00014968531513268908, 0.00016530997640261188, 0.00017988949477676375, 0.00019333169384580686, 0.00020555158769923368, 0.00021647191823365625, 0.00022602364360341045, 0.00023414637472533384, 0.000240788757077993, 0.0002459087953815025, 0.00024947411910520564, 0.00025146218712459185, 0.000251860430233543, 0.00025066633061090175, 0.00024788743773894754, 0.00024354132067313847, 0.00023765545696488672, 0.00023026705893963085, 0.00022142283842853359, 0.00021117871144124532, 0.0001995994446468871, 0.00018675824589831674, 0.00017273630138851128, 0.0001576222623653155, 0.00014151168464971047, 0.00012450642450114686, 0.00010671399464947767, 8.824688456486483e-05, 6.922184926312981e-05, 4.9759171142952236e-05, 2.998189952181845e-05, 1.0015072678617293e-05, -1.0015072678617232e-05, -2.9981899521818275e-05, -4.975917114295229e-05, -6.922184926312973e-05, -8.824688456486466e-05, -0.00010671399464947772, -0.0001245064245011468, -0.00014151168464971047, -0.00015762226236531543, -0.00017273630138851115, -0.00018675824589831677, -0.00019959944464688703, -0.00021117871144124532, -0.0002214228384285336, -0.0002302670589396308, -0.00023765545696488675, -0.00024354132067313847, -0.0002478874377389475, -0.0002506663306109018, -0.000251860430233543, -0.00025146218712459185, -0.00024947411910520564, -0.0002459087953815025, -0.000240788757077993, -0.00023414637472533384, -0.00022602364360341048, -0.00021647191823365625, -0.00020555158769923368, -0.0001933316938458068, -0.0001798894947767638, -0.00016530997640261188, -0.000149685315132689, -0.00013311429510622547, -0.00011570168364720961, -9.755756889163465e-05, -7.879666377485716e-05, -5.953758077948505e-05, -3.9902082029073934e-05, -2.0014309468777612e-05, -6.170021023822154e-20],
        },
        "qubit1_xy$id$drag$i": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit1_xy$id$drag$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit1_xy$y$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit1_xy$y$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit1_xy$sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit1_xy$sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit1_xy$-sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit1_xy$-sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit1_xy$sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit1_xy$sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit1_xy$-sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit1_xy$-sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit1_rr$readout$rect$rotation$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit1_rr$readout$rect$rotation$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit2_xy$cw$rect$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit2_xy$cw$rect$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit2_xy$x$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit2_xy$x$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.2018026023893007e-05, 4.389684699810573e-05, 6.549813797210189e-05, 8.668532862936312e-05, 0.0001073244667289015, 0.00012728506499445596, 0.0001464409260970656, 0.00016467094051521563, 0.00018185985222822547, 0.0001978989874019213, 0.00021268694145960476, 0.00022613022019443675, 0.00023814383086991498, 0.0002486518195713194, 0.0002575877514108164, 0.0002648951305502093, 0.0002705277573858173, 0.00027445002063724563, 0.0002766371224933652, 0.00027707523539205914, 0.0002757615894425223, 0.0002727044899374032, 0.00026792326484406935, 0.00026144814260697474, 0.0002533200610337013, 0.00024359040847295855, 0.0002323206989208979, 0.0002195821831098225, 0.00020545539803811777, 0.00019002965778941667, 0.00017340248886020624, 0.00015567901356591674, 0.00013697128542379688, 0.00011739758071449959, 9.708165070135491e-05, 7.615193923503812e-05, 5.474077069019108e-05, 3.298351336812155e-05, 1.1017723654816893e-05, -1.1017723654816825e-05, -3.298351336812136e-05, -5.474077069019114e-05, -7.615193923503805e-05, -9.708165070135472e-05, -0.00011739758071449965, -0.00013697128542379682, -0.00015567901356591677, -0.00017340248886020616, -0.00019002965778941653, -0.00020545539803811777, -0.00021958218310982243, -0.00023232069892089796, -0.0002435904084729586, -0.00025332006103370117, -0.00026144814260697474, -0.00026792326484406935, -0.0002727044899374032, -0.0002757615894425223, -0.00027707523539205914, -0.0002766371224933652, -0.00027445002063724563, -0.00027052775738581737, -0.0002648951305502093, -0.0002575877514108164, -0.0002486518195713195, -0.00023814383086991498, -0.00022613022019443675, -0.00021268694145960474, -0.00019789898740192136, -0.00018185985222822547, -0.00016467094051521552, -0.0001464409260970657, -0.00012728506499445596, -0.00010732446672890146, -8.668532862936323e-05, -6.549813797210193e-05, -4.38968469981057e-05, -2.2018026023893145e-05, -6.787727734619727e-20],
        },
        "qubit2_xy$x_12$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit2_xy$x_12$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.2018026023893007e-05, 4.389684699810573e-05, 6.549813797210189e-05, 8.668532862936312e-05, 0.0001073244667289015, 0.00012728506499445596, 0.0001464409260970656, 0.00016467094051521563, 0.00018185985222822547, 0.0001978989874019213, 0.00021268694145960476, 0.00022613022019443675, 0.00023814383086991498, 0.0002486518195713194, 0.0002575877514108164, 0.0002648951305502093, 0.0002705277573858173, 0.00027445002063724563, 0.0002766371224933652, 0.00027707523539205914, 0.0002757615894425223, 0.0002727044899374032, 0.00026792326484406935, 0.00026144814260697474, 0.0002533200610337013, 0.00024359040847295855, 0.0002323206989208979, 0.0002195821831098225, 0.00020545539803811777, 0.00019002965778941667, 0.00017340248886020624, 0.00015567901356591674, 0.00013697128542379688, 0.00011739758071449959, 9.708165070135491e-05, 7.615193923503812e-05, 5.474077069019108e-05, 3.298351336812155e-05, 1.1017723654816893e-05, -1.1017723654816825e-05, -3.298351336812136e-05, -5.474077069019114e-05, -7.615193923503805e-05, -9.708165070135472e-05, -0.00011739758071449965, -0.00013697128542379682, -0.00015567901356591677, -0.00017340248886020616, -0.00019002965778941653, -0.00020545539803811777, -0.00021958218310982243, -0.00023232069892089796, -0.0002435904084729586, -0.00025332006103370117, -0.00026144814260697474, -0.00026792326484406935, -0.0002727044899374032, -0.0002757615894425223, -0.00027707523539205914, -0.0002766371224933652, -0.00027445002063724563, -0.00027052775738581737, -0.0002648951305502093, -0.0002575877514108164, -0.0002486518195713195, -0.00023814383086991498, -0.00022613022019443675, -0.00021268694145960474, -0.00019789898740192136, -0.00018185985222822547, -0.00016467094051521552, -0.0001464409260970657, -0.00012728506499445596, -0.00010732446672890146, -8.668532862936323e-05, -6.549813797210193e-05, -4.38968469981057e-05, -2.2018026023893145e-05, -6.787727734619727e-20],
        },
        "qubit2_xy$id$drag$i": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit2_xy$id$drag$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit2_xy$y$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit2_xy$y$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit2_xy$sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit2_xy$sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit2_xy$-sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit2_xy$-sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit2_xy$sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit2_xy$sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit2_xy$-sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit2_xy$-sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit2_rr$readout$rect$rotation$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit2_rr$readout$rect$rotation$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit3_xy$cw$rect$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit3_xy$cw$rect$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit3_xy$x$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit3_xy$x$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.6208307793146006e-05, 5.2250918226118936e-05, 7.79631815308064e-05, 0.00010318253650000317, 0.00012774953825170284, 0.0001515088662909787, 0.00017431030649616114, 0.0001960097008208795, 0.00021646985870764558, 0.0002355614244507299, 0.0002531636950245815, 0.00026916538320721886, 0.000283465321173878, 0.000295973100112574, 0.00030660964181771926, 0.000315307698648, 0.0003220122786876165, 0.00032668099342288067, 0.0003292843257360472, 0.0003298058165220358, 0.0003282421687481927, 0.00032460326829919284, 0.00031891212147529366, 0.000311204709539098, 0.0003015297612304264, 0.00028994844468753543, 0.00027653398072245293, 0.0002613711798954289, 0.0002445559063152682, 0.00022619447155556866, 0.000206402962518723, 0.00018530650749714657, 0.00016303848507192434, 0.00013973968085047704, 0.00011555739737462315, 9.064452282648136e-05, 6.515856442015824e-05, 3.9260652590428426e-05, 1.3114522274245715e-05, -1.3114522274245634e-05, -3.92606525904282e-05, -6.51585644201583e-05, -9.064452282648127e-05, -0.00011555739737462293, -0.00013973968085047712, -0.00016303848507192426, -0.00018530650749714662, -0.0002064029625187229, -0.00022619447155556847, -0.00024455590631526826, -0.0002613711798954288, -0.00027653398072245293, -0.0002899484446875355, -0.0003015297612304264, -0.00031120470953909803, -0.0003189121214752936, -0.0003246032682991928, -0.00032824216874819274, -0.0003298058165220358, -0.00032928432573604724, -0.00032668099342288067, -0.00032201227868761654, -0.000315307698648, -0.00030660964181771926, -0.0002959731001125741, -0.000283465321173878, -0.00026916538320721886, -0.00025316369502458144, -0.0002355614244507299, -0.00021646985870764558, -0.0001960097008208794, -0.00017431030649616124, -0.00015150886629097872, -0.00012774953825170276, -0.0001031825365000033, -7.796318153080646e-05, -5.22509182261189e-05, -2.620830779314617e-05, -8.079509829443551e-20],
        },
        "qubit3_xy$x_12$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit3_xy$x_12$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.6208307793146006e-05, 5.2250918226118936e-05, 7.79631815308064e-05, 0.00010318253650000317, 0.00012774953825170284, 0.0001515088662909787, 0.00017431030649616114, 0.0001960097008208795, 0.00021646985870764558, 0.0002355614244507299, 0.0002531636950245815, 0.00026916538320721886, 0.000283465321173878, 0.000295973100112574, 0.00030660964181771926, 0.000315307698648, 0.0003220122786876165, 0.00032668099342288067, 0.0003292843257360472, 0.0003298058165220358, 0.0003282421687481927, 0.00032460326829919284, 0.00031891212147529366, 0.000311204709539098, 0.0003015297612304264, 0.00028994844468753543, 0.00027653398072245293, 0.0002613711798954289, 0.0002445559063152682, 0.00022619447155556866, 0.000206402962518723, 0.00018530650749714657, 0.00016303848507192434, 0.00013973968085047704, 0.00011555739737462315, 9.064452282648136e-05, 6.515856442015824e-05, 3.9260652590428426e-05, 1.3114522274245715e-05, -1.3114522274245634e-05, -3.92606525904282e-05, -6.51585644201583e-05, -9.064452282648127e-05, -0.00011555739737462293, -0.00013973968085047712, -0.00016303848507192426, -0.00018530650749714662, -0.0002064029625187229, -0.00022619447155556847, -0.00024455590631526826, -0.0002613711798954288, -0.00027653398072245293, -0.0002899484446875355, -0.0003015297612304264, -0.00031120470953909803, -0.0003189121214752936, -0.0003246032682991928, -0.00032824216874819274, -0.0003298058165220358, -0.00032928432573604724, -0.00032668099342288067, -0.00032201227868761654, -0.000315307698648, -0.00030660964181771926, -0.0002959731001125741, -0.000283465321173878, -0.00026916538320721886, -0.00025316369502458144, -0.0002355614244507299, -0.00021646985870764558, -0.0001960097008208794, -0.00017431030649616124, -0.00015150886629097872, -0.00012774953825170276, -0.0001031825365000033, -7.796318153080646e-05, -5.22509182261189e-05, -2.620830779314617e-05, -8.079509829443551e-20],
        },
        "qubit3_xy$id$drag$i": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit3_xy$id$drag$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit3_xy$y$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit3_xy$y$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit3_xy$sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit3_xy$sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit3_xy$-sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit3_xy$-sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit3_xy$sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit3_xy$sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit3_xy$-sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit3_xy$-sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit3_rr$readout$rect$rotation$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit3_rr$readout$rect$rotation$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit4_xy$cw$rect$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit4_xy$cw$rect$q": {
            "type": "constant",
            "sample": 0.0,
        },
        "qubit4_xy$x$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit4_xy$x$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.567880855139495e-05, 5.119526740730637e-05, 7.638805329929739e-05, 0.0001010978893236223, 0.0001251685519410943, 0.00014844785867259994, 0.00017078863024573, 0.00019204962110952743, 0.00021209641243433706, 0.0002308022619509259, 0.00024804890525591624, 0.00026372730351742187, 0.00027773833285365, 0.00028999341102599736, 0.0003004150574844834, 0.00030893738322473375, 0.00031550650735948007, 0.00032008089777087775, 0.0003226316336899265, 0.00032314258854288425, 0.000321610531908658, 0.00031804514994256675, 0.0003124689841373481, 0.0003049172888085843, 0.00029543780820556753, 0.0002840904746567844, 0.0002709470296584383, 0.0002560905703016114, 0.00023961502390570106, 0.000221624554179664, 0.00020223290266551048, 0.0001815626696276548, 0.00015974453893457006, 0.00013691645183329444, 0.00011322273484045564, 8.881318726356264e-05, 6.38421341215533e-05, 3.8467450452336926e-05, 1.2849563175972438e-05, -1.284956317597236e-05, -3.846745045233671e-05, -6.384213412155337e-05, -8.881318726356254e-05, -0.00011322273484045542, -0.00013691645183329452, -0.00015974453893457, -0.00018156266962765486, -0.00020223290266551037, -0.00022162455417966385, -0.00023961502390570106, -0.00025609057030161127, -0.0002709470296584383, -0.00028409047465678444, -0.0002954378082055675, -0.0003049172888085843, -0.00031246898413734805, -0.0003180451499425667, -0.0003216105319086581, -0.00032314258854288425, -0.0003226316336899266, -0.00032008089777087775, -0.0003155065073594801, -0.00030893738322473375, -0.0003004150574844834, -0.00028999341102599736, -0.00027773833285365, -0.00026372730351742187, -0.00024804890525591624, -0.00023080226195092593, -0.00021209641243433706, -0.00019204962110952733, -0.00017078863024573012, -0.00014844785867259994, -0.00012516855194109427, -0.00010109788932362242, -7.638805329929744e-05, -5.1195267407306334e-05, -2.5678808551395105e-05, -7.916275546552178e-20],
        },
        "qubit4_xy$x_12$drag$i": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit4_xy$x_12$drag$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.567880855139495e-05, 5.119526740730637e-05, 7.638805329929739e-05, 0.0001010978893236223, 0.0001251685519410943, 0.00014844785867259994, 0.00017078863024573, 0.00019204962110952743, 0.00021209641243433706, 0.0002308022619509259, 0.00024804890525591624, 0.00026372730351742187, 0.00027773833285365, 0.00028999341102599736, 0.0003004150574844834, 0.00030893738322473375, 0.00031550650735948007, 0.00032008089777087775, 0.0003226316336899265, 0.00032314258854288425, 0.000321610531908658, 0.00031804514994256675, 0.0003124689841373481, 0.0003049172888085843, 0.00029543780820556753, 0.0002840904746567844, 0.0002709470296584383, 0.0002560905703016114, 0.00023961502390570106, 0.000221624554179664, 0.00020223290266551048, 0.0001815626696276548, 0.00015974453893457006, 0.00013691645183329444, 0.00011322273484045564, 8.881318726356264e-05, 6.38421341215533e-05, 3.8467450452336926e-05, 1.2849563175972438e-05, -1.284956317597236e-05, -3.846745045233671e-05, -6.384213412155337e-05, -8.881318726356254e-05, -0.00011322273484045542, -0.00013691645183329452, -0.00015974453893457, -0.00018156266962765486, -0.00020223290266551037, -0.00022162455417966385, -0.00023961502390570106, -0.00025609057030161127, -0.0002709470296584383, -0.00028409047465678444, -0.0002954378082055675, -0.0003049172888085843, -0.00031246898413734805, -0.0003180451499425667, -0.0003216105319086581, -0.00032314258854288425, -0.0003226316336899266, -0.00032008089777087775, -0.0003155065073594801, -0.00030893738322473375, -0.0003004150574844834, -0.00028999341102599736, -0.00027773833285365, -0.00026372730351742187, -0.00024804890525591624, -0.00023080226195092593, -0.00021209641243433706, -0.00019204962110952733, -0.00017078863024573012, -0.00014844785867259994, -0.00012516855194109427, -0.00010109788932362242, -7.638805329929744e-05, -5.1195267407306334e-05, -2.5678808551395105e-05, -7.916275546552178e-20],
        },
        "qubit4_xy$id$drag$i": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit4_xy$id$drag$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit4_xy$y$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit4_xy$y$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
        },
        "qubit4_xy$sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit4_xy$sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit4_xy$-sy$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
        },
        "qubit4_xy$-sy$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit4_xy$sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
        },
        "qubit4_xy$sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0] * 80,
        },
        "qubit4_xy$-sx$drag_rot$i": {
            "type": "arbitrary",
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
        },
        "qubit4_xy$-sx$drag_rot$q": {
            "type": "arbitrary",
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
        },
        "qubit4_rr$readout$rect$rotation$i": {
            "type": "constant",
            "sample": 0.01,
        },
        "qubit4_rr$readout$rect$rotation$q": {
            "type": "constant",
            "sample": 0.0,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "qubit0_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(-0.0, 1000)],
        },
        "qubit0_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit0_rr$readout$rect$rotation$w3": {
            "cosine": [(-0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit1_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(-0.0, 1000)],
        },
        "qubit1_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit1_rr$readout$rect$rotation$w3": {
            "cosine": [(-0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit2_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(-0.0, 1000)],
        },
        "qubit2_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit2_rr$readout$rect$rotation$w3": {
            "cosine": [(-0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit3_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(-0.0, 1000)],
        },
        "qubit3_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit3_rr$readout$rect$rotation$w3": {
            "cosine": [(-0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit4_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(-0.0, 1000)],
        },
        "qubit4_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit4_rr$readout$rect$rotation$w3": {
            "cosine": [(-0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
    },
    "mixers": {
        "octave_Octave_2": [
            {'intermediate_frequency': 205000000.0, 'lo_frequency': 4945000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 205000000.0, 'lo_frequency': 5575000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
        ],
        "octave_Octave_1": [
            {'intermediate_frequency': -104000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': -35000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 118000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 143000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 245000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
        ],
        "octave_Octave_3": [{'intermediate_frequency': 205000000.0, 'lo_frequency': 5049000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]}],
        "octave_Octave_4": [{'intermediate_frequency': 205000000.0, 'lo_frequency': 6675000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]}],
        "octave_Octave_5": [{'intermediate_frequency': 205000000.0, 'lo_frequency': 6536000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]}],
    },
}

loaded_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "2": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "3": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "4": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "5": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "6": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "7": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "8": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "9": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "10": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
            },
            "digital_outputs": {
                "1": {
                    "shareable": False,
                    "inverted": False,
                },
                "2": {
                    "shareable": False,
                    "inverted": False,
                },
                "3": {
                    "shareable": False,
                    "inverted": False,
                },
                "4": {
                    "shareable": False,
                    "inverted": False,
                },
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                "3": {
                    "offset": 0.018766451590108237,
                    "delay": 0,
                    "shareable": False,
                },
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "2": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                },
                "4": {
                    "offset": -0.03823046968983945,
                    "delay": 0,
                    "shareable": False,
                },
                "5": {
                    "offset": 0.03181726374202038,
                    "delay": 0,
                    "shareable": False,
                },
                "6": {
                    "offset": 0.009284224666683593,
                    "delay": 0,
                    "shareable": False,
                },
                "7": {
                    "offset": 0.005801026962879232,
                    "delay": 0,
                    "shareable": False,
                },
            },
            "analog_inputs": {
                "1": {
                    "offset": 0.013969112792968748,
                    "gain_db": 0,
                    "shareable": False,
                },
                "2": {
                    "offset": 0.01661838671875,
                    "gain_db": 0,
                    "shareable": False,
                },
            },
            "digital_outputs": {
                "5": {
                    "shareable": False,
                    "inverted": False,
                },
            },
        },
    },
    "oscillators": {},
    "elements": {
        "qubit0_xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit0_xy$cw$rect",
                "x$drag": "qubit0_xy$x$drag",
                "x_12$drag": "qubit0_xy$x_12$drag",
                "id$drag": "qubit0_xy$id$drag",
                "y$drag_rot": "qubit0_xy$y$drag_rot",
                "sy$drag_rot": "qubit0_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit0_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit0_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit0_xy$-sx$drag_rot",
            },
            "mixInputs": {
                "I": ('con1', 1),
                "Q": ('con1', 2),
                "mixer": "octave_Octave_2",
                "lo_frequency": 4945000000.0,
            },
        },
        "qubit0_z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 0.0,
            "singleInput": {
                "port": ('con2', 3),
            },
        },
        "qubit0_rr": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con2', 1),
                "out2": ('con2', 2),
            },
            "time_of_flight": 284,
            "smearing": 0,
            "intermediate_frequency": 104000000.0,
            "operations": {
                "readout$rect$rotation": "qubit0_rr$readout$rect$rotation",
            },
            "mixInputs": {
                "I": ('con2', 1),
                "Q": ('con2', 2),
                "mixer": "octave_Octave_1",
                "lo_frequency": 7350000000.0,
            },
        },
        "qubit0_blanker": {
            "digitalInputs": {
                "qubit0_blanker$input": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 1),
                },
            },
            "digitalOutputs": {},
        },
        "qubit1_xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit1_xy$cw$rect",
                "x$drag": "qubit1_xy$x$drag",
                "x_12$drag": "qubit1_xy$x_12$drag",
                "id$drag": "qubit1_xy$id$drag",
                "y$drag_rot": "qubit1_xy$y$drag_rot",
                "sy$drag_rot": "qubit1_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit1_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit1_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit1_xy$-sx$drag_rot",
            },
            "mixInputs": {
                "I": ('con1', 3),
                "Q": ('con1', 4),
                "mixer": "octave_Octave_3",
                "lo_frequency": 5049000000.0,
            },
        },
        "qubit1_z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 0.0,
            "singleInput": {
                "port": ('con2', 4),
            },
        },
        "qubit1_rr": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con2', 1),
                "out2": ('con2', 2),
            },
            "time_of_flight": 284,
            "smearing": 0,
            "intermediate_frequency": 35000000.0,
            "operations": {
                "readout$rect$rotation": "qubit1_rr$readout$rect$rotation",
            },
            "mixInputs": {
                "I": ('con2', 1),
                "Q": ('con2', 2),
                "mixer": "octave_Octave_1",
                "lo_frequency": 7350000000.0,
            },
        },
        "qubit1_blanker": {
            "digitalInputs": {
                "qubit1_blanker$input": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 2),
                },
            },
            "digitalOutputs": {},
        },
        "qubit2_xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit2_xy$cw$rect",
                "x$drag": "qubit2_xy$x$drag",
                "x_12$drag": "qubit2_xy$x_12$drag",
                "id$drag": "qubit2_xy$id$drag",
                "y$drag_rot": "qubit2_xy$y$drag_rot",
                "sy$drag_rot": "qubit2_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit2_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit2_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit2_xy$-sx$drag_rot",
            },
            "mixInputs": {
                "I": ('con1', 5),
                "Q": ('con1', 6),
                "mixer": "octave_Octave_2",
                "lo_frequency": 5575000000.0,
            },
        },
        "qubit2_z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 0.0,
            "singleInput": {
                "port": ('con2', 5),
            },
        },
        "qubit2_rr": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con2', 1),
                "out2": ('con2', 2),
            },
            "time_of_flight": 284,
            "smearing": 0,
            "intermediate_frequency": 118000000.0,
            "operations": {
                "readout$rect$rotation": "qubit2_rr$readout$rect$rotation",
            },
            "mixInputs": {
                "I": ('con2', 1),
                "Q": ('con2', 2),
                "mixer": "octave_Octave_1",
                "lo_frequency": 7350000000.0,
            },
        },
        "qubit2_blanker": {
            "digitalInputs": {
                "qubit2_blanker$input": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 3),
                },
            },
            "digitalOutputs": {},
        },
        "qubit3_xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit3_xy$cw$rect",
                "x$drag": "qubit3_xy$x$drag",
                "x_12$drag": "qubit3_xy$x_12$drag",
                "id$drag": "qubit3_xy$id$drag",
                "y$drag_rot": "qubit3_xy$y$drag_rot",
                "sy$drag_rot": "qubit3_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit3_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit3_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit3_xy$-sx$drag_rot",
            },
            "mixInputs": {
                "I": ('con1', 7),
                "Q": ('con1', 8),
                "mixer": "octave_Octave_4",
                "lo_frequency": 6675000000.0,
            },
        },
        "qubit3_z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 0.0,
            "singleInput": {
                "port": ('con2', 6),
            },
        },
        "qubit3_rr": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con2', 1),
                "out2": ('con2', 2),
            },
            "time_of_flight": 284,
            "smearing": 0,
            "intermediate_frequency": 143000000.0,
            "operations": {
                "readout$rect$rotation": "qubit3_rr$readout$rect$rotation",
            },
            "mixInputs": {
                "I": ('con2', 1),
                "Q": ('con2', 2),
                "mixer": "octave_Octave_1",
                "lo_frequency": 7350000000.0,
            },
        },
        "qubit3_blanker": {
            "digitalInputs": {
                "qubit3_blanker$input": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con1', 4),
                },
            },
            "digitalOutputs": {},
        },
        "qubit4_xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 205000000.0,
            "operations": {
                "cw$rect": "qubit4_xy$cw$rect",
                "x$drag": "qubit4_xy$x$drag",
                "x_12$drag": "qubit4_xy$x_12$drag",
                "id$drag": "qubit4_xy$id$drag",
                "y$drag_rot": "qubit4_xy$y$drag_rot",
                "sy$drag_rot": "qubit4_xy$sy$drag_rot",
                "-sy$drag_rot": "qubit4_xy$-sy$drag_rot",
                "sx$drag_rot": "qubit4_xy$sx$drag_rot",
                "-sx$drag_rot": "qubit4_xy$-sx$drag_rot",
            },
            "mixInputs": {
                "I": ('con1', 9),
                "Q": ('con1', 10),
                "mixer": "octave_Octave_5",
                "lo_frequency": 6536000000.0,
            },
        },
        "qubit4_z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "intermediate_frequency": 0.0,
            "singleInput": {
                "port": ('con2', 7),
            },
        },
        "qubit4_rr": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {
                "out1": ('con2', 1),
                "out2": ('con2', 2),
            },
            "time_of_flight": 284,
            "smearing": 0,
            "intermediate_frequency": 245000000.0,
            "operations": {
                "readout$rect$rotation": "qubit4_rr$readout$rect$rotation",
            },
            "mixInputs": {
                "I": ('con2', 1),
                "Q": ('con2', 2),
                "mixer": "octave_Octave_1",
                "lo_frequency": 7350000000.0,
            },
        },
        "qubit4_blanker": {
            "digitalInputs": {
                "qubit4_blanker$input": {
                    "delay": 0,
                    "buffer": 0,
                    "port": ('con2', 5),
                },
            },
            "digitalOutputs": {},
        },
    },
    "pulses": {
        "qubit0_xy$cw$rect": {
            "length": 1000,
            "waveforms": {
                "I": "qubit0_xy$cw$rect$i",
                "Q": "qubit0_xy$cw$rect$q",
            },
            "operation": "control",
        },
        "qubit0_xy$x$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$x$drag$i",
                "Q": "qubit0_xy$x$drag$q",
            },
            "operation": "control",
        },
        "qubit0_xy$x_12$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$x_12$drag$i",
                "Q": "qubit0_xy$x_12$drag$q",
            },
            "operation": "control",
        },
        "qubit0_xy$id$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$id$drag$i",
                "Q": "qubit0_xy$id$drag$q",
            },
            "operation": "control",
        },
        "qubit0_xy$y$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$y$drag_rot$i",
                "Q": "qubit0_xy$y$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit0_xy$sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$sy$drag_rot$i",
                "Q": "qubit0_xy$sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit0_xy$-sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$-sy$drag_rot$i",
                "Q": "qubit0_xy$-sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit0_xy$sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$sx$drag_rot$i",
                "Q": "qubit0_xy$sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit0_xy$-sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit0_xy$-sx$drag_rot$i",
                "Q": "qubit0_xy$-sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit0_rr$readout$rect$rotation": {
            "length": 1000,
            "waveforms": {
                "I": "qubit0_rr$readout$rect$rotation$i",
                "Q": "qubit0_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit0_rr$readout$rect$rotation$w1",
                "w2": "qubit0_rr$readout$rect$rotation$w2",
                "w3": "qubit0_rr$readout$rect$rotation$w3",
            },
            "operation": "measurement",
        },
        "qubit1_xy$cw$rect": {
            "length": 1000,
            "waveforms": {
                "I": "qubit1_xy$cw$rect$i",
                "Q": "qubit1_xy$cw$rect$q",
            },
            "operation": "control",
        },
        "qubit1_xy$x$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$x$drag$i",
                "Q": "qubit1_xy$x$drag$q",
            },
            "operation": "control",
        },
        "qubit1_xy$x_12$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$x_12$drag$i",
                "Q": "qubit1_xy$x_12$drag$q",
            },
            "operation": "control",
        },
        "qubit1_xy$id$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$id$drag$i",
                "Q": "qubit1_xy$id$drag$q",
            },
            "operation": "control",
        },
        "qubit1_xy$y$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$y$drag_rot$i",
                "Q": "qubit1_xy$y$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit1_xy$sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$sy$drag_rot$i",
                "Q": "qubit1_xy$sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit1_xy$-sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$-sy$drag_rot$i",
                "Q": "qubit1_xy$-sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit1_xy$sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$sx$drag_rot$i",
                "Q": "qubit1_xy$sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit1_xy$-sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit1_xy$-sx$drag_rot$i",
                "Q": "qubit1_xy$-sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit1_rr$readout$rect$rotation": {
            "length": 1000,
            "waveforms": {
                "I": "qubit1_rr$readout$rect$rotation$i",
                "Q": "qubit1_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit1_rr$readout$rect$rotation$w1",
                "w2": "qubit1_rr$readout$rect$rotation$w2",
                "w3": "qubit1_rr$readout$rect$rotation$w3",
            },
            "operation": "measurement",
        },
        "qubit2_xy$cw$rect": {
            "length": 1000,
            "waveforms": {
                "I": "qubit2_xy$cw$rect$i",
                "Q": "qubit2_xy$cw$rect$q",
            },
            "operation": "control",
        },
        "qubit2_xy$x$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$x$drag$i",
                "Q": "qubit2_xy$x$drag$q",
            },
            "operation": "control",
        },
        "qubit2_xy$x_12$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$x_12$drag$i",
                "Q": "qubit2_xy$x_12$drag$q",
            },
            "operation": "control",
        },
        "qubit2_xy$id$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$id$drag$i",
                "Q": "qubit2_xy$id$drag$q",
            },
            "operation": "control",
        },
        "qubit2_xy$y$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$y$drag_rot$i",
                "Q": "qubit2_xy$y$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit2_xy$sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$sy$drag_rot$i",
                "Q": "qubit2_xy$sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit2_xy$-sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$-sy$drag_rot$i",
                "Q": "qubit2_xy$-sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit2_xy$sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$sx$drag_rot$i",
                "Q": "qubit2_xy$sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit2_xy$-sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit2_xy$-sx$drag_rot$i",
                "Q": "qubit2_xy$-sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit2_rr$readout$rect$rotation": {
            "length": 1000,
            "waveforms": {
                "I": "qubit2_rr$readout$rect$rotation$i",
                "Q": "qubit2_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit2_rr$readout$rect$rotation$w1",
                "w2": "qubit2_rr$readout$rect$rotation$w2",
                "w3": "qubit2_rr$readout$rect$rotation$w3",
            },
            "operation": "measurement",
        },
        "qubit3_xy$cw$rect": {
            "length": 1000,
            "waveforms": {
                "I": "qubit3_xy$cw$rect$i",
                "Q": "qubit3_xy$cw$rect$q",
            },
            "operation": "control",
        },
        "qubit3_xy$x$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$x$drag$i",
                "Q": "qubit3_xy$x$drag$q",
            },
            "operation": "control",
        },
        "qubit3_xy$x_12$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$x_12$drag$i",
                "Q": "qubit3_xy$x_12$drag$q",
            },
            "operation": "control",
        },
        "qubit3_xy$id$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$id$drag$i",
                "Q": "qubit3_xy$id$drag$q",
            },
            "operation": "control",
        },
        "qubit3_xy$y$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$y$drag_rot$i",
                "Q": "qubit3_xy$y$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit3_xy$sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$sy$drag_rot$i",
                "Q": "qubit3_xy$sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit3_xy$-sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$-sy$drag_rot$i",
                "Q": "qubit3_xy$-sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit3_xy$sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$sx$drag_rot$i",
                "Q": "qubit3_xy$sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit3_xy$-sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit3_xy$-sx$drag_rot$i",
                "Q": "qubit3_xy$-sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit3_rr$readout$rect$rotation": {
            "length": 1000,
            "waveforms": {
                "I": "qubit3_rr$readout$rect$rotation$i",
                "Q": "qubit3_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit3_rr$readout$rect$rotation$w1",
                "w2": "qubit3_rr$readout$rect$rotation$w2",
                "w3": "qubit3_rr$readout$rect$rotation$w3",
            },
            "operation": "measurement",
        },
        "qubit4_xy$cw$rect": {
            "length": 1000,
            "waveforms": {
                "I": "qubit4_xy$cw$rect$i",
                "Q": "qubit4_xy$cw$rect$q",
            },
            "operation": "control",
        },
        "qubit4_xy$x$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$x$drag$i",
                "Q": "qubit4_xy$x$drag$q",
            },
            "operation": "control",
        },
        "qubit4_xy$x_12$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$x_12$drag$i",
                "Q": "qubit4_xy$x_12$drag$q",
            },
            "operation": "control",
        },
        "qubit4_xy$id$drag": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$id$drag$i",
                "Q": "qubit4_xy$id$drag$q",
            },
            "operation": "control",
        },
        "qubit4_xy$y$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$y$drag_rot$i",
                "Q": "qubit4_xy$y$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit4_xy$sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$sy$drag_rot$i",
                "Q": "qubit4_xy$sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit4_xy$-sy$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$-sy$drag_rot$i",
                "Q": "qubit4_xy$-sy$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit4_xy$sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$sx$drag_rot$i",
                "Q": "qubit4_xy$sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit4_xy$-sx$drag_rot": {
            "length": 80,
            "waveforms": {
                "I": "qubit4_xy$-sx$drag_rot$i",
                "Q": "qubit4_xy$-sx$drag_rot$q",
            },
            "operation": "control",
        },
        "qubit4_rr$readout$rect$rotation": {
            "length": 1000,
            "waveforms": {
                "I": "qubit4_rr$readout$rect$rotation$i",
                "Q": "qubit4_rr$readout$rect$rotation$q",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "w1": "qubit4_rr$readout$rect$rotation$w1",
                "w2": "qubit4_rr$readout$rect$rotation$w2",
                "w3": "qubit4_rr$readout$rect$rotation$w3",
            },
            "operation": "measurement",
        },
    },
    "waveforms": {
        "qubit0_xy$cw$rect$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit0_xy$cw$rect$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit0_xy$x$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$x$drag$q": {
            "samples": [0.0, 1.961813737422993e-05, 3.9112242567516353e-05, 5.83590675702984e-05, 7.723692775799656e-05, 9.562647122038802e-05, 0.0001134114333428111, 0.00013047937186849273, 0.00014672237779469904, 0.000162037757608194, 0.0001763286825466946, 0.00018950480078148177, 0.00020148280865075246, 0.00021218697733219069, 0.00022154963162496455, 0.0002295115778141358, 0.0002360224779123837, 0.00024104116791296874, 0.00024453591804183655, 0.0002464846333634655, 0.00024687499347216343, 0.0002457045303856384, 0.00024298064414837836, 0.0002387205560461864, 0.0002329511997276678, 0.0002257090509210314, 0.00021703989682279182, 0.000206998546616371, 0.00019564848495079342, 0.00018306147057029523, 0.00016931708263243875, 0.00015450221758305573, 0.00013871053976893965, 0.00012204188926168753, 0.00010460165063662163, 8.650008669757402e-05, 6.78516413599388e-05, 4.877421609939171e-05, 2.9388424540800346e-05, 9.816829900053113e-06, -9.816829900053054e-06, -2.9388424540800176e-05, -4.8774216099391756e-05, -6.785164135993873e-05, -8.650008669757386e-05, -0.00010460165063662167, -0.00012204188926168749, -0.00013871053976893967, -0.00015450221758305568, -0.00016931708263243862, -0.00018306147057029526, -0.00019564848495079334, -0.00020699854661637105, -0.00021703989682279182, -0.00022570905092103135, -0.00023295119972766782, -0.00023872055604618635, -0.00024298064414837828, -0.0002457045303856384, -0.00024687499347216343, -0.00024648463336346556, -0.00024453591804183655, -0.0002410411679129688, -0.0002360224779123837, -0.0002295115778141358, -0.0002215496316249646, -0.00021218697733219069, -0.00020148280865075246, -0.00018950480078148174, -0.00017632868254669463, -0.000162037757608194, -0.00014672237779469896, -0.00013047937186849278, -0.0001134114333428111, -9.562647122038798e-05, -7.723692775799665e-05, -5.835906757029844e-05, -3.9112242567516326e-05, -1.9618137374230055e-05, -6.047888898493356e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$x_12$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$x_12$drag$q": {
            "samples": [0.0, 1.961813737422993e-05, 3.9112242567516353e-05, 5.83590675702984e-05, 7.723692775799656e-05, 9.562647122038802e-05, 0.0001134114333428111, 0.00013047937186849273, 0.00014672237779469904, 0.000162037757608194, 0.0001763286825466946, 0.00018950480078148177, 0.00020148280865075246, 0.00021218697733219069, 0.00022154963162496455, 0.0002295115778141358, 0.0002360224779123837, 0.00024104116791296874, 0.00024453591804183655, 0.0002464846333634655, 0.00024687499347216343, 0.0002457045303856384, 0.00024298064414837836, 0.0002387205560461864, 0.0002329511997276678, 0.0002257090509210314, 0.00021703989682279182, 0.000206998546616371, 0.00019564848495079342, 0.00018306147057029523, 0.00016931708263243875, 0.00015450221758305573, 0.00013871053976893965, 0.00012204188926168753, 0.00010460165063662163, 8.650008669757402e-05, 6.78516413599388e-05, 4.877421609939171e-05, 2.9388424540800346e-05, 9.816829900053113e-06, -9.816829900053054e-06, -2.9388424540800176e-05, -4.8774216099391756e-05, -6.785164135993873e-05, -8.650008669757386e-05, -0.00010460165063662167, -0.00012204188926168749, -0.00013871053976893967, -0.00015450221758305568, -0.00016931708263243862, -0.00018306147057029526, -0.00019564848495079334, -0.00020699854661637105, -0.00021703989682279182, -0.00022570905092103135, -0.00023295119972766782, -0.00023872055604618635, -0.00024298064414837828, -0.0002457045303856384, -0.00024687499347216343, -0.00024648463336346556, -0.00024453591804183655, -0.0002410411679129688, -0.0002360224779123837, -0.0002295115778141358, -0.0002215496316249646, -0.00021218697733219069, -0.00020148280865075246, -0.00018950480078148174, -0.00017632868254669463, -0.000162037757608194, -0.00014672237779469896, -0.00013047937186849278, -0.0001134114333428111, -9.562647122038798e-05, -7.723692775799665e-05, -5.835906757029844e-05, -3.9112242567516326e-05, -1.9618137374230055e-05, -6.047888898493356e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$id$drag$i": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$id$drag$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$y$drag_rot$i": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$y$drag_rot$q": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$sy$drag_rot$q": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$-sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$-sy$drag_rot$q": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$sx$drag_rot$i": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$sx$drag_rot$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$-sx$drag_rot$i": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_xy$-sx$drag_rot$q": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit0_rr$readout$rect$rotation$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit0_rr$readout$rect$rotation$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit1_xy$cw$rect$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit1_xy$cw$rect$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit1_xy$x$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$x$drag$q": {
            "samples": [0.0, 2.0014309468777483e-05, 3.990208202907396e-05, 5.9537580779485e-05, 7.879666377485707e-05, 9.755756889163469e-05, 0.00011570168364720961, 0.00013311429510622539, 0.00014968531513268908, 0.00016530997640261188, 0.00017988949477676375, 0.00019333169384580686, 0.00020555158769923368, 0.00021647191823365625, 0.00022602364360341045, 0.00023414637472533384, 0.000240788757077993, 0.0002459087953815025, 0.00024947411910520564, 0.00025146218712459185, 0.000251860430233543, 0.00025066633061090175, 0.00024788743773894754, 0.00024354132067313847, 0.00023765545696488672, 0.00023026705893963085, 0.00022142283842853359, 0.00021117871144124532, 0.0001995994446468871, 0.00018675824589831674, 0.00017273630138851128, 0.0001576222623653155, 0.00014151168464971047, 0.00012450642450114686, 0.00010671399464947767, 8.824688456486483e-05, 6.922184926312981e-05, 4.9759171142952236e-05, 2.998189952181845e-05, 1.0015072678617293e-05, -1.0015072678617232e-05, -2.9981899521818275e-05, -4.975917114295229e-05, -6.922184926312973e-05, -8.824688456486466e-05, -0.00010671399464947772, -0.0001245064245011468, -0.00014151168464971047, -0.00015762226236531543, -0.00017273630138851115, -0.00018675824589831677, -0.00019959944464688703, -0.00021117871144124532, -0.0002214228384285336, -0.0002302670589396308, -0.00023765545696488675, -0.00024354132067313847, -0.0002478874377389475, -0.0002506663306109018, -0.000251860430233543, -0.00025146218712459185, -0.00024947411910520564, -0.0002459087953815025, -0.000240788757077993, -0.00023414637472533384, -0.00022602364360341048, -0.00021647191823365625, -0.00020555158769923368, -0.0001933316938458068, -0.0001798894947767638, -0.00016530997640261188, -0.000149685315132689, -0.00013311429510622547, -0.00011570168364720961, -9.755756889163465e-05, -7.879666377485716e-05, -5.953758077948505e-05, -3.9902082029073934e-05, -2.0014309468777612e-05, -6.170021023822154e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$x_12$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$x_12$drag$q": {
            "samples": [0.0, 2.0014309468777483e-05, 3.990208202907396e-05, 5.9537580779485e-05, 7.879666377485707e-05, 9.755756889163469e-05, 0.00011570168364720961, 0.00013311429510622539, 0.00014968531513268908, 0.00016530997640261188, 0.00017988949477676375, 0.00019333169384580686, 0.00020555158769923368, 0.00021647191823365625, 0.00022602364360341045, 0.00023414637472533384, 0.000240788757077993, 0.0002459087953815025, 0.00024947411910520564, 0.00025146218712459185, 0.000251860430233543, 0.00025066633061090175, 0.00024788743773894754, 0.00024354132067313847, 0.00023765545696488672, 0.00023026705893963085, 0.00022142283842853359, 0.00021117871144124532, 0.0001995994446468871, 0.00018675824589831674, 0.00017273630138851128, 0.0001576222623653155, 0.00014151168464971047, 0.00012450642450114686, 0.00010671399464947767, 8.824688456486483e-05, 6.922184926312981e-05, 4.9759171142952236e-05, 2.998189952181845e-05, 1.0015072678617293e-05, -1.0015072678617232e-05, -2.9981899521818275e-05, -4.975917114295229e-05, -6.922184926312973e-05, -8.824688456486466e-05, -0.00010671399464947772, -0.0001245064245011468, -0.00014151168464971047, -0.00015762226236531543, -0.00017273630138851115, -0.00018675824589831677, -0.00019959944464688703, -0.00021117871144124532, -0.0002214228384285336, -0.0002302670589396308, -0.00023765545696488675, -0.00024354132067313847, -0.0002478874377389475, -0.0002506663306109018, -0.000251860430233543, -0.00025146218712459185, -0.00024947411910520564, -0.0002459087953815025, -0.000240788757077993, -0.00023414637472533384, -0.00022602364360341048, -0.00021647191823365625, -0.00020555158769923368, -0.0001933316938458068, -0.0001798894947767638, -0.00016530997640261188, -0.000149685315132689, -0.00013311429510622547, -0.00011570168364720961, -9.755756889163465e-05, -7.879666377485716e-05, -5.953758077948505e-05, -3.9902082029073934e-05, -2.0014309468777612e-05, -6.170021023822154e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$id$drag$i": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$id$drag$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$y$drag_rot$i": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$y$drag_rot$q": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$sy$drag_rot$q": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$-sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$-sy$drag_rot$q": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$sx$drag_rot$i": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$sx$drag_rot$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$-sx$drag_rot$i": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_xy$-sx$drag_rot$q": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit1_rr$readout$rect$rotation$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit1_rr$readout$rect$rotation$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit2_xy$cw$rect$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit2_xy$cw$rect$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit2_xy$x$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$x$drag$q": {
            "samples": [0.0, 2.2018026023893007e-05, 4.389684699810573e-05, 6.549813797210189e-05, 8.668532862936312e-05, 0.0001073244667289015, 0.00012728506499445596, 0.0001464409260970656, 0.00016467094051521563, 0.00018185985222822547, 0.0001978989874019213, 0.00021268694145960476, 0.00022613022019443675, 0.00023814383086991498, 0.0002486518195713194, 0.0002575877514108164, 0.0002648951305502093, 0.0002705277573858173, 0.00027445002063724563, 0.0002766371224933652, 0.00027707523539205914, 0.0002757615894425223, 0.0002727044899374032, 0.00026792326484406935, 0.00026144814260697474, 0.0002533200610337013, 0.00024359040847295855, 0.0002323206989208979, 0.0002195821831098225, 0.00020545539803811777, 0.00019002965778941667, 0.00017340248886020624, 0.00015567901356591674, 0.00013697128542379688, 0.00011739758071449959, 9.708165070135491e-05, 7.615193923503812e-05, 5.474077069019108e-05, 3.298351336812155e-05, 1.1017723654816893e-05, -1.1017723654816825e-05, -3.298351336812136e-05, -5.474077069019114e-05, -7.615193923503805e-05, -9.708165070135472e-05, -0.00011739758071449965, -0.00013697128542379682, -0.00015567901356591677, -0.00017340248886020616, -0.00019002965778941653, -0.00020545539803811777, -0.00021958218310982243, -0.00023232069892089796, -0.0002435904084729586, -0.00025332006103370117, -0.00026144814260697474, -0.00026792326484406935, -0.0002727044899374032, -0.0002757615894425223, -0.00027707523539205914, -0.0002766371224933652, -0.00027445002063724563, -0.00027052775738581737, -0.0002648951305502093, -0.0002575877514108164, -0.0002486518195713195, -0.00023814383086991498, -0.00022613022019443675, -0.00021268694145960474, -0.00019789898740192136, -0.00018185985222822547, -0.00016467094051521552, -0.0001464409260970657, -0.00012728506499445596, -0.00010732446672890146, -8.668532862936323e-05, -6.549813797210193e-05, -4.38968469981057e-05, -2.2018026023893145e-05, -6.787727734619727e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$x_12$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$x_12$drag$q": {
            "samples": [0.0, 2.2018026023893007e-05, 4.389684699810573e-05, 6.549813797210189e-05, 8.668532862936312e-05, 0.0001073244667289015, 0.00012728506499445596, 0.0001464409260970656, 0.00016467094051521563, 0.00018185985222822547, 0.0001978989874019213, 0.00021268694145960476, 0.00022613022019443675, 0.00023814383086991498, 0.0002486518195713194, 0.0002575877514108164, 0.0002648951305502093, 0.0002705277573858173, 0.00027445002063724563, 0.0002766371224933652, 0.00027707523539205914, 0.0002757615894425223, 0.0002727044899374032, 0.00026792326484406935, 0.00026144814260697474, 0.0002533200610337013, 0.00024359040847295855, 0.0002323206989208979, 0.0002195821831098225, 0.00020545539803811777, 0.00019002965778941667, 0.00017340248886020624, 0.00015567901356591674, 0.00013697128542379688, 0.00011739758071449959, 9.708165070135491e-05, 7.615193923503812e-05, 5.474077069019108e-05, 3.298351336812155e-05, 1.1017723654816893e-05, -1.1017723654816825e-05, -3.298351336812136e-05, -5.474077069019114e-05, -7.615193923503805e-05, -9.708165070135472e-05, -0.00011739758071449965, -0.00013697128542379682, -0.00015567901356591677, -0.00017340248886020616, -0.00019002965778941653, -0.00020545539803811777, -0.00021958218310982243, -0.00023232069892089796, -0.0002435904084729586, -0.00025332006103370117, -0.00026144814260697474, -0.00026792326484406935, -0.0002727044899374032, -0.0002757615894425223, -0.00027707523539205914, -0.0002766371224933652, -0.00027445002063724563, -0.00027052775738581737, -0.0002648951305502093, -0.0002575877514108164, -0.0002486518195713195, -0.00023814383086991498, -0.00022613022019443675, -0.00021268694145960474, -0.00019789898740192136, -0.00018185985222822547, -0.00016467094051521552, -0.0001464409260970657, -0.00012728506499445596, -0.00010732446672890146, -8.668532862936323e-05, -6.549813797210193e-05, -4.38968469981057e-05, -2.2018026023893145e-05, -6.787727734619727e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$id$drag$i": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$id$drag$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$y$drag_rot$i": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$y$drag_rot$q": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$sy$drag_rot$q": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$-sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$-sy$drag_rot$q": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$sx$drag_rot$i": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$sx$drag_rot$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$-sx$drag_rot$i": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_xy$-sx$drag_rot$q": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit2_rr$readout$rect$rotation$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit2_rr$readout$rect$rotation$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit3_xy$cw$rect$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit3_xy$cw$rect$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit3_xy$x$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$x$drag$q": {
            "samples": [0.0, 2.6208307793146006e-05, 5.2250918226118936e-05, 7.79631815308064e-05, 0.00010318253650000317, 0.00012774953825170284, 0.0001515088662909787, 0.00017431030649616114, 0.0001960097008208795, 0.00021646985870764558, 0.0002355614244507299, 0.0002531636950245815, 0.00026916538320721886, 0.000283465321173878, 0.000295973100112574, 0.00030660964181771926, 0.000315307698648, 0.0003220122786876165, 0.00032668099342288067, 0.0003292843257360472, 0.0003298058165220358, 0.0003282421687481927, 0.00032460326829919284, 0.00031891212147529366, 0.000311204709539098, 0.0003015297612304264, 0.00028994844468753543, 0.00027653398072245293, 0.0002613711798954289, 0.0002445559063152682, 0.00022619447155556866, 0.000206402962518723, 0.00018530650749714657, 0.00016303848507192434, 0.00013973968085047704, 0.00011555739737462315, 9.064452282648136e-05, 6.515856442015824e-05, 3.9260652590428426e-05, 1.3114522274245715e-05, -1.3114522274245634e-05, -3.92606525904282e-05, -6.51585644201583e-05, -9.064452282648127e-05, -0.00011555739737462293, -0.00013973968085047712, -0.00016303848507192426, -0.00018530650749714662, -0.0002064029625187229, -0.00022619447155556847, -0.00024455590631526826, -0.0002613711798954288, -0.00027653398072245293, -0.0002899484446875355, -0.0003015297612304264, -0.00031120470953909803, -0.0003189121214752936, -0.0003246032682991928, -0.00032824216874819274, -0.0003298058165220358, -0.00032928432573604724, -0.00032668099342288067, -0.00032201227868761654, -0.000315307698648, -0.00030660964181771926, -0.0002959731001125741, -0.000283465321173878, -0.00026916538320721886, -0.00025316369502458144, -0.0002355614244507299, -0.00021646985870764558, -0.0001960097008208794, -0.00017431030649616124, -0.00015150886629097872, -0.00012774953825170276, -0.0001031825365000033, -7.796318153080646e-05, -5.22509182261189e-05, -2.620830779314617e-05, -8.079509829443551e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$x_12$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$x_12$drag$q": {
            "samples": [0.0, 2.6208307793146006e-05, 5.2250918226118936e-05, 7.79631815308064e-05, 0.00010318253650000317, 0.00012774953825170284, 0.0001515088662909787, 0.00017431030649616114, 0.0001960097008208795, 0.00021646985870764558, 0.0002355614244507299, 0.0002531636950245815, 0.00026916538320721886, 0.000283465321173878, 0.000295973100112574, 0.00030660964181771926, 0.000315307698648, 0.0003220122786876165, 0.00032668099342288067, 0.0003292843257360472, 0.0003298058165220358, 0.0003282421687481927, 0.00032460326829919284, 0.00031891212147529366, 0.000311204709539098, 0.0003015297612304264, 0.00028994844468753543, 0.00027653398072245293, 0.0002613711798954289, 0.0002445559063152682, 0.00022619447155556866, 0.000206402962518723, 0.00018530650749714657, 0.00016303848507192434, 0.00013973968085047704, 0.00011555739737462315, 9.064452282648136e-05, 6.515856442015824e-05, 3.9260652590428426e-05, 1.3114522274245715e-05, -1.3114522274245634e-05, -3.92606525904282e-05, -6.51585644201583e-05, -9.064452282648127e-05, -0.00011555739737462293, -0.00013973968085047712, -0.00016303848507192426, -0.00018530650749714662, -0.0002064029625187229, -0.00022619447155556847, -0.00024455590631526826, -0.0002613711798954288, -0.00027653398072245293, -0.0002899484446875355, -0.0003015297612304264, -0.00031120470953909803, -0.0003189121214752936, -0.0003246032682991928, -0.00032824216874819274, -0.0003298058165220358, -0.00032928432573604724, -0.00032668099342288067, -0.00032201227868761654, -0.000315307698648, -0.00030660964181771926, -0.0002959731001125741, -0.000283465321173878, -0.00026916538320721886, -0.00025316369502458144, -0.0002355614244507299, -0.00021646985870764558, -0.0001960097008208794, -0.00017431030649616124, -0.00015150886629097872, -0.00012774953825170276, -0.0001031825365000033, -7.796318153080646e-05, -5.22509182261189e-05, -2.620830779314617e-05, -8.079509829443551e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$id$drag$i": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$id$drag$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$y$drag_rot$i": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$y$drag_rot$q": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$sy$drag_rot$q": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$-sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$-sy$drag_rot$q": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$sx$drag_rot$i": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$sx$drag_rot$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$-sx$drag_rot$i": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_xy$-sx$drag_rot$q": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit3_rr$readout$rect$rotation$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit3_rr$readout$rect$rotation$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit4_xy$cw$rect$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit4_xy$cw$rect$q": {
            "sample": 0.0,
            "type": "constant",
        },
        "qubit4_xy$x$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$x$drag$q": {
            "samples": [0.0, 2.567880855139495e-05, 5.119526740730637e-05, 7.638805329929739e-05, 0.0001010978893236223, 0.0001251685519410943, 0.00014844785867259994, 0.00017078863024573, 0.00019204962110952743, 0.00021209641243433706, 0.0002308022619509259, 0.00024804890525591624, 0.00026372730351742187, 0.00027773833285365, 0.00028999341102599736, 0.0003004150574844834, 0.00030893738322473375, 0.00031550650735948007, 0.00032008089777087775, 0.0003226316336899265, 0.00032314258854288425, 0.000321610531908658, 0.00031804514994256675, 0.0003124689841373481, 0.0003049172888085843, 0.00029543780820556753, 0.0002840904746567844, 0.0002709470296584383, 0.0002560905703016114, 0.00023961502390570106, 0.000221624554179664, 0.00020223290266551048, 0.0001815626696276548, 0.00015974453893457006, 0.00013691645183329444, 0.00011322273484045564, 8.881318726356264e-05, 6.38421341215533e-05, 3.8467450452336926e-05, 1.2849563175972438e-05, -1.284956317597236e-05, -3.846745045233671e-05, -6.384213412155337e-05, -8.881318726356254e-05, -0.00011322273484045542, -0.00013691645183329452, -0.00015974453893457, -0.00018156266962765486, -0.00020223290266551037, -0.00022162455417966385, -0.00023961502390570106, -0.00025609057030161127, -0.0002709470296584383, -0.00028409047465678444, -0.0002954378082055675, -0.0003049172888085843, -0.00031246898413734805, -0.0003180451499425667, -0.0003216105319086581, -0.00032314258854288425, -0.0003226316336899266, -0.00032008089777087775, -0.0003155065073594801, -0.00030893738322473375, -0.0003004150574844834, -0.00028999341102599736, -0.00027773833285365, -0.00026372730351742187, -0.00024804890525591624, -0.00023080226195092593, -0.00021209641243433706, -0.00019204962110952733, -0.00017078863024573012, -0.00014844785867259994, -0.00012516855194109427, -0.00010109788932362242, -7.638805329929744e-05, -5.1195267407306334e-05, -2.5678808551395105e-05, -7.916275546552178e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$x_12$drag$i": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$x_12$drag$q": {
            "samples": [0.0, 2.567880855139495e-05, 5.119526740730637e-05, 7.638805329929739e-05, 0.0001010978893236223, 0.0001251685519410943, 0.00014844785867259994, 0.00017078863024573, 0.00019204962110952743, 0.00021209641243433706, 0.0002308022619509259, 0.00024804890525591624, 0.00026372730351742187, 0.00027773833285365, 0.00028999341102599736, 0.0003004150574844834, 0.00030893738322473375, 0.00031550650735948007, 0.00032008089777087775, 0.0003226316336899265, 0.00032314258854288425, 0.000321610531908658, 0.00031804514994256675, 0.0003124689841373481, 0.0003049172888085843, 0.00029543780820556753, 0.0002840904746567844, 0.0002709470296584383, 0.0002560905703016114, 0.00023961502390570106, 0.000221624554179664, 0.00020223290266551048, 0.0001815626696276548, 0.00015974453893457006, 0.00013691645183329444, 0.00011322273484045564, 8.881318726356264e-05, 6.38421341215533e-05, 3.8467450452336926e-05, 1.2849563175972438e-05, -1.284956317597236e-05, -3.846745045233671e-05, -6.384213412155337e-05, -8.881318726356254e-05, -0.00011322273484045542, -0.00013691645183329452, -0.00015974453893457, -0.00018156266962765486, -0.00020223290266551037, -0.00022162455417966385, -0.00023961502390570106, -0.00025609057030161127, -0.0002709470296584383, -0.00028409047465678444, -0.0002954378082055675, -0.0003049172888085843, -0.00031246898413734805, -0.0003180451499425667, -0.0003216105319086581, -0.00032314258854288425, -0.0003226316336899266, -0.00032008089777087775, -0.0003155065073594801, -0.00030893738322473375, -0.0003004150574844834, -0.00028999341102599736, -0.00027773833285365, -0.00026372730351742187, -0.00024804890525591624, -0.00023080226195092593, -0.00021209641243433706, -0.00019204962110952733, -0.00017078863024573012, -0.00014844785867259994, -0.00012516855194109427, -0.00010109788932362242, -7.638805329929744e-05, -5.1195267407306334e-05, -2.5678808551395105e-05, -7.916275546552178e-20],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$id$drag$i": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$id$drag$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$y$drag_rot$i": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$y$drag_rot$q": {
            "samples": [0.0, 3.951330642908542e-05, 0.0001578034098709735, 0.0003541224422057662, 0.0006272292113020878, 0.000975397048233357, 0.0013964247238346845, 0.0018876503655823815, 0.0024459682868091014, 0.003067848621854887, 0.003749359643014389, 0.004486192618185318, 0.005273689052060087, 0.006106870138633031, 0.006980468238815008, 0.007888960184143763, 0.008826602196033229, 0.00978746619979106, 0.010765477303815538, 0.011754452207016187, 0.012748138291634124, 0.013740253154304681, 0.014724524325434219, 0.015694728925772488, 0.016644733009458772, 0.017568530344802616, 0.01846028038761439, 0.01931434520700612, 0.02012532513020586, 0.02088809288102744, 0.02159782599616135, 0.022250037314340533, 0.022840603345618576, 0.023365790341400836, 0.023822277900405198, 0.024207179961308063, 0.02451806304935325, 0.024752961661562926, 0.024910390693280243] + [0.024989354827479174] * 2 + [0.024910390693280247, 0.024752961661562926, 0.02451806304935325, 0.024207179961308067, 0.023822277900405198, 0.023365790341400836, 0.022840603345618576, 0.022250037314340537, 0.021597825996161352, 0.02088809288102744, 0.02012532513020587, 0.019314345207006117, 0.01846028038761439, 0.017568530344802626, 0.01664473300945877, 0.01569472892577249, 0.014724524325434233, 0.013740253154304674, 0.012748138291634126, 0.011754452207016198, 0.010765477303815534, 0.009787466199791064, 0.00882660219603323, 0.007888960184143765, 0.006980468238815013, 0.006106870138633031, 0.005273689052060085, 0.004486192618185316, 0.0037493596430143907, 0.003067848621854887, 0.002445968286809099, 0.0018876503655823843, 0.0013964247238346845, 0.0009753970482333556, 0.0006272292113020892, 0.00035412244220576756, 0.0001578034098709721, 3.951330642908542e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$sy$drag_rot$q": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$-sy$drag_rot$i": {
            "samples": [0.0, 1.2097461060526998e-21, 4.8313360198256385e-21, 1.0841872883838379e-20, 1.9203356148820517e-20, 2.986292182541893e-20, 4.2753176707359325e-20, 5.779262445299487e-20, 7.488618083141753e-20, 9.392577487558015e-20, 1.1479103214174587e-19, 1.3735003575547835e-19, 1.6146016043259562e-19, 1.8696897420213735e-19, 2.1371520213036403e-19, 2.4152974595281436e-19, 2.702367531679773e-19, 2.9965472883342577e-19, 3.295976830352794e-19, 3.598763067763229e-19, 3.9029916884843846e-19, 4.2067392612233877e-19, 4.508085396027589e-19, 4.805124885608163e-19, 5.095979750673994e-19, 5.378811113121417e-19, 5.651830822013656e-19, 5.913312758846767e-19, 6.161603750626599e-19, 6.395134021760717e-19, 6.612427118685123e-19, 6.812109244479076e-19, 6.992917944451529e-19, 7.153710087786169e-19, 7.293469094782489e-19, 7.411311363999968e-19, 7.506491858670863e-19, 7.578408817062547e-19, 7.626607557008917e-19] + [7.650783350557458e-19] * 2 + [7.626607557008918e-19, 7.578408817062547e-19, 7.506491858670863e-19, 7.411311363999969e-19, 7.293469094782489e-19, 7.153710087786169e-19, 6.992917944451529e-19, 6.812109244479077e-19, 6.612427118685124e-19, 6.395134021760717e-19, 6.161603750626602e-19, 5.913312758846766e-19, 5.651830822013656e-19, 5.37881111312142e-19, 5.095979750673993e-19, 4.805124885608164e-19, 4.508085396027593e-19, 4.206739261223386e-19, 3.902991688484385e-19, 3.5987630677632323e-19, 3.295976830352793e-19, 2.9965472883342586e-19, 2.7023675316797736e-19, 2.415297459528144e-19, 2.1371520213036418e-19, 1.8696897420213735e-19, 1.6146016043259557e-19, 1.3735003575547828e-19, 1.1479103214174592e-19, 9.392577487558015e-20, 7.488618083141745e-20, 5.779262445299495e-20, 4.2753176707359325e-20, 2.9862921825418887e-20, 1.920335614882056e-20, 1.084187288383842e-20, 4.8313360198255956e-21, 1.2097461060526998e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$-sy$drag_rot$q": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$sx$drag_rot$i": {
            "samples": [0.0, 1.975665321454271e-05, 7.890170493548676e-05, 0.0001770612211028831, 0.0003136146056510439, 0.0004876985241166785, 0.0006982123619173423, 0.0009438251827911907, 0.0012229841434045507, 0.0015339243109274434, 0.0018746798215071945, 0.002243096309092659, 0.0026368445260300435, 0.0030534350693165154, 0.003490234119407504, 0.003944480092071882, 0.0044133010980166145, 0.00489373309989553, 0.005382738651907769, 0.005877226103508094, 0.006374069145817062, 0.0068701265771523405, 0.007362262162717109, 0.007847364462886244, 0.008322366504729386, 0.008784265172401308, 0.009230140193807196, 0.00965717260350306, 0.01006266256510293, 0.01044404644051372, 0.010798912998080674, 0.011125018657170267, 0.011420301672809288, 0.011682895170700418, 0.011911138950202599, 0.012103589980654032, 0.012259031524676625, 0.012376480830781463, 0.012455195346640122] + [0.012494677413739587] * 2 + [0.012455195346640123, 0.012376480830781463, 0.012259031524676625, 0.012103589980654033, 0.011911138950202599, 0.011682895170700418, 0.011420301672809288, 0.011125018657170268, 0.010798912998080676, 0.01044404644051372, 0.010062662565102935, 0.009657172603503059, 0.009230140193807196, 0.008784265172401313, 0.008322366504729384, 0.007847364462886246, 0.007362262162717116, 0.006870126577152337, 0.006374069145817063, 0.005877226103508099, 0.005382738651907767, 0.004893733099895532, 0.004413301098016615, 0.0039444800920718824, 0.0034902341194075065, 0.0030534350693165154, 0.0026368445260300426, 0.002243096309092658, 0.0018746798215071954, 0.0015339243109274434, 0.0012229841434045494, 0.0009438251827911921, 0.0006982123619173423, 0.0004876985241166778, 0.0003136146056510446, 0.00017706122110288378, 7.890170493548605e-05, 1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$sx$drag_rot$q": {
            "samples": [0.0] * 80,
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$-sx$drag_rot$i": {
            "samples": [0.0, -1.975665321454271e-05, -7.890170493548676e-05, -0.0001770612211028831, -0.0003136146056510439, -0.0004876985241166785, -0.0006982123619173423, -0.0009438251827911907, -0.0012229841434045507, -0.0015339243109274434, -0.0018746798215071945, -0.002243096309092659, -0.0026368445260300435, -0.0030534350693165154, -0.003490234119407504, -0.003944480092071882, -0.0044133010980166145, -0.00489373309989553, -0.005382738651907769, -0.005877226103508094, -0.006374069145817062, -0.0068701265771523405, -0.007362262162717109, -0.007847364462886244, -0.008322366504729386, -0.008784265172401308, -0.009230140193807196, -0.00965717260350306, -0.01006266256510293, -0.01044404644051372, -0.010798912998080674, -0.011125018657170267, -0.011420301672809288, -0.011682895170700418, -0.011911138950202599, -0.012103589980654032, -0.012259031524676625, -0.012376480830781463, -0.012455195346640122] + [-0.012494677413739587] * 2 + [-0.012455195346640123, -0.012376480830781463, -0.012259031524676625, -0.012103589980654033, -0.011911138950202599, -0.011682895170700418, -0.011420301672809288, -0.011125018657170268, -0.010798912998080676, -0.01044404644051372, -0.010062662565102935, -0.009657172603503059, -0.009230140193807196, -0.008784265172401313, -0.008322366504729384, -0.007847364462886246, -0.007362262162717116, -0.006870126577152337, -0.006374069145817063, -0.005877226103508099, -0.005382738651907767, -0.004893733099895532, -0.004413301098016615, -0.0039444800920718824, -0.0034902341194075065, -0.0030534350693165154, -0.0026368445260300426, -0.002243096309092658, -0.0018746798215071954, -0.0015339243109274434, -0.0012229841434045494, -0.0009438251827911921, -0.0006982123619173423, -0.0004876985241166778, -0.0003136146056510446, -0.00017706122110288378, -7.890170493548605e-05, -1.975665321454271e-05, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_xy$-sx$drag_rot$q": {
            "samples": [0.0, 2.4194922121053997e-21, 9.662672039651277e-21, 2.1683745767676757e-20, 3.8406712297641033e-20, 5.972584365083786e-20, 8.550635341471865e-20, 1.1558524890598974e-19, 1.4977236166283505e-19, 1.878515497511603e-19, 2.2958206428349174e-19, 2.747000715109567e-19, 3.2292032086519123e-19, 3.739379484042747e-19, 4.2743040426072807e-19, 4.830594919056287e-19, 5.404735063359546e-19, 5.993094576668515e-19, 6.591953660705588e-19, 7.197526135526458e-19, 7.805983376968769e-19, 8.413478522446775e-19, 9.016170792055178e-19, 9.610249771216327e-19, 1.0191959501347989e-18, 1.0757622226242835e-18, 1.1303661644027313e-18, 1.1826625517693535e-18, 1.2323207501253197e-18, 1.2790268043521434e-18, 1.3224854237370246e-18, 1.3624218488958152e-18, 1.3985835888903059e-18, 1.4307420175572338e-18, 1.4586938189564978e-18, 1.4822622727999935e-18, 1.5012983717341726e-18, 1.5156817634125093e-18, 1.5253215114017834e-18] + [1.5301566701114915e-18] * 2 + [1.5253215114017836e-18, 1.5156817634125093e-18, 1.5012983717341726e-18, 1.4822622727999937e-18, 1.4586938189564978e-18, 1.4307420175572338e-18, 1.3985835888903059e-18, 1.3624218488958154e-18, 1.3224854237370248e-18, 1.2790268043521434e-18, 1.2323207501253203e-18, 1.1826625517693533e-18, 1.1303661644027313e-18, 1.075762222624284e-18, 1.0191959501347987e-18, 9.610249771216329e-19, 9.016170792055186e-19, 8.413478522446772e-19, 7.80598337696877e-19, 7.197526135526465e-19, 6.591953660705586e-19, 5.993094576668517e-19, 5.404735063359547e-19, 4.830594919056288e-19, 4.2743040426072836e-19, 3.739379484042747e-19, 3.2292032086519114e-19, 2.7470007151095656e-19, 2.2958206428349183e-19, 1.878515497511603e-19, 1.497723616628349e-19, 1.155852489059899e-19, 8.550635341471865e-20, 5.972584365083777e-20, 3.840671229764112e-20, 2.168374576767684e-20, 9.662672039651191e-21, 2.4194922121053997e-21, 0.0],
            "type": "arbitrary",
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "qubit4_rr$readout$rect$rotation$i": {
            "sample": 0.01,
            "type": "constant",
        },
        "qubit4_rr$readout$rect$rotation$q": {
            "sample": 0.0,
            "type": "constant",
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "qubit0_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "qubit0_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit0_rr$readout$rect$rotation$w3": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit1_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "qubit1_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit1_rr$readout$rect$rotation$w3": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit2_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "qubit2_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit2_rr$readout$rect$rotation$w3": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit3_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "qubit3_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit3_rr$readout$rect$rotation$w3": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "qubit4_rr$readout$rect$rotation$w1": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "qubit4_rr$readout$rect$rotation$w2": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "qubit4_rr$readout$rect$rotation$w3": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
    },
    "mixers": {
        "octave_Octave_2": [
            {'intermediate_frequency': 205000000.0, 'lo_frequency': 4945000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 205000000.0, 'lo_frequency': 5575000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
        ],
        "octave_Octave_1": [
            {'intermediate_frequency': 104000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 35000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 118000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 143000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
            {'intermediate_frequency': 245000000.0, 'lo_frequency': 7350000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
        ],
        "octave_Octave_3": [{'intermediate_frequency': 205000000.0, 'lo_frequency': 5049000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]}],
        "octave_Octave_4": [{'intermediate_frequency': 205000000.0, 'lo_frequency': 6675000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]}],
        "octave_Octave_5": [{'intermediate_frequency': 205000000.0, 'lo_frequency': 6536000000.0, 'correction': [1.0, 0.0, 0.0, 1.0]}],
    },
}