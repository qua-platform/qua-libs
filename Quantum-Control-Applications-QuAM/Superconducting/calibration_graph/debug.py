
# Single QUA script generated at 2025-02-23 00:36:42.311621
# QUA library version: 1.2.1

from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    v2 = declare(int, )
    v3 = declare(fixed, )
    a1 = declare(fixed, size=4)
    a2 = declare(fixed, size=4)
    a3 = declare(fixed, size=4)
    a4 = declare(fixed, size=4)
    a5 = declare(fixed, size=4)
    a6 = declare(fixed, size=4)
    a7 = declare(fixed, size=4)
    a8 = declare(fixed, size=4)
    a9 = declare(fixed, size=4)
    a10 = declare(fixed, size=4)
    a11 = declare(fixed, size=4)
    a12 = declare(fixed, size=4)
    a13 = declare(fixed, size=4)
    a14 = declare(fixed, size=4)
    a15 = declare(fixed, size=4)
    a16 = declare(fixed, size=4)
    a17 = declare(fixed, size=4)
    a18 = declare(fixed, size=4)
    a19 = declare(fixed, size=4)
    a20 = declare(fixed, size=4)
    a21 = declare(fixed, size=4)
    a22 = declare(fixed, size=4)
    a23 = declare(fixed, size=4)
    a24 = declare(fixed, size=4)
    a25 = declare(fixed, size=4)
    a26 = declare(fixed, size=4)
    a27 = declare(fixed, size=4)
    a28 = declare(fixed, size=4)
    a29 = declare(fixed, size=4)
    a30 = declare(fixed, size=4)
    a31 = declare(fixed, size=4)
    a32 = declare(fixed, size=4)
    a33 = declare(fixed, size=4)
    a34 = declare(fixed, size=4)
    a35 = declare(fixed, size=4)
    a36 = declare(fixed, size=4)
    v4 = declare(int, )
    v5 = declare(int, )
    v6 = declare(int, )
    set_dc_offset("q1.z", "single", 0.2901956009099971)
    set_dc_offset("q2.z", "single", 0.22741533285274984)
    set_dc_offset("q3.z", "single", 0.20207247849218538)
    set_dc_offset("q4.z", "single", 0.2040483262013868)
    set_dc_offset("q5.z", "single", 0.1539461527562185)
    set_dc_offset("coupler_q1_q2", "single", 0.2)
    set_dc_offset("coupler_q2_q3", "single", 0.2)
    set_dc_offset("coupler_q3_q4", "single", 0.2)
    set_dc_offset("coupler_q4_q5", "single", 0.12)
    wait(1000, "q1.z")
    align("q1.xy", "q1.z", "q1.resonator")
    with for_(v1,0,(v1<4),(v1+1)):
        r13 = declare_stream()
        save(v1, r13)
        with for_(v2,-2500000,(v2<=2300000),(v2+200000)):
            with for_(v3,0.5,(v3<2.043214285714286),(v3+0.10642857142857143)):
                update_frequency("q1.resonator", (-16482781.0+v2), "Hz", False)
                update_frequency("q2.resonator", (74264448.0+v2), "Hz", False)
                update_frequency("q3.resonator", (-83726496.0+v2), "Hz", False)
                update_frequency("q4.resonator", (129555618.0+v2), "Hz", False)
                update_frequency("q5.resonator", (19883495.0+v2), "Hz", False)
                wait(111475, )
                align()
                measure("readout"*amp(v3), "q1.resonator", None, demod.accumulated("iw1", a1, 65, "out1"), demod.accumulated("iw2", a2, 65, "out2"), demod.accumulated("iw3", a3, 65, "out1"), demod.accumulated("iw1", a4, 65, "out2"))
                measure("readout"*amp(v3), "q2.resonator", None, demod.accumulated("iw1", a13, 65, "out1"), demod.accumulated("iw2", a14, 65, "out2"), demod.accumulated("iw3", a15, 65, "out1"), demod.accumulated("iw1", a16, 65, "out2"))
                measure("readout"*amp(v3), "q3.resonator", None, demod.accumulated("iw1", a25, 65, "out1"), demod.accumulated("iw2", a26, 65, "out2"), demod.accumulated("iw3", a27, 65, "out1"), demod.accumulated("iw1", a28, 65, "out2"))
                play("readout"*amp(v3), "q4.resonator")
                play("readout"*amp(v3), "q5.resonator")
                align()
                wait(111475, )
                play("x180", "q1.xy")
                play("x180", "q2.xy")
                play("x180", "q3.xy")
                play("x180", "q4.xy")
                play("x180", "q5.xy")
                align()
                measure("readout"*amp(v3), "q1.resonator", None, demod.accumulated("iw1", a7, 65, "out1"), demod.accumulated("iw2", a8, 65, "out2"), demod.accumulated("iw3", a9, 65, "out1"), demod.accumulated("iw1", a10, 65, "out2"))
                measure("readout"*amp(v3), "q2.resonator", None, demod.accumulated("iw1", a19, 65, "out1"), demod.accumulated("iw2", a20, 65, "out2"), demod.accumulated("iw3", a21, 65, "out1"), demod.accumulated("iw1", a22, 65, "out2"))
                measure("readout"*amp(v3), "q3.resonator", None, demod.accumulated("iw1", a31, 65, "out1"), demod.accumulated("iw2", a32, 65, "out2"), demod.accumulated("iw3", a33, 65, "out1"), demod.accumulated("iw1", a34, 65, "out2"))
                play("readout"*amp(v3), "q4.resonator")
                play("readout"*amp(v3), "q5.resonator")
                with for_(v4,0,(v4<4),(v4+1)):
                    assign(a5[v4], (a1[v4]+a2[v4]))
                    r1 = declare_stream()
                    save(a5[v4], r1)
                    assign(a6[v4], (a3[v4]+a4[v4]))
                    r2 = declare_stream()
                    save(a6[v4], r2)
                    assign(a11[v4], (a7[v4]+a8[v4]))
                    r3 = declare_stream()
                    save(a11[v4], r3)
                    assign(a12[v4], (a9[v4]+a10[v4]))
                    r4 = declare_stream()
                    save(a12[v4], r4)
                with for_(v5,0,(v5<4),(v5+1)):
                    assign(a17[v5], (a13[v5]+a14[v5]))
                    r5 = declare_stream()
                    save(a17[v5], r5)
                    assign(a18[v5], (a15[v5]+a16[v5]))
                    r6 = declare_stream()
                    save(a18[v5], r6)
                    assign(a23[v5], (a19[v5]+a20[v5]))
                    r7 = declare_stream()
                    save(a23[v5], r7)
                    assign(a24[v5], (a21[v5]+a22[v5]))
                    r8 = declare_stream()
                    save(a24[v5], r8)
                with for_(v6,0,(v6<4),(v6+1)):
                    assign(a29[v6], (a25[v6]+a26[v6]))
                    r9 = declare_stream()
                    save(a29[v6], r9)
                    assign(a30[v6], (a27[v6]+a28[v6]))
                    r10 = declare_stream()
                    save(a30[v6], r10)
                    assign(a35[v6], (a31[v6]+a32[v6]))
                    r11 = declare_stream()
                    save(a35[v6], r11)
                    assign(a36[v6], (a33[v6]+a34[v6]))
                    r12 = declare_stream()
                    save(a36[v6], r12)
    with stream_processing():
        r13.save("n")
        r1.buffer(4).buffer(15).buffer(25).buffer(4).save("I_g1")
        r2.buffer(4).buffer(15).buffer(25).buffer(4).save("Q_g1")
        r3.buffer(4).buffer(15).buffer(25).buffer(4).save("I_e1")
        r4.buffer(4).buffer(15).buffer(25).buffer(4).save("Q_e1")
        r5.buffer(4).buffer(15).buffer(25).buffer(4).save("I_g2")
        r6.buffer(4).buffer(15).buffer(25).buffer(4).save("Q_g2")
        r7.buffer(4).buffer(15).buffer(25).buffer(4).save("I_e2")
        r8.buffer(4).buffer(15).buffer(25).buffer(4).save("Q_e2")
        r9.buffer(4).buffer(15).buffer(25).buffer(4).save("I_g3")
        r10.buffer(4).buffer(15).buffer(25).buffer(4).save("Q_g3")
        r11.buffer(4).buffer(15).buffer(25).buffer(4).save("I_e3")
        r12.buffer(4).buffer(15).buffer(25).buffer(4).save("Q_e3")


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "fems": {
                "5": {
                    "type": "LF",
                    "analog_outputs": {
                        "7": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "1": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "2": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "3": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "4": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "5": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "6": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                        "8": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                    },
                },
                "3": {
                    "type": "LF",
                    "analog_outputs": {
                        "8": {
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                            "output_mode": "amplified",
                            "offset": 0.0,
                        },
                    },
                },
                "1": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "band": 2,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": -14,
                            "upconverter_frequency": 5950000000,
                        },
                        "2": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 4900000000.0,
                        },
                        "3": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 4900000000.0,
                        },
                        "4": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 5000000000.0,
                        },
                        "5": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 5000000000.0,
                        },
                        "6": {
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "upconverter_frequency": 4900000000.0,
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 2,
                            "downconverter_frequency": 5950000000,
                            "sampling_rate": 1000000000.0,
                            "shareable": False,
                        },
                    },
                },
            },
        },
    },
    "elements": {
        "q1.xy": {
            "operations": {
                "x180_DragCosine": "q1.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q1.xy.-y90_DragCosine.pulse",
                "x180_Square": "q1.xy.x180_Square.pulse",
                "x90_Square": "q1.xy.x90_Square.pulse",
                "-x90_Square": "q1.xy.-x90_Square.pulse",
                "y180_Square": "q1.xy.y180_Square.pulse",
                "y90_Square": "q1.xy.y90_Square.pulse",
                "-y90_Square": "q1.xy.-y90_Square.pulse",
                "x180": "q1.xy.x180_DragCosine.pulse",
                "x90": "q1.xy.x90_DragCosine.pulse",
                "-x90": "q1.xy.-x90_DragCosine.pulse",
                "y180": "q1.xy.y180_DragCosine.pulse",
                "y90": "q1.xy.y90_DragCosine.pulse",
                "-y90": "q1.xy.-y90_DragCosine.pulse",
                "saturation": "q1.xy.saturation.pulse",
                "EF_x180": "q1.xy.EF_x180.pulse",
                "EF_x90": "q1.xy.EF_x90.pulse",
            },
            "intermediate_frequency": 213082937.51994482,
            "thread": "a",
            "MWInput": {
                "port": ('con1', 1, 2),
                "upconverter": 1,
            },
        },
        "q1.resonator": {
            "operations": {
                "readout": "q1.resonator.readout.pulse",
                "const": "q1.resonator.const.pulse",
            },
            "intermediate_frequency": -16482781.0,
            "thread": "a",
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
        },
        "q1.z": {
            "operations": {
                "const": "q1.z.const.pulse",
                "flux_pulse": "q1.z.flux_pulse.pulse",
                "cz1_2": "q1.z.cz1_2.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 7),
            },
        },
        "q2.xy": {
            "operations": {
                "x180_DragCosine": "q2.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q2.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q2.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q2.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q2.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q2.xy.-y90_DragCosine.pulse",
                "x180_Square": "q2.xy.x180_Square.pulse",
                "x90_Square": "q2.xy.x90_Square.pulse",
                "-x90_Square": "q2.xy.-x90_Square.pulse",
                "y180_Square": "q2.xy.y180_Square.pulse",
                "y90_Square": "q2.xy.y90_Square.pulse",
                "-y90_Square": "q2.xy.-y90_Square.pulse",
                "x180": "q2.xy.x180_DragCosine.pulse",
                "x90": "q2.xy.x90_DragCosine.pulse",
                "-x90": "q2.xy.-x90_DragCosine.pulse",
                "y180": "q2.xy.y180_DragCosine.pulse",
                "y90": "q2.xy.y90_DragCosine.pulse",
                "-y90": "q2.xy.-y90_DragCosine.pulse",
                "saturation": "q2.xy.saturation.pulse",
                "EF_x180": "q2.xy.EF_x180.pulse",
                "EF_x90": "q2.xy.EF_x90.pulse",
            },
            "intermediate_frequency": -62048681.59228143,
            "thread": "b",
            "MWInput": {
                "port": ('con1', 1, 3),
                "upconverter": 1,
            },
        },
        "q2.resonator": {
            "operations": {
                "readout": "q2.resonator.readout.pulse",
                "const": "q2.resonator.const.pulse",
            },
            "intermediate_frequency": 74264448.0,
            "thread": "b",
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 376,
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
        },
        "q2.z": {
            "operations": {
                "const": "q2.z.const.pulse",
                "flux_pulse": "q2.z.flux_pulse.pulse",
                "cz": "q2.z.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 1),
            },
        },
        "q3.xy": {
            "operations": {
                "x180_DragCosine": "q3.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q3.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q3.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q3.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q3.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q3.xy.-y90_DragCosine.pulse",
                "x180_Square": "q3.xy.x180_Square.pulse",
                "x90_Square": "q3.xy.x90_Square.pulse",
                "-x90_Square": "q3.xy.-x90_Square.pulse",
                "y180_Square": "q3.xy.y180_Square.pulse",
                "y90_Square": "q3.xy.y90_Square.pulse",
                "-y90_Square": "q3.xy.-y90_Square.pulse",
                "x180": "q3.xy.x180_DragCosine.pulse",
                "x90": "q3.xy.x90_DragCosine.pulse",
                "-x90": "q3.xy.-x90_DragCosine.pulse",
                "y180": "q3.xy.y180_DragCosine.pulse",
                "y90": "q3.xy.y90_DragCosine.pulse",
                "-y90": "q3.xy.-y90_DragCosine.pulse",
                "saturation": "q3.xy.saturation.pulse",
                "EF_x180": "q3.xy.EF_x180.pulse",
                "EF_x90": "q3.xy.EF_x90.pulse",
            },
            "intermediate_frequency": 145290051.2800316,
            "thread": "c",
            "MWInput": {
                "port": ('con1', 1, 4),
                "upconverter": 1,
            },
        },
        "q3.resonator": {
            "operations": {
                "readout": "q3.resonator.readout.pulse",
                "const": "q3.resonator.const.pulse",
            },
            "intermediate_frequency": -83726496.0,
            "thread": "c",
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
        },
        "q3.z": {
            "operations": {
                "const": "q3.z.const.pulse",
                "flux_pulse": "q3.z.flux_pulse.pulse",
                "cz3_2": "q3.z.cz3_2.pulse",
                "cz3_4": "q3.z.cz3_4.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 2),
            },
        },
        "q4.xy": {
            "operations": {
                "x180_DragCosine": "q4.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q4.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q4.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q4.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q4.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q4.xy.-y90_DragCosine.pulse",
                "x180_Square": "q4.xy.x180_Square.pulse",
                "x90_Square": "q4.xy.x90_Square.pulse",
                "-x90_Square": "q4.xy.-x90_Square.pulse",
                "y180_Square": "q4.xy.y180_Square.pulse",
                "y90_Square": "q4.xy.y90_Square.pulse",
                "-y90_Square": "q4.xy.-y90_Square.pulse",
                "x180": "q4.xy.x180_DragCosine.pulse",
                "x90": "q4.xy.x90_DragCosine.pulse",
                "-x90": "q4.xy.-x90_DragCosine.pulse",
                "y180": "q4.xy.y180_DragCosine.pulse",
                "y90": "q4.xy.y90_DragCosine.pulse",
                "-y90": "q4.xy.-y90_DragCosine.pulse",
                "saturation": "q4.xy.saturation.pulse",
                "EF_x180": "q4.xy.EF_x180.pulse",
                "EF_x90": "q4.xy.EF_x90.pulse",
            },
            "intermediate_frequency": -323672230.4042064,
            "thread": "d",
            "MWInput": {
                "port": ('con1', 1, 5),
                "upconverter": 1,
            },
        },
        "q4.resonator": {
            "operations": {
                "readout": "q4.resonator.readout.pulse",
                "const": "q4.resonator.const.pulse",
            },
            "intermediate_frequency": 129555618.0,
            "thread": "d",
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
        },
        "q4.z": {
            "operations": {
                "const": "q4.z.const.pulse",
                "flux_pulse": "q4.z.flux_pulse.pulse",
                "cz": "q4.z.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 3),
            },
        },
        "q5.xy": {
            "operations": {
                "x180_DragCosine": "q5.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q5.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q5.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q5.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q5.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q5.xy.-y90_DragCosine.pulse",
                "x180_Square": "q5.xy.x180_Square.pulse",
                "x90_Square": "q5.xy.x90_Square.pulse",
                "-x90_Square": "q5.xy.-x90_Square.pulse",
                "y180_Square": "q5.xy.y180_Square.pulse",
                "y90_Square": "q5.xy.y90_Square.pulse",
                "-y90_Square": "q5.xy.-y90_Square.pulse",
                "x180": "q5.xy.x180_DragCosine.pulse",
                "x90": "q5.xy.x90_DragCosine.pulse",
                "-x90": "q5.xy.-x90_DragCosine.pulse",
                "y180": "q5.xy.y180_DragCosine.pulse",
                "y90": "q5.xy.y90_DragCosine.pulse",
                "-y90": "q5.xy.-y90_DragCosine.pulse",
                "saturation": "q5.xy.saturation.pulse",
                "EF_x180": "q5.xy.EF_x180.pulse",
                "EF_x90": "q5.xy.EF_x90.pulse",
            },
            "intermediate_frequency": -14015480.963453503,
            "thread": "e",
            "MWInput": {
                "port": ('con1', 1, 6),
                "upconverter": 1,
            },
        },
        "q5.resonator": {
            "operations": {
                "readout": "q5.resonator.readout.pulse",
                "const": "q5.resonator.const.pulse",
            },
            "intermediate_frequency": 19883495.0,
            "thread": "e",
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
        },
        "q5.z": {
            "operations": {
                "const": "q5.z.const.pulse",
                "flux_pulse": "q5.z.flux_pulse.pulse",
                "cz5_4": "q5.z.cz5_4.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 4),
            },
        },
        "coupler_q1_q2": {
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
                "flux_pulse": "coupler_q1_q2.flux_pulse.pulse",
                "cz": "coupler_q1_q2.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 5),
            },
        },
        "coupler_q2_q3": {
            "operations": {
                "const": "coupler_q2_q3.const.pulse",
                "flux_pulse": "coupler_q2_q3.flux_pulse.pulse",
                "cz": "coupler_q2_q3.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 6),
            },
        },
        "coupler_q3_q4": {
            "operations": {
                "const": "coupler_q3_q4.const.pulse",
                "flux_pulse": "coupler_q3_q4.flux_pulse.pulse",
                "cz": "coupler_q3_q4.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 3, 8),
            },
        },
        "coupler_q4_q5": {
            "operations": {
                "const": "coupler_q4_q5.const.pulse",
                "flux_pulse": "coupler_q4_q5.flux_pulse.pulse",
                "cz": "coupler_q4_q5.cz.pulse",
            },
            "singleInput": {
                "port": ('con1', 5, 8),
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
        "q1.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x180_DragCosine.wf.I",
                "Q": "q1.xy.x180_DragCosine.wf.Q",
            },
        },
        "q1.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x90_DragCosine.wf.I",
                "Q": "q1.xy.x90_DragCosine.wf.Q",
            },
        },
        "q1.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-x90_DragCosine.wf.I",
                "Q": "q1.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q1.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y180_DragCosine.wf.I",
                "Q": "q1.xy.y180_DragCosine.wf.Q",
            },
        },
        "q1.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y90_DragCosine.wf.I",
                "Q": "q1.xy.y90_DragCosine.wf.Q",
            },
        },
        "q1.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-y90_DragCosine.wf.I",
                "Q": "q1.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q1.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x180_Square.wf.I",
                "Q": "q1.xy.x180_Square.wf.Q",
            },
        },
        "q1.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.x90_Square.wf.I",
                "Q": "q1.xy.x90_Square.wf.Q",
            },
        },
        "q1.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-x90_Square.wf.I",
                "Q": "q1.xy.-x90_Square.wf.Q",
            },
        },
        "q1.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y180_Square.wf.I",
                "Q": "q1.xy.y180_Square.wf.Q",
            },
        },
        "q1.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.y90_Square.wf.I",
                "Q": "q1.xy.y90_Square.wf.Q",
            },
        },
        "q1.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.-y90_Square.wf.I",
                "Q": "q1.xy.-y90_Square.wf.Q",
            },
        },
        "q1.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.saturation.wf.I",
                "Q": "q1.xy.saturation.wf.Q",
            },
        },
        "q1.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.EF_x180.wf.I",
                "Q": "q1.xy.EF_x180.wf.Q",
            },
        },
        "q1.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.xy.EF_x90.wf.I",
                "Q": "q1.xy.EF_x90.wf.Q",
            },
        },
        "q1.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q1.resonator.readout.wf.I",
                "Q": "q1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q1.resonator.readout.iw1",
                "iw2": "q1.resonator.readout.iw2",
                "iw3": "q1.resonator.readout.iw3",
            },
        },
        "q1.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q1.resonator.const.wf.I",
                "Q": "q1.resonator.const.wf.Q",
            },
        },
        "q1.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q1.z.const.wf",
            },
        },
        "q1.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q1.z.flux_pulse.wf",
            },
        },
        "q1.z.cz1_2.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "q1.z.cz1_2.wf",
            },
        },
        "q2.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x180_DragCosine.wf.I",
                "Q": "q2.xy.x180_DragCosine.wf.Q",
            },
        },
        "q2.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x90_DragCosine.wf.I",
                "Q": "q2.xy.x90_DragCosine.wf.Q",
            },
        },
        "q2.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-x90_DragCosine.wf.I",
                "Q": "q2.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q2.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y180_DragCosine.wf.I",
                "Q": "q2.xy.y180_DragCosine.wf.Q",
            },
        },
        "q2.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y90_DragCosine.wf.I",
                "Q": "q2.xy.y90_DragCosine.wf.Q",
            },
        },
        "q2.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-y90_DragCosine.wf.I",
                "Q": "q2.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q2.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x180_Square.wf.I",
                "Q": "q2.xy.x180_Square.wf.Q",
            },
        },
        "q2.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.x90_Square.wf.I",
                "Q": "q2.xy.x90_Square.wf.Q",
            },
        },
        "q2.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-x90_Square.wf.I",
                "Q": "q2.xy.-x90_Square.wf.Q",
            },
        },
        "q2.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y180_Square.wf.I",
                "Q": "q2.xy.y180_Square.wf.Q",
            },
        },
        "q2.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.y90_Square.wf.I",
                "Q": "q2.xy.y90_Square.wf.Q",
            },
        },
        "q2.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.-y90_Square.wf.I",
                "Q": "q2.xy.-y90_Square.wf.Q",
            },
        },
        "q2.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.saturation.wf.I",
                "Q": "q2.xy.saturation.wf.Q",
            },
        },
        "q2.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.EF_x180.wf.I",
                "Q": "q2.xy.EF_x180.wf.Q",
            },
        },
        "q2.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.xy.EF_x90.wf.I",
                "Q": "q2.xy.EF_x90.wf.Q",
            },
        },
        "q2.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q2.resonator.readout.wf.I",
                "Q": "q2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q2.resonator.readout.iw1",
                "iw2": "q2.resonator.readout.iw2",
                "iw3": "q2.resonator.readout.iw3",
            },
        },
        "q2.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q2.resonator.const.wf.I",
                "Q": "q2.resonator.const.wf.Q",
            },
        },
        "q2.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q2.z.const.wf",
            },
        },
        "q2.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q2.z.flux_pulse.wf",
            },
        },
        "q2.z.cz.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "single": "q2.z.cz.wf",
            },
        },
        "q3.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x180_DragCosine.wf.I",
                "Q": "q3.xy.x180_DragCosine.wf.Q",
            },
        },
        "q3.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x90_DragCosine.wf.I",
                "Q": "q3.xy.x90_DragCosine.wf.Q",
            },
        },
        "q3.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-x90_DragCosine.wf.I",
                "Q": "q3.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q3.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y180_DragCosine.wf.I",
                "Q": "q3.xy.y180_DragCosine.wf.Q",
            },
        },
        "q3.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y90_DragCosine.wf.I",
                "Q": "q3.xy.y90_DragCosine.wf.Q",
            },
        },
        "q3.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-y90_DragCosine.wf.I",
                "Q": "q3.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q3.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x180_Square.wf.I",
                "Q": "q3.xy.x180_Square.wf.Q",
            },
        },
        "q3.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.x90_Square.wf.I",
                "Q": "q3.xy.x90_Square.wf.Q",
            },
        },
        "q3.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-x90_Square.wf.I",
                "Q": "q3.xy.-x90_Square.wf.Q",
            },
        },
        "q3.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y180_Square.wf.I",
                "Q": "q3.xy.y180_Square.wf.Q",
            },
        },
        "q3.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.y90_Square.wf.I",
                "Q": "q3.xy.y90_Square.wf.Q",
            },
        },
        "q3.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.-y90_Square.wf.I",
                "Q": "q3.xy.-y90_Square.wf.Q",
            },
        },
        "q3.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.saturation.wf.I",
                "Q": "q3.xy.saturation.wf.Q",
            },
        },
        "q3.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.EF_x180.wf.I",
                "Q": "q3.xy.EF_x180.wf.Q",
            },
        },
        "q3.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.xy.EF_x90.wf.I",
                "Q": "q3.xy.EF_x90.wf.Q",
            },
        },
        "q3.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q3.resonator.readout.wf.I",
                "Q": "q3.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q3.resonator.readout.iw1",
                "iw2": "q3.resonator.readout.iw2",
                "iw3": "q3.resonator.readout.iw3",
            },
        },
        "q3.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q3.resonator.const.wf.I",
                "Q": "q3.resonator.const.wf.Q",
            },
        },
        "q3.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q3.z.const.wf",
            },
        },
        "q3.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q3.z.flux_pulse.wf",
            },
        },
        "q3.z.cz3_2.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q3.z.cz3_2.wf",
            },
        },
        "q3.z.cz3_4.pulse": {
            "operation": "control",
            "length": 60,
            "waveforms": {
                "single": "q3.z.cz3_4.wf",
            },
        },
        "q4.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x180_DragCosine.wf.I",
                "Q": "q4.xy.x180_DragCosine.wf.Q",
            },
        },
        "q4.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x90_DragCosine.wf.I",
                "Q": "q4.xy.x90_DragCosine.wf.Q",
            },
        },
        "q4.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-x90_DragCosine.wf.I",
                "Q": "q4.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q4.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y180_DragCosine.wf.I",
                "Q": "q4.xy.y180_DragCosine.wf.Q",
            },
        },
        "q4.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y90_DragCosine.wf.I",
                "Q": "q4.xy.y90_DragCosine.wf.Q",
            },
        },
        "q4.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-y90_DragCosine.wf.I",
                "Q": "q4.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q4.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x180_Square.wf.I",
                "Q": "q4.xy.x180_Square.wf.Q",
            },
        },
        "q4.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.x90_Square.wf.I",
                "Q": "q4.xy.x90_Square.wf.Q",
            },
        },
        "q4.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-x90_Square.wf.I",
                "Q": "q4.xy.-x90_Square.wf.Q",
            },
        },
        "q4.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y180_Square.wf.I",
                "Q": "q4.xy.y180_Square.wf.Q",
            },
        },
        "q4.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.y90_Square.wf.I",
                "Q": "q4.xy.y90_Square.wf.Q",
            },
        },
        "q4.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.-y90_Square.wf.I",
                "Q": "q4.xy.-y90_Square.wf.Q",
            },
        },
        "q4.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.saturation.wf.I",
                "Q": "q4.xy.saturation.wf.Q",
            },
        },
        "q4.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.EF_x180.wf.I",
                "Q": "q4.xy.EF_x180.wf.Q",
            },
        },
        "q4.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.xy.EF_x90.wf.I",
                "Q": "q4.xy.EF_x90.wf.Q",
            },
        },
        "q4.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q4.resonator.readout.wf.I",
                "Q": "q4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q4.resonator.readout.iw1",
                "iw2": "q4.resonator.readout.iw2",
                "iw3": "q4.resonator.readout.iw3",
            },
        },
        "q4.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q4.resonator.const.wf.I",
                "Q": "q4.resonator.const.wf.Q",
            },
        },
        "q4.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q4.z.const.wf",
            },
        },
        "q4.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q4.z.flux_pulse.wf",
            },
        },
        "q4.z.cz.pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "single": "q4.z.cz.wf",
            },
        },
        "q5.xy.x180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x180_DragCosine.wf.I",
                "Q": "q5.xy.x180_DragCosine.wf.Q",
            },
        },
        "q5.xy.x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x90_DragCosine.wf.I",
                "Q": "q5.xy.x90_DragCosine.wf.Q",
            },
        },
        "q5.xy.-x90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-x90_DragCosine.wf.I",
                "Q": "q5.xy.-x90_DragCosine.wf.Q",
            },
        },
        "q5.xy.y180_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y180_DragCosine.wf.I",
                "Q": "q5.xy.y180_DragCosine.wf.Q",
            },
        },
        "q5.xy.y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y90_DragCosine.wf.I",
                "Q": "q5.xy.y90_DragCosine.wf.Q",
            },
        },
        "q5.xy.-y90_DragCosine.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-y90_DragCosine.wf.I",
                "Q": "q5.xy.-y90_DragCosine.wf.Q",
            },
        },
        "q5.xy.x180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x180_Square.wf.I",
                "Q": "q5.xy.x180_Square.wf.Q",
            },
        },
        "q5.xy.x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.x90_Square.wf.I",
                "Q": "q5.xy.x90_Square.wf.Q",
            },
        },
        "q5.xy.-x90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-x90_Square.wf.I",
                "Q": "q5.xy.-x90_Square.wf.Q",
            },
        },
        "q5.xy.y180_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y180_Square.wf.I",
                "Q": "q5.xy.y180_Square.wf.Q",
            },
        },
        "q5.xy.y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.y90_Square.wf.I",
                "Q": "q5.xy.y90_Square.wf.Q",
            },
        },
        "q5.xy.-y90_Square.pulse": {
            "operation": "control",
            "length": 40,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.-y90_Square.wf.I",
                "Q": "q5.xy.-y90_Square.wf.Q",
            },
        },
        "q5.xy.saturation.pulse": {
            "operation": "control",
            "length": 20000,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.saturation.wf.I",
                "Q": "q5.xy.saturation.wf.Q",
            },
        },
        "q5.xy.EF_x180.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.EF_x180.wf.I",
                "Q": "q5.xy.EF_x180.wf.Q",
            },
        },
        "q5.xy.EF_x90.pulse": {
            "operation": "control",
            "length": 16,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.xy.EF_x90.wf.I",
                "Q": "q5.xy.EF_x90.wf.Q",
            },
        },
        "q5.resonator.readout.pulse": {
            "operation": "measurement",
            "length": 1040,
            "digital_marker": "ON",
            "waveforms": {
                "I": "q5.resonator.readout.wf.I",
                "Q": "q5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q5.resonator.readout.iw1",
                "iw2": "q5.resonator.readout.iw2",
                "iw3": "q5.resonator.readout.iw3",
            },
        },
        "q5.resonator.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "q5.resonator.const.wf.I",
                "Q": "q5.resonator.const.wf.Q",
            },
        },
        "q5.z.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q5.z.const.wf",
            },
        },
        "q5.z.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "q5.z.flux_pulse.wf",
            },
        },
        "q5.z.cz5_4.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "q5.z.cz5_4.wf",
            },
        },
        "coupler_q1_q2.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.const.wf",
            },
        },
        "coupler_q1_q2.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.flux_pulse.wf",
            },
        },
        "coupler_q1_q2.cz.pulse": {
            "operation": "control",
            "length": 80,
            "waveforms": {
                "single": "coupler_q1_q2.cz.wf",
            },
        },
        "coupler_q2_q3.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.const.wf",
            },
        },
        "coupler_q2_q3.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.flux_pulse.wf",
            },
        },
        "coupler_q2_q3.cz.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q2_q3.cz.wf",
            },
        },
        "coupler_q3_q4.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.const.wf",
            },
        },
        "coupler_q3_q4.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.flux_pulse.wf",
            },
        },
        "coupler_q3_q4.cz.pulse": {
            "operation": "control",
            "length": 60,
            "waveforms": {
                "single": "coupler_q3_q4.cz.wf",
            },
        },
        "coupler_q4_q5.const.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.const.wf",
            },
        },
        "coupler_q4_q5.flux_pulse.pulse": {
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.flux_pulse.wf",
            },
        },
        "coupler_q4_q5.cz.pulse": {
            "operation": "control",
            "length": 88,
            "waveforms": {
                "single": "coupler_q4_q5.cz.wf",
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
        "q1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00689333908728961, 0.02638143539694644, 0.0550946200332339, 0.08806812252144451, 0.11960052472650065, 0.14423958784426721] + [0.157724994022821] * 2 + [0.14423958784426721, 0.1196005247265007, 0.08806812252144458, 0.05509462003323392, 0.026381435396946422, 0.006893339087289601, 0.0],
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.010500440545542306, -0.019185259527252734, -0.024552772844090034, -0.025674888681245372, -0.022357583016372237, -0.015174448135703067, -0.005367513316837296, 0.00536751331683729, 0.015174448135703062, 0.02235758301637223, 0.02567488868124537, 0.024552772844090038, 0.01918525952725273, 0.010500440545542306, 2.9252665264350483e-17],
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00343277168258172, 0.013137529320656793, 0.027436232153646715, 0.0438565045620903, 0.05955913227307592, 0.07182898830147982] + [0.07854450307184835] * 2 + [0.07182898830147982, 0.059559132273075945, 0.04385650456209034, 0.027436232153646725, 0.013137529320656784, 0.003432771682581716, 0.0],
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00076108715916614, -0.0013905754342526768, -0.0017796205837757005, -0.0018609531670184527, -0.0016205100414549394, -0.001099866007852313, -0.00038904514952302344, 0.00038904514952302295, 0.0010998660078523125, 0.0016205100414549388, 0.0018609531670184525, 0.0017796205837757005, 0.0013905754342526766, 0.00076108715916614, 2.120275602487371e-18],
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00343277168258172, -0.013137529320656793, -0.027436232153646715, -0.0438565045620903, -0.05955913227307592, -0.07182898830147982] + [-0.07854450307184835] * 2 + [-0.07182898830147982, -0.059559132273075945, -0.04385650456209034, -0.027436232153646725, -0.013137529320656784, -0.003432771682581716, -2.596588729896385e-34],
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0007610871591661405, 0.0013905754342526783, 0.0017796205837757038, 0.0018609531670184581, 0.0016205100414549468, 0.0010998660078523218, 0.00038904514952303303, -0.00038904514952301336, -0.0010998660078523036, -0.0016205100414549314, -0.001860953167018447, -0.0017796205837756973, -0.001390575434252675, -0.0007610871591661396, -2.120275602487371e-18],
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010500440545542306, 0.019185259527252734, 0.024552772844090038, 0.02567488868124538, 0.022357583016372244, 0.015174448135703076, 0.0053675133168373055, -0.00536751331683728, -0.015174448135703053, -0.022357583016372223, -0.025674888681245362, -0.024552772844090034, -0.01918525952725273, -0.010500440545542306, -2.9252665264350483e-17],
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006893339087289609, 0.02638143539694644, 0.0550946200332339, 0.08806812252144451, 0.11960052472650065, 0.14423958784426721] + [0.157724994022821] * 2 + [0.14423958784426721, 0.1196005247265007, 0.08806812252144458, 0.05509462003323392, 0.026381435396946422, 0.006893339087289602, 1.791209144125789e-33],
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0007610871591661403, 0.0013905754342526777, 0.0017796205837757022, 0.0018609531670184553, 0.0016205100414549431, 0.0010998660078523172, 0.00038904514952302826, -0.0003890451495230181, -0.0010998660078523081, -0.001620510041454935, -0.00186095316701845, -0.0017796205837756988, -0.0013905754342526757, -0.0007610871591661398, -2.120275602487371e-18],
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00343277168258172, 0.013137529320656793, 0.027436232153646715, 0.0438565045620903, 0.05955913227307592, 0.07182898830147982] + [0.07854450307184835] * 2 + [0.07182898830147982, 0.059559132273075945, 0.04385650456209034, 0.027436232153646725, 0.013137529320656784, 0.003432771682581716, 1.2982943649481925e-34],
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0007610871591661398, -0.001390575434252676, -0.0017796205837756988, -0.0018609531670184501, -0.0016205100414549358, -0.0010998660078523086, -0.0003890451495230186, 0.0003890451495230278, 0.0010998660078523168, 0.0016205100414549425, 0.001860953167018455, 0.0017796205837757022, 0.0013905754342526775, 0.0007610871591661403, 2.120275602487371e-18],
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00343277168258172, -0.013137529320656793, -0.027436232153646715, -0.0438565045620903, -0.05955913227307592, -0.07182898830147982] + [-0.07854450307184835] * 2 + [-0.07182898830147982, -0.059559132273075945, -0.04385650456209034, -0.027436232153646725, -0.013137529320656784, -0.003432771682581716, 1.2982943649481925e-34],
        },
        "q1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.06583496945367821,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004857743557162809, 0.018591026236402998, 0.03882523869952356, 0.062061701789601555, 0.08428261994172241, 0.10164579453668046] + [0.11114897494751827] * 2 + [0.10164579453668046, 0.08428261994172244, 0.0620617017896016, 0.038825238699523576, 0.018591026236402988, 0.004857743557162803, 0.0],
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.007396079806845666, -0.013513310223812048, -0.017293966538511965, -0.01808433892794943, -0.01574776482568296, -0.010688259121103763, -0.003780656314699915, 0.0037806563146999105, 0.010688259121103758, 0.015747764825682957, 0.018084338927949428, 0.017293966538511965, 0.013513310223812044, 0.007396079806845666, 2.0604378065825656e-17],
        },
        "q1.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004593670635837815, 0.01758039515822729, 0.036714650916558844, 0.0586879512609605, 0.07970091294071842, 0.09612020398463854] + [0.10510677980665072] * 2 + [0.09612020398463854, 0.07970091294071845, 0.05868795126096055, 0.03671465091655885, 0.017580395158227277, 0.004593670635837809, 0.0],
        },
        "q1.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0026711344749756555, -0.0048804055327331276, -0.006245810136788678, -0.006531257426789689, -0.005687390973087767, -0.003860122951814034, -0.0013654046040555497, 0.0013654046040555482, 0.0038601229518140322, 0.005687390973087765, 0.006531257426789689, 0.006245810136788678, 0.004880405532733127, 0.0026711344749756555, 7.441383276599842e-18],
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.15,
        },
        "q1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q1.z.cz1_2.wf": {
            "type": "constant",
            "sample": -0.07009506167631502,
        },
        "q2.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010461145880255703, 0.0400357563617998, 0.08361011261547184, 0.13364981258441283, 0.18150253754759008, 0.21889411663051164] + [0.23935920611790854] * 2 + [0.21889411663051164, 0.18150253754759013, 0.1336498125844129, 0.08361011261547187, 0.04003575636179977, 0.01046114588025569, 0.0],
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012501046533903815, -0.022840548553653223, -0.029230712228606546, -0.03056662020654985, -0.026617281861753765, -0.018065573672646034, -0.00639016367495332, 0.006390163674953312, 0.018065573672646027, 0.02661728186175375, 0.030566620206549846, 0.029230712228606546, 0.02284054855365322, 0.012501046533903815, 3.482605592825352e-17],
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005188727012159342, 0.01985773001012994, 0.041470605111854075, 0.06629028986543736, 0.09002523529722789, 0.1085714537168591] + [0.11872213547247179] * 2 + [0.1085714537168591, 0.09002523529722793, 0.0662902898654374, 0.04147060511185408, 0.019857730010129925, 0.005188727012159336, 0.0],
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0015234359102312821, -0.0027834559116028178, -0.0035621910989551164, -0.0037249910838078847, -0.003243706269788647, -0.0022015551735766023, -0.0007787351873522982, 0.0007787351873522972, 0.0022015551735766014, 0.0032437062697886457, 0.0037249910838078842, 0.003562191098955117, 0.0027834559116028173, 0.0015234359102312821, 4.244065812324945e-18],
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005188727012159342, -0.01985773001012994, -0.041470605111854075, -0.06629028986543736, -0.09002523529722789, -0.1085714537168591] + [-0.11872213547247179] * 2 + [-0.1085714537168591, -0.09002523529722793, -0.0662902898654374, -0.04147060511185408, -0.019857730010129925, -0.005188727012159336, -5.197481612434455e-34],
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015234359102312828, 0.0027834559116028204, 0.0035621910989551216, 0.003724991083807893, 0.003243706269788658, 0.0022015551735766158, 0.0007787351873523127, -0.0007787351873522827, -0.002201555173576588, -0.003243706269788635, -0.003724991083807876, -0.0035621910989551116, -0.0027834559116028147, -0.0015234359102312815, -4.244065812324945e-18],
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.012501046533903815, 0.022840548553653226, 0.02923071222860655, 0.030566620206549856, 0.026617281861753776, 0.018065573672646048, 0.0063901636749533345, -0.006390163674953297, -0.018065573672646013, -0.02661728186175374, -0.03056662020654984, -0.029230712228606542, -0.022840548553653216, -0.012501046533903815, -3.482605592825352e-17],
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010461145880255703, 0.0400357563617998, 0.08361011261547184, 0.13364981258441283, 0.18150253754759008, 0.21889411663051164] + [0.23935920611790854] * 2 + [0.21889411663051164, 0.18150253754759013, 0.1336498125844129, 0.08361011261547187, 0.04003575636179977, 0.01046114588025569, 2.1324808959731188e-33],
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015234359102312823, 0.002783455911602819, 0.003562191098955119, 0.0037249910838078886, 0.0032437062697886526, 0.002201555173576609, 0.0007787351873523055, -0.00077873518735229, -0.002201555173576595, -0.00324370626978864, -0.0037249910838078803, -0.003562191098955114, -0.002783455911602816, -0.001523435910231282, -4.244065812324945e-18],
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005188727012159342, 0.01985773001012994, 0.041470605111854075, 0.06629028986543736, 0.09002523529722789, 0.1085714537168591] + [0.11872213547247179] * 2 + [0.1085714537168591, 0.09002523529722793, 0.0662902898654374, 0.04147060511185408, 0.019857730010129925, 0.005188727012159336, 2.5987408062172275e-34],
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.001523435910231282, -0.0027834559116028165, -0.0035621910989551138, -0.0037249910838078808, -0.0032437062697886414, -0.002201555173576596, -0.0007787351873522909, 0.0007787351873523045, 0.002201555173576608, 0.0032437062697886513, 0.003724991083807888, 0.0035621910989551194, 0.0027834559116028186, 0.0015234359102312823, 4.244065812324945e-18],
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005188727012159342, -0.01985773001012994, -0.041470605111854075, -0.06629028986543736, -0.09002523529722789, -0.1085714537168591] + [-0.11872213547247179] * 2 + [-0.1085714537168591, -0.09002523529722793, -0.0662902898654374, -0.04147060511185408, -0.019857730010129925, -0.005188727012159336, 2.5987408062172275e-34],
        },
        "q2.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q2.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q2.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.09775959661668145,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005563561092243391, 0.02129225411275889, 0.04446644514760913, 0.07107910603595226, 0.09652866594175959, 0.11641466475532354] + [0.1272986326231512] * 2 + [0.11641466475532354, 0.09652866594175963, 0.07107910603595233, 0.04446644514760914, 0.02129225411275887, 0.005563561092243384, 0.0],
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008269983263978223, -0.015110011291175274, -0.019337381095984943, -0.020221142034706617, -0.01760848381231923, -0.01195115877072839, -0.004227369804809668, 0.004227369804809662, 0.011951158770728387, 0.017608483812319223, 0.020221142034706614, 0.019337381095984946, 0.015110011291175271, 0.008269983263978223, 2.303894309135784e-17],
        },
        "q2.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005559686906605402, 0.021277427252099614, 0.044435480939558375, 0.07102961010210704, 0.09646144820746218, 0.11633359940621195] + [0.1272099882232664] * 2 + [0.11633359940621195, 0.09646144820746223, 0.0710296101021071, 0.044435480939558396, 0.0212774272520996, 0.005559686906605395, 0.0],
        },
        "q2.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036246827501982067, -0.006622624923678124, -0.008475455083195447, -0.008862802061736062, -0.007717690047773467, -0.005238119311537855, -0.0018528301595173225, 0.0018528301595173203, 0.005238119311537853, 0.007717690047773464, 0.008862802061736062, 0.008475455083195449, 0.006622624923678123, 0.0036246827501982067, 1.0097826916988402e-17],
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.252757258898498,
        },
        "q2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q2.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q3.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007766441642679343, 0.02972287825479078, 0.062072842478135645, 0.0992227316082989, 0.1347489923185278, 0.1625088113860774] + [0.17770226390412922] * 2 + [0.1625088113860774, 0.13474899231852785, 0.09922273160829898, 0.062072842478135666, 0.02972287825479076, 0.007766441642679332, 0.0],
        },
        "q3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012630769558612049, -0.023077564313600953, -0.029534038545678227, -0.030883809204890714, -0.02689348868197908, -0.018253039646278665, -0.0064564742320772734, 0.006456474232077266, 0.018253039646278658, 0.02689348868197907, 0.030883809204890714, 0.029534038545678234, 0.02307756431360095, 0.012630769558612049, 3.518744497687586e-17],
        },
        "q3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003852090915889313, 0.014742302149052635, 0.03078761724281289, 0.04921365545034744, 0.06683438737027161, 0.08060302837430647] + [0.08813885534867771] * 2 + [0.08060302837430647, 0.06683438737027164, 0.049213655450347484, 0.030787617242812898, 0.014742302149052625, 0.003852090915889308, 0.0],
        },
        "q3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.007851291972546248, -0.014345024236290884, -0.018358371489126866, -0.019197389530923675, -0.01671700452001502, -0.011346097558377427, -0.004013347252835981, 0.004013347252835976, 0.011346097558377422, 0.016717004520015014, 0.019197389530923672, 0.01835837148912687, 0.01434502423629088, 0.007851291972546248, 2.1872531439938347e-17],
        },
        "q3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.003852090915889312, -0.014742302149052634, -0.030787617242812888, -0.04921365545034744, -0.06683438737027161, -0.08060302837430647] + [-0.08813885534867771] * 2 + [-0.08060302837430647, -0.06683438737027164, -0.049213655450347484, -0.0307876172428129, -0.014742302149052627, -0.0038520909158893087, -2.6786125617170344e-33],
        },
        "q3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007851291972546248, 0.014345024236290886, 0.01835837148912687, 0.019197389530923682, 0.016717004520015028, 0.011346097558377437, 0.004013347252835991, -0.004013347252835965, -0.011346097558377411, -0.016717004520015007, -0.019197389530923665, -0.018358371489126866, -0.014345024236290879, -0.007851291972546248, -2.1872531439938347e-17],
        },
        "q3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.012630769558612049, 0.023077564313600956, 0.02953403854567823, 0.03088380920489072, 0.026893488681979088, 0.018253039646278675, 0.006456474232077285, -0.006456474232077254, -0.018253039646278647, -0.026893488681979064, -0.030883809204890707, -0.02953403854567823, -0.023077564313600946, -0.012630769558612049, -3.518744497687586e-17],
        },
        "q3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007766441642679342, 0.02972287825479078, 0.062072842478135645, 0.0992227316082989, 0.1347489923185278, 0.1625088113860774] + [0.17770226390412922] * 2 + [0.1625088113860774, 0.13474899231852785, 0.09922273160829898, 0.062072842478135666, 0.02972287825479076, 0.007766441642679333, 2.1546095930552317e-33],
        },
        "q3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007851291972546248, 0.014345024236290886, 0.01835837148912687, 0.01919738953092368, 0.016717004520015025, 0.011346097558377432, 0.004013347252835986, -0.0040133472528359705, -0.011346097558377417, -0.01671700452001501, -0.01919738953092367, -0.018358371489126866, -0.014345024236290879, -0.007851291972546248, -2.1872531439938347e-17],
        },
        "q3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038520909158893126, 0.014742302149052634, 0.03078761724281289, 0.04921365545034744, 0.06683438737027161, 0.08060302837430647] + [0.08813885534867771] * 2 + [0.08060302837430647, 0.06683438737027164, 0.049213655450347484, 0.030787617242812898, 0.014742302149052627, 0.0038520909158893083, 1.3393062808585172e-33],
        },
        "q3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.007851291972546248, -0.014345024236290882, -0.018358371489126862, -0.019197389530923672, -0.016717004520015018, -0.011346097558377422, -0.004013347252835976, 0.004013347252835981, 0.011346097558377427, 0.016717004520015018, 0.019197389530923675, 0.018358371489126873, 0.014345024236290882, 0.007851291972546248, 2.1872531439938347e-17],
        },
        "q3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0038520909158893135, -0.014742302149052637, -0.03078761724281289, -0.04921365545034744, -0.06683438737027161, -0.08060302837430647] + [-0.08813885534867771] * 2 + [-0.08060302837430647, -0.06683438737027164, -0.049213655450347484, -0.030787617242812898, -0.014742302149052623, -0.0038520909158893074, 1.3393062808585172e-33],
        },
        "q3.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q3.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q3.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.13625246720385348,
        },
        "q3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008708196739379495, 0.033327060629795464, 0.0695997664492379, 0.11125443383433788, 0.15108859237343839, 0.18221455417315943] + [0.19925035766784344] * 2 + [0.18221455417315943, 0.15108859237343844, 0.11125443383433796, 0.06959976644923793, 0.033327060629795444, 0.008708196739379484, 0.0],
        },
        "q3.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q3.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038310543317542738, 0.014661793229020795, 0.030619483542288252, 0.048944895645327356, 0.06646939930438014, 0.08016284863167868] + [0.08765752183745126] * 2 + [0.08016284863167868, 0.06646939930438017, 0.0489448956453274, 0.030619483542288262, 0.014661793229020784, 0.003831054331754269, 0.0],
        },
        "q3.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0063272797074245945, -0.011560515271903887, -0.014794832721886832, -0.01547098918741978, -0.013472071072923433, -0.009143709479995186, -0.0032343174499829446, 0.0032343174499829407, 0.009143709479995182, 0.013472071072923426, 0.01547098918741978, 0.014794832721886836, 0.011560515271903885, 0.0063272797074245945, 1.7626859988630127e-17],
        },
        "q3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.2214559913518434,
        },
        "q3.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q3.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q3.z.cz3_2.wf": {
            "type": "constant",
            "sample": -0.07709397834356477,
        },
        "q3.z.cz3_4.wf": {
            "type": "constant",
            "sample": -0.23305216356732317,
        },
        "q4.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009887219669888385, 0.03783928857605898, 0.07902304006806085, 0.12631742937029025, 0.17154482691635337, 0.20688500479252814] + [0.22622732518858696] * 2 + [0.20688500479252814, 0.17154482691635345, 0.12631742937029036, 0.07902304006806088, 0.03783928857605895, 0.009887219669888373, 0.0],
        },
        "q4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00480253588690003, -0.008774669689286205, -0.01122958338702322, -0.011742800099581547, -0.010225579994932385, -0.006940264212681516, -0.0024549136977370144, 0.002454913697737011, 0.006940264212681514, 0.010225579994932382, 0.011742800099581546, 0.01122958338702322, 0.008774669689286203, 0.00480253588690003, 1.3379150532799059e-17],
        },
        "q4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004903742013694642, 0.018767066511513116, 0.0391928787434334, 0.06264937021187782, 0.08508070044641718, 0.10260828866726192] + [0.11220145563788786] * 2 + [0.10260828866726192, 0.08508070044641722, 0.06264937021187787, 0.03919287874343341, 0.018767066511513105, 0.004903742013694635, 0.0],
        },
        "q4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -9.365644863722523e-07, -1.7111884646294932e-06, -2.1899324116931283e-06, -2.290017149863633e-06, -1.9941371186700266e-06, -1.353452663491381e-06, -4.787439470636344e-07, 4.787439470636339e-07, 1.3534526634913805e-06, 1.9941371186700258e-06, 2.290017149863633e-06, 2.1899324116931283e-06, 1.7111884646294928e-06, 9.365644863722523e-07, 2.60912933124092e-21],
        },
        "q4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004903742013694642, -0.018767066511513116, -0.0391928787434334, -0.06264937021187782, -0.08508070044641718, -0.10260828866726192] + [-0.11220145563788786] * 2 + [-0.10260828866726192, -0.08508070044641722, -0.06264937021187787, -0.03919287874343341, -0.018767066511513105, -0.004903742013694635, -3.195261884065667e-37],
        },
        "q4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 9.365644863728529e-07, 1.7111884646317914e-06, 2.189932411697928e-06, 2.2900171498713055e-06, 1.994137118680446e-06, 1.3534526635039469e-06, 4.787439470773751e-07, -4.787439470498932e-07, -1.3534526634788146e-06, -1.9941371186596064e-06, -2.2900171498559606e-06, -2.1899324116883285e-06, -1.7111884646271945e-06, -9.365644863716518e-07, -2.60912933124092e-21],
        },
        "q4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004802535886900031, 0.008774669689286207, 0.011229583387023226, 0.011742800099581554, 0.010225579994932395, 0.006940264212681529, 0.0024549136977370283, -0.002454913697736997, -0.006940264212681501, -0.010225579994932371, -0.011742800099581539, -0.011229583387023215, -0.008774669689286202, -0.004802535886900029, -1.3379150532799059e-17],
        },
        "q4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.009887219669888385, 0.03783928857605898, 0.07902304006806085, 0.12631742937029025, 0.17154482691635337, 0.20688500479252814] + [0.22622732518858696] * 2 + [0.20688500479252814, 0.17154482691635345, 0.12631742937029036, 0.07902304006806088, 0.03783928857605895, 0.009887219669888373, 8.192366937651486e-34],
        },
        "q4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 9.365644863725526e-07, 1.7111884646306424e-06, 2.1899324116955283e-06, 2.2900171498674693e-06, 1.9941371186752363e-06, 1.3534526634976638e-06, 4.787439470705048e-07, -4.787439470567635e-07, -1.3534526634850976e-06, -1.994137118664816e-06, -2.290017149859797e-06, -2.189932411690728e-06, -1.7111884646283435e-06, -9.365644863719521e-07, -2.60912933124092e-21],
        },
        "q4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004903742013694642, 0.018767066511513116, 0.0391928787434334, 0.06264937021187782, 0.08508070044641718, 0.10260828866726192] + [0.11220145563788786] * 2 + [0.10260828866726192, 0.08508070044641722, 0.06264937021187787, 0.03919287874343341, 0.018767066511513105, 0.004903742013694635, 1.5976309420328335e-37],
        },
        "q4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -9.365644863719521e-07, -1.711188464628344e-06, -2.189932411690728e-06, -2.290017149859797e-06, -1.994137118664817e-06, -1.353452663485098e-06, -4.787439470567641e-07, 4.787439470705043e-07, 1.3534526634976634e-06, 1.9941371186752355e-06, 2.2900171498674693e-06, 2.1899324116955283e-06, 1.711188464630642e-06, 9.365644863725526e-07, 2.60912933124092e-21],
        },
        "q4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004903742013694642, -0.018767066511513116, -0.0391928787434334, -0.06264937021187782, -0.08508070044641718, -0.10260828866726192] + [-0.11220145563788786] * 2 + [-0.10260828866726192, -0.08508070044641722, -0.06264937021187787, -0.03919287874343341, -0.018767066511513105, -0.004903742013694635, 1.5976309420328335e-37],
        },
        "q4.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q4.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q4.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.14844517355097134,
        },
        "q4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009681443178495789, 0.03705176323527111, 0.07737838318095787, 0.12368846407110583, 0.16797457220593126, 0.20257923716226092] + [0.22151899799563357] * 2 + [0.20257923716226092, 0.16797457220593132, 0.12368846407110592, 0.0773783831809579, 0.03705176323527108, 0.009681443178495777, 0.0],
        },
        "q4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q4.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.012968181353609854, 0.0],
        },
        "q4.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00017685586774239802, -0.00032313174926701664, -0.00041353521578358374, -0.0004324346866416748, -0.0003765622716336233, -0.0002555788188992768, -9.040346651656707e-05, 9.040346651656696e-05, 0.0002555788188992767, 0.00037656227163362314, 0.00043243468664167477, 0.0004135352157835838, 0.0003231317492670166, 0.00017685586774239802, 4.926941376093873e-19],
        },
        "q4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.1436923076923077,
        },
        "q4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q4.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007448136311174729, 0.028504694812302632, 0.05952880524571009, 0.09515611707226578, 0.12922634441319125, 0.15584843544294205] + [0.17041918876836984] * 2 + [0.15584843544294205, 0.1292263444131913, 0.09515611707226584, 0.0595288052457101, 0.02850469481230261, 0.00744813631117472, 0.0],
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.009243514293683218, -0.016888740991297516, -0.02161375094212127, -0.022601547000289227, -0.019681330253498677, -0.013358032706606009, -0.004725009950823751, 0.004725009950823745, 0.013358032706606003, 0.01968133025349867, 0.022601547000289223, 0.02161375094212127, 0.016888740991297512, 0.009243514293683218, 2.5751055713004787e-17],
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0036795564274120533, 0.014081996975627937, 0.029408645170655546, 0.04700938430134693, 0.06384088667431032, 0.07699283261986167] + [0.08419113120233718] * 2 + [0.07699283261986167, 0.06384088667431034, 0.04700938430134697, 0.029408645170655553, 0.014081996975627929, 0.003679556427412049, 0.0],
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0002678862941982928, -0.0004894526144591197, -0.0006263881311425552, -0.0006550154493939186, -0.0005703846458165272, -0.00038712915519562566, -0.00013693551668343546, 0.0001369355166834353, 0.00038712915519562556, 0.0005703846458165269, 0.0006550154493939183, 0.0006263881311425552, 0.0004894526144591196, 0.0002678862941982928, 7.462913635958554e-19],
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036795564274120533, -0.014081996975627937, -0.029408645170655546, -0.04700938430134693, -0.06384088667431032, -0.07699283261986167] + [-0.08419113120233718] * 2 + [-0.07699283261986167, -0.06384088667431034, -0.04700938430134697, -0.029408645170655553, -0.014081996975627929, -0.003679556427412049, -9.13943329658978e-35],
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0002678862941982932, 0.0004894526144591214, 0.0006263881311425588, 0.0006550154493939243, 0.000570384645816535, 0.0003871291551956351, 0.00013693551668344576, -0.000136935516683425, -0.0003871291551956161, -0.0005703846458165191, -0.0006550154493939126, -0.0006263881311425516, -0.0004894526144591178, -0.00026788629419829236, -7.462913635958554e-19],
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009243514293683218, 0.01688874099129752, 0.021613750942121274, 0.022601547000289234, 0.019681330253498684, 0.013358032706606019, 0.004725009950823761, -0.0047250099508237345, -0.013358032706605993, -0.019681330253498663, -0.022601547000289216, -0.021613750942121267, -0.01688874099129751, -0.009243514293683218, -2.5751055713004787e-17],
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0074481363111747285, 0.028504694812302632, 0.05952880524571009, 0.09515611707226578, 0.12922634441319125, 0.15584843544294205] + [0.17041918876836984] * 2 + [0.15584843544294205, 0.1292263444131913, 0.09515611707226584, 0.0595288052457101, 0.02850469481230261, 0.007448136311174721, 1.576797397679824e-33],
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.000267886294198293, 0.0004894526144591205, 0.0006263881311425571, 0.0006550154493939215, 0.0005703846458165311, 0.0003871291551956304, 0.0001369355166834406, -0.00013693551668343015, -0.00038712915519562084, -0.000570384645816523, -0.0006550154493939154, -0.0006263881311425534, -0.0004894526144591187, -0.0002678862941982926, -7.462913635958554e-19],
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0036795564274120533, 0.014081996975627937, 0.029408645170655546, 0.04700938430134693, 0.06384088667431032, 0.07699283261986167] + [0.08419113120233718] * 2 + [0.07699283261986167, 0.06384088667431034, 0.04700938430134697, 0.029408645170655553, 0.014081996975627929, 0.003679556427412049, 4.56971664829489e-35],
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0002678862941982926, -0.0004894526144591188, -0.0006263881311425534, -0.0006550154493939156, -0.0005703846458165233, -0.00038712915519562095, -0.0001369355166834303, 0.00013693551668344045, 0.00038712915519563027, 0.0005703846458165308, 0.0006550154493939213, 0.0006263881311425571, 0.0004894526144591204, 0.000267886294198293, 7.462913635958554e-19],
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036795564274120533, -0.014081996975627937, -0.029408645170655546, -0.04700938430134693, -0.06384088667431032, -0.07699283261986167] + [-0.08419113120233718] * 2 + [-0.07699283261986167, -0.06384088667431034, -0.04700938430134697, -0.029408645170655553, -0.014081996975627929, -0.003679556427412049, 4.56971664829489e-35],
        },
        "q5.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q5.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q5.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q5.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q5.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q5.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q5.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q5.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q5.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.12782954309787248,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.012968181353609854, 0.0],
        },
        "q5.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
        },
        "q5.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004618343160591265, 0.017674819153568368, 0.03691184466538491, 0.05900316322215746, 0.08012898515209145, 0.09663646392143418] + [0.10567130648522964] * 2 + [0.09663646392143418, 0.08012898515209148, 0.059003163222157505, 0.03691184466538492, 0.017674819153568357, 0.004618343160591259, 0.0],
        },
        "q5.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008065355175237464, -0.0014736137169224855, -0.0018858907173049866, -0.001972080079886148, -0.0017172788808699087, -0.0011655445623624013, -0.0004122770003825005, 0.00041227700038250003, 0.001165544562362401, 0.001717278880869908, 0.001972080079886148, 0.0018858907173049866, 0.0014736137169224855, 0.0008065355175237464, 2.246887967757484e-18],
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.136350332271279,
        },
        "q5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.z.cz5_4.wf": {
            "type": "constant",
            "sample": -0.1732650364352627,
        },
        "coupler_q1_q2.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "coupler_q1_q2.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q1_q2.cz.wf": {
            "type": "constant",
            "sample": -0.04662292425886875,
        },
        "coupler_q2_q3.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "coupler_q2_q3.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q2_q3.cz.wf": {
            "type": "constant",
            "sample": -0.10043328056510244,
        },
        "coupler_q3_q4.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "coupler_q3_q4.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q3_q4.cz.wf": {
            "type": "constant",
            "sample": -0.11048567109,
        },
        "coupler_q4_q5.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "coupler_q4_q5.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q4_q5.cz.wf": {
            "type": "constant",
            "sample": -0.09938541439500001,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [[1, 0]],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(-0.6829156716957551, 1040)],
            "sine": [(0.7304972178950004, 1040)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(-0.7304972178950004, 1040)],
            "sine": [(-0.6829156716957551, 1040)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(0.7304972178950004, 1040)],
            "sine": [(0.6829156716957551, 1040)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(0.6031753077014853, 1040)],
            "sine": [(0.7976086434958052, 1040)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.7976086434958052, 1040)],
            "sine": [(0.6031753077014853, 1040)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.7976086434958052, 1040)],
            "sine": [(-0.6031753077014853, 1040)],
        },
        "q3.resonator.readout.iw1": {
            "cosine": [(-0.3203852993033521, 1040)],
            "sine": [(-0.947287316493946, 1040)],
        },
        "q3.resonator.readout.iw2": {
            "cosine": [(0.947287316493946, 1040)],
            "sine": [(-0.3203852993033521, 1040)],
        },
        "q3.resonator.readout.iw3": {
            "cosine": [(-0.947287316493946, 1040)],
            "sine": [(0.3203852993033521, 1040)],
        },
        "q4.resonator.readout.iw1": {
            "cosine": [(-0.9722269651436221, 1040)],
            "sine": [(0.2340400142018932, 1040)],
        },
        "q4.resonator.readout.iw2": {
            "cosine": [(-0.2340400142018932, 1040)],
            "sine": [(-0.9722269651436221, 1040)],
        },
        "q4.resonator.readout.iw3": {
            "cosine": [(0.2340400142018932, 1040)],
            "sine": [(0.9722269651436221, 1040)],
        },
        "q5.resonator.readout.iw1": {
            "cosine": [(0.5461865405282237, 1040)],
            "sine": [(0.837663573844423, 1040)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(-0.837663573844423, 1040)],
            "sine": [(0.5461865405282237, 1040)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(0.837663573844423, 1040)],
            "sine": [(-0.5461865405282237, 1040)],
        },
    },
    "mixers": {},
    "oscillators": {},
}

loaded_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                "5": {
                    "type": "LF",
                    "analog_outputs": {
                        "7": {
                            "offset": 0.0,
                            "delay": 0,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
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
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
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
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                        "3": {
                            "offset": 0.0,
                            "delay": 0,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
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
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
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
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
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
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
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
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                    },
                },
                "3": {
                    "type": "LF",
                    "analog_outputs": {
                        "8": {
                            "offset": 0.0,
                            "delay": 0,
                            "shareable": False,
                            "filter": {
                                "feedforward": [],
                                "feedback": [],
                            },
                            "crosstalk": {},
                            "output_mode": "amplified",
                            "sampling_rate": 1000000000.0,
                            "upsampling_mode": "pulse",
                        },
                    },
                },
                "1": {
                    "type": "MW",
                    "analog_outputs": {
                        "1": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": -14,
                            "band": 2,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 5950000000.0,
                                },
                            },
                        },
                        "2": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4900000000.0,
                                },
                            },
                        },
                        "3": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4900000000.0,
                                },
                            },
                        },
                        "4": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 5000000000.0,
                                },
                            },
                        },
                        "5": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 5000000000.0,
                                },
                            },
                        },
                        "6": {
                            "sampling_rate": 1000000000.0,
                            "full_scale_power_dbm": 4,
                            "band": 1,
                            "delay": 0,
                            "shareable": False,
                            "upconverters": {
                                "1": {
                                    "frequency": 4900000000.0,
                                },
                            },
                        },
                    },
                    "analog_inputs": {
                        "1": {
                            "band": 2,
                            "shareable": False,
                            "gain_db": 0,
                            "sampling_rate": 1000000000.0,
                            "downconverter_frequency": 5950000000.0,
                        },
                    },
                },
            },
        },
    },
    "oscillators": {},
    "elements": {
        "q1.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q1.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q1.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q1.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q1.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q1.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q1.xy.-y90_DragCosine.pulse",
                "x180_Square": "q1.xy.x180_Square.pulse",
                "x90_Square": "q1.xy.x90_Square.pulse",
                "-x90_Square": "q1.xy.-x90_Square.pulse",
                "y180_Square": "q1.xy.y180_Square.pulse",
                "y90_Square": "q1.xy.y90_Square.pulse",
                "-y90_Square": "q1.xy.-y90_Square.pulse",
                "x180": "q1.xy.x180_DragCosine.pulse",
                "x90": "q1.xy.x90_DragCosine.pulse",
                "-x90": "q1.xy.-x90_DragCosine.pulse",
                "y180": "q1.xy.y180_DragCosine.pulse",
                "y90": "q1.xy.y90_DragCosine.pulse",
                "-y90": "q1.xy.-y90_DragCosine.pulse",
                "saturation": "q1.xy.saturation.pulse",
                "EF_x180": "q1.xy.EF_x180.pulse",
                "EF_x90": "q1.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "a",
            "MWInput": {
                "port": ('con1', 1, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": 213082937.51994482,
        },
        "q1.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q1.resonator.readout.pulse",
                "const": "q1.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "a",
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": -16482781.0,
        },
        "q1.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q1.z.const.pulse",
                "flux_pulse": "q1.z.flux_pulse.pulse",
                "cz1_2": "q1.z.cz1_2.pulse",
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
                "port": ('con1', 5, 7),
            },
        },
        "q2.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q2.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q2.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q2.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q2.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q2.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q2.xy.-y90_DragCosine.pulse",
                "x180_Square": "q2.xy.x180_Square.pulse",
                "x90_Square": "q2.xy.x90_Square.pulse",
                "-x90_Square": "q2.xy.-x90_Square.pulse",
                "y180_Square": "q2.xy.y180_Square.pulse",
                "y90_Square": "q2.xy.y90_Square.pulse",
                "-y90_Square": "q2.xy.-y90_Square.pulse",
                "x180": "q2.xy.x180_DragCosine.pulse",
                "x90": "q2.xy.x90_DragCosine.pulse",
                "-x90": "q2.xy.-x90_DragCosine.pulse",
                "y180": "q2.xy.y180_DragCosine.pulse",
                "y90": "q2.xy.y90_DragCosine.pulse",
                "-y90": "q2.xy.-y90_DragCosine.pulse",
                "saturation": "q2.xy.saturation.pulse",
                "EF_x180": "q2.xy.EF_x180.pulse",
                "EF_x90": "q2.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "b",
            "MWInput": {
                "port": ('con1', 1, 3),
                "upconverter": 1,
            },
            "intermediate_frequency": -62048681.59228143,
        },
        "q2.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q2.resonator.readout.pulse",
                "const": "q2.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "b",
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 376,
            "intermediate_frequency": 74264448.0,
        },
        "q2.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q2.z.const.pulse",
                "flux_pulse": "q2.z.flux_pulse.pulse",
                "cz": "q2.z.cz.pulse",
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
                "port": ('con1', 5, 1),
            },
        },
        "q3.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q3.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q3.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q3.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q3.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q3.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q3.xy.-y90_DragCosine.pulse",
                "x180_Square": "q3.xy.x180_Square.pulse",
                "x90_Square": "q3.xy.x90_Square.pulse",
                "-x90_Square": "q3.xy.-x90_Square.pulse",
                "y180_Square": "q3.xy.y180_Square.pulse",
                "y90_Square": "q3.xy.y90_Square.pulse",
                "-y90_Square": "q3.xy.-y90_Square.pulse",
                "x180": "q3.xy.x180_DragCosine.pulse",
                "x90": "q3.xy.x90_DragCosine.pulse",
                "-x90": "q3.xy.-x90_DragCosine.pulse",
                "y180": "q3.xy.y180_DragCosine.pulse",
                "y90": "q3.xy.y90_DragCosine.pulse",
                "-y90": "q3.xy.-y90_DragCosine.pulse",
                "saturation": "q3.xy.saturation.pulse",
                "EF_x180": "q3.xy.EF_x180.pulse",
                "EF_x90": "q3.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "c",
            "MWInput": {
                "port": ('con1', 1, 4),
                "upconverter": 1,
            },
            "intermediate_frequency": 145290051.2800316,
        },
        "q3.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q3.resonator.readout.pulse",
                "const": "q3.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "c",
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": -83726496.0,
        },
        "q3.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q3.z.const.pulse",
                "flux_pulse": "q3.z.flux_pulse.pulse",
                "cz3_2": "q3.z.cz3_2.pulse",
                "cz3_4": "q3.z.cz3_4.pulse",
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
                "port": ('con1', 5, 2),
            },
        },
        "q4.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q4.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q4.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q4.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q4.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q4.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q4.xy.-y90_DragCosine.pulse",
                "x180_Square": "q4.xy.x180_Square.pulse",
                "x90_Square": "q4.xy.x90_Square.pulse",
                "-x90_Square": "q4.xy.-x90_Square.pulse",
                "y180_Square": "q4.xy.y180_Square.pulse",
                "y90_Square": "q4.xy.y90_Square.pulse",
                "-y90_Square": "q4.xy.-y90_Square.pulse",
                "x180": "q4.xy.x180_DragCosine.pulse",
                "x90": "q4.xy.x90_DragCosine.pulse",
                "-x90": "q4.xy.-x90_DragCosine.pulse",
                "y180": "q4.xy.y180_DragCosine.pulse",
                "y90": "q4.xy.y90_DragCosine.pulse",
                "-y90": "q4.xy.-y90_DragCosine.pulse",
                "saturation": "q4.xy.saturation.pulse",
                "EF_x180": "q4.xy.EF_x180.pulse",
                "EF_x90": "q4.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "d",
            "MWInput": {
                "port": ('con1', 1, 5),
                "upconverter": 1,
            },
            "intermediate_frequency": -323672230.4042064,
        },
        "q4.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q4.resonator.readout.pulse",
                "const": "q4.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "d",
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": 129555618.0,
        },
        "q4.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q4.z.const.pulse",
                "flux_pulse": "q4.z.flux_pulse.pulse",
                "cz": "q4.z.cz.pulse",
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
                "port": ('con1', 5, 3),
            },
        },
        "q5.xy": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "x180_DragCosine": "q5.xy.x180_DragCosine.pulse",
                "x90_DragCosine": "q5.xy.x90_DragCosine.pulse",
                "-x90_DragCosine": "q5.xy.-x90_DragCosine.pulse",
                "y180_DragCosine": "q5.xy.y180_DragCosine.pulse",
                "y90_DragCosine": "q5.xy.y90_DragCosine.pulse",
                "-y90_DragCosine": "q5.xy.-y90_DragCosine.pulse",
                "x180_Square": "q5.xy.x180_Square.pulse",
                "x90_Square": "q5.xy.x90_Square.pulse",
                "-x90_Square": "q5.xy.-x90_Square.pulse",
                "y180_Square": "q5.xy.y180_Square.pulse",
                "y90_Square": "q5.xy.y90_Square.pulse",
                "-y90_Square": "q5.xy.-y90_Square.pulse",
                "x180": "q5.xy.x180_DragCosine.pulse",
                "x90": "q5.xy.x90_DragCosine.pulse",
                "-x90": "q5.xy.-x90_DragCosine.pulse",
                "y180": "q5.xy.y180_DragCosine.pulse",
                "y90": "q5.xy.y90_DragCosine.pulse",
                "-y90": "q5.xy.-y90_DragCosine.pulse",
                "saturation": "q5.xy.saturation.pulse",
                "EF_x180": "q5.xy.EF_x180.pulse",
                "EF_x90": "q5.xy.EF_x90.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "e",
            "MWInput": {
                "port": ('con1', 1, 6),
                "upconverter": 1,
            },
            "intermediate_frequency": -14015480.963453503,
        },
        "q5.resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "readout": "q5.resonator.readout.pulse",
                "const": "q5.resonator.const.pulse",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "thread": "e",
            "MWInput": {
                "port": ('con1', 1, 1),
                "upconverter": 1,
            },
            "MWOutput": {
                "port": ('con1', 1, 1),
            },
            "smearing": 0,
            "time_of_flight": 384,
            "intermediate_frequency": 19883495.0,
        },
        "q5.z": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "q5.z.const.pulse",
                "flux_pulse": "q5.z.flux_pulse.pulse",
                "cz5_4": "q5.z.cz5_4.pulse",
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
                "port": ('con1', 5, 4),
            },
        },
        "coupler_q1_q2": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q1_q2.const.pulse",
                "flux_pulse": "coupler_q1_q2.flux_pulse.pulse",
                "cz": "coupler_q1_q2.cz.pulse",
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
                "port": ('con1', 5, 5),
            },
        },
        "coupler_q2_q3": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q2_q3.const.pulse",
                "flux_pulse": "coupler_q2_q3.flux_pulse.pulse",
                "cz": "coupler_q2_q3.cz.pulse",
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
                "port": ('con1', 5, 6),
            },
        },
        "coupler_q3_q4": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q3_q4.const.pulse",
                "flux_pulse": "coupler_q3_q4.flux_pulse.pulse",
                "cz": "coupler_q3_q4.cz.pulse",
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
                "port": ('con1', 3, 8),
            },
        },
        "coupler_q4_q5": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "const": "coupler_q4_q5.const.pulse",
                "flux_pulse": "coupler_q4_q5.flux_pulse.pulse",
                "cz": "coupler_q4_q5.cz.pulse",
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
                "port": ('con1', 5, 8),
            },
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
        "q1.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.x180_DragCosine.wf.I",
                "Q": "q1.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.x90_DragCosine.wf.I",
                "Q": "q1.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.-x90_DragCosine.wf.I",
                "Q": "q1.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.y180_DragCosine.wf.I",
                "Q": "q1.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.y90_DragCosine.wf.I",
                "Q": "q1.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.-y90_DragCosine.wf.I",
                "Q": "q1.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.x180_Square.wf.I",
                "Q": "q1.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.x90_Square.wf.I",
                "Q": "q1.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.-x90_Square.wf.I",
                "Q": "q1.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.y180_Square.wf.I",
                "Q": "q1.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.y90_Square.wf.I",
                "Q": "q1.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q1.xy.-y90_Square.wf.I",
                "Q": "q1.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q1.xy.saturation.wf.I",
                "Q": "q1.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.EF_x180.wf.I",
                "Q": "q1.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q1.xy.EF_x90.wf.I",
                "Q": "q1.xy.EF_x90.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q1.resonator.readout.pulse": {
            "length": 1040,
            "waveforms": {
                "I": "q1.resonator.readout.wf.I",
                "Q": "q1.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q1.resonator.readout.iw1",
                "iw2": "q1.resonator.readout.iw2",
                "iw3": "q1.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q1.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q1.resonator.const.wf.I",
                "Q": "q1.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q1.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q1.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q1.z.cz1_2.pulse": {
            "length": 80,
            "waveforms": {
                "single": "q1.z.cz1_2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.x180_DragCosine.wf.I",
                "Q": "q2.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.x90_DragCosine.wf.I",
                "Q": "q2.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.-x90_DragCosine.wf.I",
                "Q": "q2.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.y180_DragCosine.wf.I",
                "Q": "q2.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.y90_DragCosine.wf.I",
                "Q": "q2.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.-y90_DragCosine.wf.I",
                "Q": "q2.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x180_Square.wf.I",
                "Q": "q2.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.x90_Square.wf.I",
                "Q": "q2.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.-x90_Square.wf.I",
                "Q": "q2.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.y180_Square.wf.I",
                "Q": "q2.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.y90_Square.wf.I",
                "Q": "q2.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q2.xy.-y90_Square.wf.I",
                "Q": "q2.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q2.xy.saturation.wf.I",
                "Q": "q2.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.EF_x180.wf.I",
                "Q": "q2.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q2.xy.EF_x90.wf.I",
                "Q": "q2.xy.EF_x90.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q2.resonator.readout.pulse": {
            "length": 1040,
            "waveforms": {
                "I": "q2.resonator.readout.wf.I",
                "Q": "q2.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q2.resonator.readout.iw1",
                "iw2": "q2.resonator.readout.iw2",
                "iw3": "q2.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q2.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q2.resonator.const.wf.I",
                "Q": "q2.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q2.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q2.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q2.z.cz.pulse": {
            "length": 40,
            "waveforms": {
                "single": "q2.z.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.x180_DragCosine.wf.I",
                "Q": "q3.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.x90_DragCosine.wf.I",
                "Q": "q3.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.-x90_DragCosine.wf.I",
                "Q": "q3.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.y180_DragCosine.wf.I",
                "Q": "q3.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.y90_DragCosine.wf.I",
                "Q": "q3.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.-y90_DragCosine.wf.I",
                "Q": "q3.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.x180_Square.wf.I",
                "Q": "q3.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.x90_Square.wf.I",
                "Q": "q3.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.-x90_Square.wf.I",
                "Q": "q3.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.y180_Square.wf.I",
                "Q": "q3.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.y90_Square.wf.I",
                "Q": "q3.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q3.xy.-y90_Square.wf.I",
                "Q": "q3.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q3.xy.saturation.wf.I",
                "Q": "q3.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.EF_x180.wf.I",
                "Q": "q3.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q3.xy.EF_x90.wf.I",
                "Q": "q3.xy.EF_x90.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q3.resonator.readout.pulse": {
            "length": 1040,
            "waveforms": {
                "I": "q3.resonator.readout.wf.I",
                "Q": "q3.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q3.resonator.readout.iw1",
                "iw2": "q3.resonator.readout.iw2",
                "iw3": "q3.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q3.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q3.resonator.const.wf.I",
                "Q": "q3.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q3.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q3.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.cz3_2.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q3.z.cz3_2.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q3.z.cz3_4.pulse": {
            "length": 60,
            "waveforms": {
                "single": "q3.z.cz3_4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.x180_DragCosine.wf.I",
                "Q": "q4.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.x90_DragCosine.wf.I",
                "Q": "q4.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.-x90_DragCosine.wf.I",
                "Q": "q4.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.y180_DragCosine.wf.I",
                "Q": "q4.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.y90_DragCosine.wf.I",
                "Q": "q4.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.-y90_DragCosine.wf.I",
                "Q": "q4.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.x180_Square.wf.I",
                "Q": "q4.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.x90_Square.wf.I",
                "Q": "q4.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.-x90_Square.wf.I",
                "Q": "q4.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.y180_Square.wf.I",
                "Q": "q4.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.y90_Square.wf.I",
                "Q": "q4.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q4.xy.-y90_Square.wf.I",
                "Q": "q4.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q4.xy.saturation.wf.I",
                "Q": "q4.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.EF_x180.wf.I",
                "Q": "q4.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q4.xy.EF_x90.wf.I",
                "Q": "q4.xy.EF_x90.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q4.resonator.readout.pulse": {
            "length": 1040,
            "waveforms": {
                "I": "q4.resonator.readout.wf.I",
                "Q": "q4.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q4.resonator.readout.iw1",
                "iw2": "q4.resonator.readout.iw2",
                "iw3": "q4.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q4.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q4.resonator.const.wf.I",
                "Q": "q4.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q4.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q4.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q4.z.cz.pulse": {
            "length": 40,
            "waveforms": {
                "single": "q4.z.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.xy.x180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.x180_DragCosine.wf.I",
                "Q": "q5.xy.x180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.x90_DragCosine.wf.I",
                "Q": "q5.xy.x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-x90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.-x90_DragCosine.wf.I",
                "Q": "q5.xy.-x90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y180_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.y180_DragCosine.wf.I",
                "Q": "q5.xy.y180_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.y90_DragCosine.wf.I",
                "Q": "q5.xy.y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-y90_DragCosine.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.-y90_DragCosine.wf.I",
                "Q": "q5.xy.-y90_DragCosine.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.x180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.x180_Square.wf.I",
                "Q": "q5.xy.x180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.x90_Square.wf.I",
                "Q": "q5.xy.x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-x90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.-x90_Square.wf.I",
                "Q": "q5.xy.-x90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y180_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.y180_Square.wf.I",
                "Q": "q5.xy.y180_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.y90_Square.wf.I",
                "Q": "q5.xy.y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.-y90_Square.pulse": {
            "length": 40,
            "waveforms": {
                "I": "q5.xy.-y90_Square.wf.I",
                "Q": "q5.xy.-y90_Square.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.saturation.pulse": {
            "length": 20000,
            "waveforms": {
                "I": "q5.xy.saturation.wf.I",
                "Q": "q5.xy.saturation.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.EF_x180.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.EF_x180.wf.I",
                "Q": "q5.xy.EF_x180.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.xy.EF_x90.pulse": {
            "length": 16,
            "waveforms": {
                "I": "q5.xy.EF_x90.wf.I",
                "Q": "q5.xy.EF_x90.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
        "q5.resonator.readout.pulse": {
            "length": 1040,
            "waveforms": {
                "I": "q5.resonator.readout.wf.I",
                "Q": "q5.resonator.readout.wf.Q",
            },
            "integration_weights": {
                "iw1": "q5.resonator.readout.iw1",
                "iw2": "q5.resonator.readout.iw2",
                "iw3": "q5.resonator.readout.iw3",
            },
            "operation": "measurement",
            "digital_marker": "ON",
        },
        "q5.resonator.const.pulse": {
            "length": 100,
            "waveforms": {
                "I": "q5.resonator.const.wf.I",
                "Q": "q5.resonator.const.wf.Q",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.z.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q5.z.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.z.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "q5.z.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "q5.z.cz5_4.pulse": {
            "length": 88,
            "waveforms": {
                "single": "q5.z.cz5_4.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q1_q2.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q1_q2.cz.pulse": {
            "length": 80,
            "waveforms": {
                "single": "coupler_q1_q2.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q2_q3.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q2_q3.cz.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q2_q3.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q3_q4.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q3_q4.cz.pulse": {
            "length": 60,
            "waveforms": {
                "single": "coupler_q3_q4.cz.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.const.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.const.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.flux_pulse.pulse": {
            "length": 100,
            "waveforms": {
                "single": "coupler_q4_q5.flux_pulse.wf",
            },
            "integration_weights": {},
            "operation": "control",
        },
        "coupler_q4_q5.cz.pulse": {
            "length": 88,
            "waveforms": {
                "single": "coupler_q4_q5.cz.wf",
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
        "q1.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00689333908728961, 0.02638143539694644, 0.0550946200332339, 0.08806812252144451, 0.11960052472650065, 0.14423958784426721] + [0.157724994022821] * 2 + [0.14423958784426721, 0.1196005247265007, 0.08806812252144458, 0.05509462003323392, 0.026381435396946422, 0.006893339087289601, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.010500440545542306, -0.019185259527252734, -0.024552772844090034, -0.025674888681245372, -0.022357583016372237, -0.015174448135703067, -0.005367513316837296, 0.00536751331683729, 0.015174448135703062, 0.02235758301637223, 0.02567488868124537, 0.024552772844090038, 0.01918525952725273, 0.010500440545542306, 2.9252665264350483e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.00343277168258172, 0.013137529320656793, 0.027436232153646715, 0.0438565045620903, 0.05955913227307592, 0.07182898830147982] + [0.07854450307184835] * 2 + [0.07182898830147982, 0.059559132273075945, 0.04385650456209034, 0.027436232153646725, 0.013137529320656784, 0.003432771682581716, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00076108715916614, -0.0013905754342526768, -0.0017796205837757005, -0.0018609531670184527, -0.0016205100414549394, -0.001099866007852313, -0.00038904514952302344, 0.00038904514952302295, 0.0010998660078523125, 0.0016205100414549388, 0.0018609531670184525, 0.0017796205837757005, 0.0013905754342526766, 0.00076108715916614, 2.120275602487371e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.00343277168258172, -0.013137529320656793, -0.027436232153646715, -0.0438565045620903, -0.05955913227307592, -0.07182898830147982] + [-0.07854450307184835] * 2 + [-0.07182898830147982, -0.059559132273075945, -0.04385650456209034, -0.027436232153646725, -0.013137529320656784, -0.003432771682581716, -2.596588729896385e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0007610871591661405, 0.0013905754342526783, 0.0017796205837757038, 0.0018609531670184581, 0.0016205100414549468, 0.0010998660078523218, 0.00038904514952303303, -0.00038904514952301336, -0.0010998660078523036, -0.0016205100414549314, -0.001860953167018447, -0.0017796205837756973, -0.001390575434252675, -0.0007610871591661396, -2.120275602487371e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010500440545542306, 0.019185259527252734, 0.024552772844090038, 0.02567488868124538, 0.022357583016372244, 0.015174448135703076, 0.0053675133168373055, -0.00536751331683728, -0.015174448135703053, -0.022357583016372223, -0.025674888681245362, -0.024552772844090034, -0.01918525952725273, -0.010500440545542306, -2.9252665264350483e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.006893339087289609, 0.02638143539694644, 0.0550946200332339, 0.08806812252144451, 0.11960052472650065, 0.14423958784426721] + [0.157724994022821] * 2 + [0.14423958784426721, 0.1196005247265007, 0.08806812252144458, 0.05509462003323392, 0.026381435396946422, 0.006893339087289602, 1.791209144125789e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0007610871591661403, 0.0013905754342526777, 0.0017796205837757022, 0.0018609531670184553, 0.0016205100414549431, 0.0010998660078523172, 0.00038904514952302826, -0.0003890451495230181, -0.0010998660078523081, -0.001620510041454935, -0.00186095316701845, -0.0017796205837756988, -0.0013905754342526757, -0.0007610871591661398, -2.120275602487371e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.00343277168258172, 0.013137529320656793, 0.027436232153646715, 0.0438565045620903, 0.05955913227307592, 0.07182898830147982] + [0.07854450307184835] * 2 + [0.07182898830147982, 0.059559132273075945, 0.04385650456209034, 0.027436232153646725, 0.013137529320656784, 0.003432771682581716, 1.2982943649481925e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0007610871591661398, -0.001390575434252676, -0.0017796205837756988, -0.0018609531670184501, -0.0016205100414549358, -0.0010998660078523086, -0.0003890451495230186, 0.0003890451495230278, 0.0010998660078523168, 0.0016205100414549425, 0.001860953167018455, 0.0017796205837757022, 0.0013905754342526775, 0.0007610871591661403, 2.120275602487371e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00343277168258172, -0.013137529320656793, -0.027436232153646715, -0.0438565045620903, -0.05955913227307592, -0.07182898830147982] + [-0.07854450307184835] * 2 + [-0.07182898830147982, -0.059559132273075945, -0.04385650456209034, -0.027436232153646725, -0.013137529320656784, -0.003432771682581716, 1.2982943649481925e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q1.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q1.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q1.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q1.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q1.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q1.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q1.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q1.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.06583496945367821,
        },
        "q1.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004857743557162809, 0.018591026236402998, 0.03882523869952356, 0.062061701789601555, 0.08428261994172241, 0.10164579453668046] + [0.11114897494751827] * 2 + [0.10164579453668046, 0.08428261994172244, 0.0620617017896016, 0.038825238699523576, 0.018591026236402988, 0.004857743557162803, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.007396079806845666, -0.013513310223812048, -0.017293966538511965, -0.01808433892794943, -0.01574776482568296, -0.010688259121103763, -0.003780656314699915, 0.0037806563146999105, 0.010688259121103758, 0.015747764825682957, 0.018084338927949428, 0.017293966538511965, 0.013513310223812044, 0.007396079806845666, 2.0604378065825656e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004593670635837815, 0.01758039515822729, 0.036714650916558844, 0.0586879512609605, 0.07970091294071842, 0.09612020398463854] + [0.10510677980665072] * 2 + [0.09612020398463854, 0.07970091294071845, 0.05868795126096055, 0.03671465091655885, 0.017580395158227277, 0.004593670635837809, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0026711344749756555, -0.0048804055327331276, -0.006245810136788678, -0.006531257426789689, -0.005687390973087767, -0.003860122951814034, -0.0013654046040555497, 0.0013654046040555482, 0.0038601229518140322, 0.005687390973087765, 0.006531257426789689, 0.006245810136788678, 0.004880405532733127, 0.0026711344749756555, 7.441383276599842e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q1.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.15,
        },
        "q1.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q1.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q1.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q1.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q1.z.cz1_2.wf": {
            "type": "constant",
            "sample": -0.07009506167631502,
        },
        "q2.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.010461145880255703, 0.0400357563617998, 0.08361011261547184, 0.13364981258441283, 0.18150253754759008, 0.21889411663051164] + [0.23935920611790854] * 2 + [0.21889411663051164, 0.18150253754759013, 0.1336498125844129, 0.08361011261547187, 0.04003575636179977, 0.01046114588025569, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012501046533903815, -0.022840548553653223, -0.029230712228606546, -0.03056662020654985, -0.026617281861753765, -0.018065573672646034, -0.00639016367495332, 0.006390163674953312, 0.018065573672646027, 0.02661728186175375, 0.030566620206549846, 0.029230712228606546, 0.02284054855365322, 0.012501046533903815, 3.482605592825352e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005188727012159342, 0.01985773001012994, 0.041470605111854075, 0.06629028986543736, 0.09002523529722789, 0.1085714537168591] + [0.11872213547247179] * 2 + [0.1085714537168591, 0.09002523529722793, 0.0662902898654374, 0.04147060511185408, 0.019857730010129925, 0.005188727012159336, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0015234359102312821, -0.0027834559116028178, -0.0035621910989551164, -0.0037249910838078847, -0.003243706269788647, -0.0022015551735766023, -0.0007787351873522982, 0.0007787351873522972, 0.0022015551735766014, 0.0032437062697886457, 0.0037249910838078842, 0.003562191098955117, 0.0027834559116028173, 0.0015234359102312821, 4.244065812324945e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.005188727012159342, -0.01985773001012994, -0.041470605111854075, -0.06629028986543736, -0.09002523529722789, -0.1085714537168591] + [-0.11872213547247179] * 2 + [-0.1085714537168591, -0.09002523529722793, -0.0662902898654374, -0.04147060511185408, -0.019857730010129925, -0.005188727012159336, -5.197481612434455e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015234359102312828, 0.0027834559116028204, 0.0035621910989551216, 0.003724991083807893, 0.003243706269788658, 0.0022015551735766158, 0.0007787351873523127, -0.0007787351873522827, -0.002201555173576588, -0.003243706269788635, -0.003724991083807876, -0.0035621910989551116, -0.0027834559116028147, -0.0015234359102312815, -4.244065812324945e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.012501046533903815, 0.022840548553653226, 0.02923071222860655, 0.030566620206549856, 0.026617281861753776, 0.018065573672646048, 0.0063901636749533345, -0.006390163674953297, -0.018065573672646013, -0.02661728186175374, -0.03056662020654984, -0.029230712228606542, -0.022840548553653216, -0.012501046533903815, -3.482605592825352e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.010461145880255703, 0.0400357563617998, 0.08361011261547184, 0.13364981258441283, 0.18150253754759008, 0.21889411663051164] + [0.23935920611790854] * 2 + [0.21889411663051164, 0.18150253754759013, 0.1336498125844129, 0.08361011261547187, 0.04003575636179977, 0.01046114588025569, 2.1324808959731188e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0015234359102312823, 0.002783455911602819, 0.003562191098955119, 0.0037249910838078886, 0.0032437062697886526, 0.002201555173576609, 0.0007787351873523055, -0.00077873518735229, -0.002201555173576595, -0.00324370626978864, -0.0037249910838078803, -0.003562191098955114, -0.002783455911602816, -0.001523435910231282, -4.244065812324945e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.005188727012159342, 0.01985773001012994, 0.041470605111854075, 0.06629028986543736, 0.09002523529722789, 0.1085714537168591] + [0.11872213547247179] * 2 + [0.1085714537168591, 0.09002523529722793, 0.0662902898654374, 0.04147060511185408, 0.019857730010129925, 0.005188727012159336, 2.5987408062172275e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.001523435910231282, -0.0027834559116028165, -0.0035621910989551138, -0.0037249910838078808, -0.0032437062697886414, -0.002201555173576596, -0.0007787351873522909, 0.0007787351873523045, 0.002201555173576608, 0.0032437062697886513, 0.003724991083807888, 0.0035621910989551194, 0.0027834559116028186, 0.0015234359102312823, 4.244065812324945e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.005188727012159342, -0.01985773001012994, -0.041470605111854075, -0.06629028986543736, -0.09002523529722789, -0.1085714537168591] + [-0.11872213547247179] * 2 + [-0.1085714537168591, -0.09002523529722793, -0.0662902898654374, -0.04147060511185408, -0.019857730010129925, -0.005188727012159336, 2.5987408062172275e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q2.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q2.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q2.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q2.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q2.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q2.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q2.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q2.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.09775959661668145,
        },
        "q2.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005563561092243391, 0.02129225411275889, 0.04446644514760913, 0.07107910603595226, 0.09652866594175959, 0.11641466475532354] + [0.1272986326231512] * 2 + [0.11641466475532354, 0.09652866594175963, 0.07107910603595233, 0.04446644514760914, 0.02129225411275887, 0.005563561092243384, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.008269983263978223, -0.015110011291175274, -0.019337381095984943, -0.020221142034706617, -0.01760848381231923, -0.01195115877072839, -0.004227369804809668, 0.004227369804809662, 0.011951158770728387, 0.017608483812319223, 0.020221142034706614, 0.019337381095984946, 0.015110011291175271, 0.008269983263978223, 2.303894309135784e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.005559686906605402, 0.021277427252099614, 0.044435480939558375, 0.07102961010210704, 0.09646144820746218, 0.11633359940621195] + [0.1272099882232664] * 2 + [0.11633359940621195, 0.09646144820746223, 0.0710296101021071, 0.044435480939558396, 0.0212774272520996, 0.005559686906605395, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036246827501982067, -0.006622624923678124, -0.008475455083195447, -0.008862802061736062, -0.007717690047773467, -0.005238119311537855, -0.0018528301595173225, 0.0018528301595173203, 0.005238119311537853, 0.007717690047773464, 0.008862802061736062, 0.008475455083195449, 0.006622624923678123, 0.0036246827501982067, 1.0097826916988402e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q2.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.252757258898498,
        },
        "q2.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q2.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q2.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q2.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q2.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q3.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007766441642679343, 0.02972287825479078, 0.062072842478135645, 0.0992227316082989, 0.1347489923185278, 0.1625088113860774] + [0.17770226390412922] * 2 + [0.1625088113860774, 0.13474899231852785, 0.09922273160829898, 0.062072842478135666, 0.02972287825479076, 0.007766441642679332, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.012630769558612049, -0.023077564313600953, -0.029534038545678227, -0.030883809204890714, -0.02689348868197908, -0.018253039646278665, -0.0064564742320772734, 0.006456474232077266, 0.018253039646278658, 0.02689348868197907, 0.030883809204890714, 0.029534038545678234, 0.02307756431360095, 0.012630769558612049, 3.518744497687586e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.003852090915889313, 0.014742302149052635, 0.03078761724281289, 0.04921365545034744, 0.06683438737027161, 0.08060302837430647] + [0.08813885534867771] * 2 + [0.08060302837430647, 0.06683438737027164, 0.049213655450347484, 0.030787617242812898, 0.014742302149052625, 0.003852090915889308, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.007851291972546248, -0.014345024236290884, -0.018358371489126866, -0.019197389530923675, -0.01671700452001502, -0.011346097558377427, -0.004013347252835981, 0.004013347252835976, 0.011346097558377422, 0.016717004520015014, 0.019197389530923672, 0.01835837148912687, 0.01434502423629088, 0.007851291972546248, 2.1872531439938347e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.003852090915889312, -0.014742302149052634, -0.030787617242812888, -0.04921365545034744, -0.06683438737027161, -0.08060302837430647] + [-0.08813885534867771] * 2 + [-0.08060302837430647, -0.06683438737027164, -0.049213655450347484, -0.0307876172428129, -0.014742302149052627, -0.0038520909158893087, -2.6786125617170344e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007851291972546248, 0.014345024236290886, 0.01835837148912687, 0.019197389530923682, 0.016717004520015028, 0.011346097558377437, 0.004013347252835991, -0.004013347252835965, -0.011346097558377411, -0.016717004520015007, -0.019197389530923665, -0.018358371489126866, -0.014345024236290879, -0.007851291972546248, -2.1872531439938347e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.012630769558612049, 0.023077564313600956, 0.02953403854567823, 0.03088380920489072, 0.026893488681979088, 0.018253039646278675, 0.006456474232077285, -0.006456474232077254, -0.018253039646278647, -0.026893488681979064, -0.030883809204890707, -0.02953403854567823, -0.023077564313600946, -0.012630769558612049, -3.518744497687586e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.007766441642679342, 0.02972287825479078, 0.062072842478135645, 0.0992227316082989, 0.1347489923185278, 0.1625088113860774] + [0.17770226390412922] * 2 + [0.1625088113860774, 0.13474899231852785, 0.09922273160829898, 0.062072842478135666, 0.02972287825479076, 0.007766441642679333, 2.1546095930552317e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007851291972546248, 0.014345024236290886, 0.01835837148912687, 0.01919738953092368, 0.016717004520015025, 0.011346097558377432, 0.004013347252835986, -0.0040133472528359705, -0.011346097558377417, -0.01671700452001501, -0.01919738953092367, -0.018358371489126866, -0.014345024236290879, -0.007851291972546248, -2.1872531439938347e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038520909158893126, 0.014742302149052634, 0.03078761724281289, 0.04921365545034744, 0.06683438737027161, 0.08060302837430647] + [0.08813885534867771] * 2 + [0.08060302837430647, 0.06683438737027164, 0.049213655450347484, 0.030787617242812898, 0.014742302149052627, 0.0038520909158893083, 1.3393062808585172e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.007851291972546248, -0.014345024236290882, -0.018358371489126862, -0.019197389530923672, -0.016717004520015018, -0.011346097558377422, -0.004013347252835976, 0.004013347252835981, 0.011346097558377427, 0.016717004520015018, 0.019197389530923675, 0.018358371489126873, 0.014345024236290882, 0.007851291972546248, 2.1872531439938347e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0038520909158893135, -0.014742302149052637, -0.03078761724281289, -0.04921365545034744, -0.06683438737027161, -0.08060302837430647] + [-0.08813885534867771] * 2 + [-0.08060302837430647, -0.06683438737027164, -0.049213655450347484, -0.030787617242812898, -0.014742302149052623, -0.0038520909158893074, 1.3393062808585172e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q3.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q3.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q3.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q3.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q3.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q3.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q3.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q3.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.13625246720385348,
        },
        "q3.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.008708196739379495, 0.033327060629795464, 0.0695997664492379, 0.11125443383433788, 0.15108859237343839, 0.18221455417315943] + [0.19925035766784344] * 2 + [0.18221455417315943, 0.15108859237343844, 0.11125443383433796, 0.06959976644923793, 0.033327060629795444, 0.008708196739379484, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0038310543317542738, 0.014661793229020795, 0.030619483542288252, 0.048944895645327356, 0.06646939930438014, 0.08016284863167868] + [0.08765752183745126] * 2 + [0.08016284863167868, 0.06646939930438017, 0.0489448956453274, 0.030619483542288262, 0.014661793229020784, 0.003831054331754269, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0063272797074245945, -0.011560515271903887, -0.014794832721886832, -0.01547098918741978, -0.013472071072923433, -0.009143709479995186, -0.0032343174499829446, 0.0032343174499829407, 0.009143709479995182, 0.013472071072923426, 0.01547098918741978, 0.014794832721886836, 0.011560515271903885, 0.0063272797074245945, 1.7626859988630127e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q3.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.2214559913518434,
        },
        "q3.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q3.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q3.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q3.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q3.z.cz3_2.wf": {
            "type": "constant",
            "sample": -0.07709397834356477,
        },
        "q3.z.cz3_4.wf": {
            "type": "constant",
            "sample": -0.23305216356732317,
        },
        "q4.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009887219669888385, 0.03783928857605898, 0.07902304006806085, 0.12631742937029025, 0.17154482691635337, 0.20688500479252814] + [0.22622732518858696] * 2 + [0.20688500479252814, 0.17154482691635345, 0.12631742937029036, 0.07902304006806088, 0.03783928857605895, 0.009887219669888373, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00480253588690003, -0.008774669689286205, -0.01122958338702322, -0.011742800099581547, -0.010225579994932385, -0.006940264212681516, -0.0024549136977370144, 0.002454913697737011, 0.006940264212681514, 0.010225579994932382, 0.011742800099581546, 0.01122958338702322, 0.008774669689286203, 0.00480253588690003, 1.3379150532799059e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004903742013694642, 0.018767066511513116, 0.0391928787434334, 0.06264937021187782, 0.08508070044641718, 0.10260828866726192] + [0.11220145563788786] * 2 + [0.10260828866726192, 0.08508070044641722, 0.06264937021187787, 0.03919287874343341, 0.018767066511513105, 0.004903742013694635, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -9.365644863722523e-07, -1.7111884646294932e-06, -2.1899324116931283e-06, -2.290017149863633e-06, -1.9941371186700266e-06, -1.353452663491381e-06, -4.787439470636344e-07, 4.787439470636339e-07, 1.3534526634913805e-06, 1.9941371186700258e-06, 2.290017149863633e-06, 2.1899324116931283e-06, 1.7111884646294928e-06, 9.365644863722523e-07, 2.60912933124092e-21],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.004903742013694642, -0.018767066511513116, -0.0391928787434334, -0.06264937021187782, -0.08508070044641718, -0.10260828866726192] + [-0.11220145563788786] * 2 + [-0.10260828866726192, -0.08508070044641722, -0.06264937021187787, -0.03919287874343341, -0.018767066511513105, -0.004903742013694635, -3.195261884065667e-37],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 9.365644863728529e-07, 1.7111884646317914e-06, 2.189932411697928e-06, 2.2900171498713055e-06, 1.994137118680446e-06, 1.3534526635039469e-06, 4.787439470773751e-07, -4.787439470498932e-07, -1.3534526634788146e-06, -1.9941371186596064e-06, -2.2900171498559606e-06, -2.1899324116883285e-06, -1.7111884646271945e-06, -9.365644863716518e-07, -2.60912933124092e-21],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004802535886900031, 0.008774669689286207, 0.011229583387023226, 0.011742800099581554, 0.010225579994932395, 0.006940264212681529, 0.0024549136977370283, -0.002454913697736997, -0.006940264212681501, -0.010225579994932371, -0.011742800099581539, -0.011229583387023215, -0.008774669689286202, -0.004802535886900029, -1.3379150532799059e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.009887219669888385, 0.03783928857605898, 0.07902304006806085, 0.12631742937029025, 0.17154482691635337, 0.20688500479252814] + [0.22622732518858696] * 2 + [0.20688500479252814, 0.17154482691635345, 0.12631742937029036, 0.07902304006806088, 0.03783928857605895, 0.009887219669888373, 8.192366937651486e-34],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 9.365644863725526e-07, 1.7111884646306424e-06, 2.1899324116955283e-06, 2.2900171498674693e-06, 1.9941371186752363e-06, 1.3534526634976638e-06, 4.787439470705048e-07, -4.787439470567635e-07, -1.3534526634850976e-06, -1.994137118664816e-06, -2.290017149859797e-06, -2.189932411690728e-06, -1.7111884646283435e-06, -9.365644863719521e-07, -2.60912933124092e-21],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.004903742013694642, 0.018767066511513116, 0.0391928787434334, 0.06264937021187782, 0.08508070044641718, 0.10260828866726192] + [0.11220145563788786] * 2 + [0.10260828866726192, 0.08508070044641722, 0.06264937021187787, 0.03919287874343341, 0.018767066511513105, 0.004903742013694635, 1.5976309420328335e-37],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -9.365644863719521e-07, -1.711188464628344e-06, -2.189932411690728e-06, -2.290017149859797e-06, -1.994137118664817e-06, -1.353452663485098e-06, -4.787439470567641e-07, 4.787439470705043e-07, 1.3534526634976634e-06, 1.9941371186752355e-06, 2.2900171498674693e-06, 2.1899324116955283e-06, 1.711188464630642e-06, 9.365644863725526e-07, 2.60912933124092e-21],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.004903742013694642, -0.018767066511513116, -0.0391928787434334, -0.06264937021187782, -0.08508070044641718, -0.10260828866726192] + [-0.11220145563788786] * 2 + [-0.10260828866726192, -0.08508070044641722, -0.06264937021187787, -0.03919287874343341, -0.018767066511513105, -0.004903742013694635, 1.5976309420328335e-37],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q4.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q4.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q4.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q4.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q4.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q4.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q4.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q4.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.14844517355097134,
        },
        "q4.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009681443178495789, 0.03705176323527111, 0.07737838318095787, 0.12368846407110583, 0.16797457220593126, 0.20257923716226092] + [0.22151899799563357] * 2 + [0.20257923716226092, 0.16797457220593132, 0.12368846407110592, 0.0773783831809579, 0.03705176323527108, 0.009681443178495777, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.012968181353609854, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.00017685586774239802, -0.00032313174926701664, -0.00041353521578358374, -0.0004324346866416748, -0.0003765622716336233, -0.0002555788188992768, -9.040346651656707e-05, 9.040346651656696e-05, 0.0002555788188992767, 0.00037656227163362314, 0.00043243468664167477, 0.0004135352157835838, 0.0003231317492670166, 0.00017685586774239802, 4.926941376093873e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q4.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.1436923076923077,
        },
        "q4.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q4.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q4.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q4.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q4.z.cz.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.xy.x180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.007448136311174729, 0.028504694812302632, 0.05952880524571009, 0.09515611707226578, 0.12922634441319125, 0.15584843544294205] + [0.17041918876836984] * 2 + [0.15584843544294205, 0.1292263444131913, 0.09515611707226584, 0.0595288052457101, 0.02850469481230261, 0.00744813631117472, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.009243514293683218, -0.016888740991297516, -0.02161375094212127, -0.022601547000289227, -0.019681330253498677, -0.013358032706606009, -0.004725009950823751, 0.004725009950823745, 0.013358032706606003, 0.01968133025349867, 0.022601547000289223, 0.02161375094212127, 0.016888740991297512, 0.009243514293683218, 2.5751055713004787e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.0036795564274120533, 0.014081996975627937, 0.029408645170655546, 0.04700938430134693, 0.06384088667431032, 0.07699283261986167] + [0.08419113120233718] * 2 + [0.07699283261986167, 0.06384088667431034, 0.04700938430134697, 0.029408645170655553, 0.014081996975627929, 0.003679556427412049, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0002678862941982928, -0.0004894526144591197, -0.0006263881311425552, -0.0006550154493939186, -0.0005703846458165272, -0.00038712915519562566, -0.00013693551668343546, 0.0001369355166834353, 0.00038712915519562556, 0.0005703846458165269, 0.0006550154493939183, 0.0006263881311425552, 0.0004894526144591196, 0.0002678862941982928, 7.462913635958554e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036795564274120533, -0.014081996975627937, -0.029408645170655546, -0.04700938430134693, -0.06384088667431032, -0.07699283261986167] + [-0.08419113120233718] * 2 + [-0.07699283261986167, -0.06384088667431034, -0.04700938430134697, -0.029408645170655553, -0.014081996975627929, -0.003679556427412049, -9.13943329658978e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-x90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0002678862941982932, 0.0004894526144591214, 0.0006263881311425588, 0.0006550154493939243, 0.000570384645816535, 0.0003871291551956351, 0.00013693551668344576, -0.000136935516683425, -0.0003871291551956161, -0.0005703846458165191, -0.0006550154493939126, -0.0006263881311425516, -0.0004894526144591178, -0.00026788629419829236, -7.462913635958554e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.009243514293683218, 0.01688874099129752, 0.021613750942121274, 0.022601547000289234, 0.019681330253498684, 0.013358032706606019, 0.004725009950823761, -0.0047250099508237345, -0.013358032706605993, -0.019681330253498663, -0.022601547000289216, -0.021613750942121267, -0.01688874099129751, -0.009243514293683218, -2.5751055713004787e-17],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y180_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0074481363111747285, 0.028504694812302632, 0.05952880524571009, 0.09515611707226578, 0.12922634441319125, 0.15584843544294205] + [0.17041918876836984] * 2 + [0.15584843544294205, 0.1292263444131913, 0.09515611707226584, 0.0595288052457101, 0.02850469481230261, 0.007448136311174721, 1.576797397679824e-33],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.000267886294198293, 0.0004894526144591205, 0.0006263881311425571, 0.0006550154493939215, 0.0005703846458165311, 0.0003871291551956304, 0.0001369355166834406, -0.00013693551668343015, -0.00038712915519562084, -0.000570384645816523, -0.0006550154493939154, -0.0006263881311425534, -0.0004894526144591187, -0.0002678862941982926, -7.462913635958554e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, 0.0036795564274120533, 0.014081996975627937, 0.029408645170655546, 0.04700938430134693, 0.06384088667431032, 0.07699283261986167] + [0.08419113120233718] * 2 + [0.07699283261986167, 0.06384088667431034, 0.04700938430134697, 0.029408645170655553, 0.014081996975627929, 0.003679556427412049, 4.56971664829489e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, -0.0002678862941982926, -0.0004894526144591188, -0.0006263881311425534, -0.0006550154493939156, -0.0005703846458165233, -0.00038712915519562095, -0.0001369355166834303, 0.00013693551668344045, 0.00038712915519563027, 0.0005703846458165308, 0.0006550154493939213, 0.0006263881311425571, 0.0004894526144591204, 0.000267886294198293, 7.462913635958554e-19],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.-y90_DragCosine.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0036795564274120533, -0.014081996975627937, -0.029408645170655546, -0.04700938430134693, -0.06384088667431032, -0.07699283261986167] + [-0.08419113120233718] * 2 + [-0.07699283261986167, -0.06384088667431034, -0.04700938430134697, -0.029408645170655553, -0.014081996975627929, -0.003679556427412049, 4.56971664829489e-35],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.x180_Square.wf.I": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.xy.x180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.x90_Square.wf.I": {
            "type": "constant",
            "sample": 0.05,
        },
        "q5.xy.x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.-x90_Square.wf.I": {
            "type": "constant",
            "sample": -0.125,
        },
        "q5.xy.-x90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.y180_Square.wf.I": {
            "type": "constant",
            "sample": -0.11201840403229253,
        },
        "q5.xy.y180_Square.wf.Q": {
            "type": "constant",
            "sample": 0.22349916590013946,
        },
        "q5.xy.y90_Square.wf.I": {
            "type": "constant",
            "sample": -0.056009202016146266,
        },
        "q5.xy.y90_Square.wf.Q": {
            "type": "constant",
            "sample": 0.11174958295006973,
        },
        "q5.xy.-y90_Square.wf.I": {
            "type": "constant",
            "sample": 0.056009202016146266,
        },
        "q5.xy.-y90_Square.wf.Q": {
            "type": "constant",
            "sample": -0.11174958295006973,
        },
        "q5.xy.saturation.wf.I": {
            "type": "constant",
            "sample": 0.12782954309787248,
        },
        "q5.xy.saturation.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.xy.EF_x180.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.01296818135360987, 0.04963040904617126, 0.10364745084375789, 0.165679269490148, 0.22499999999999995, 0.27135254915624213] + [0.29672214011007086] * 2 + [0.27135254915624213, 0.22500000000000006, 0.16567926949014813, 0.10364745084375791, 0.04963040904617123, 0.012968181353609854, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x180.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0] * 16,
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x90.wf.I": {
            "type": "arbitrary",
            "samples": [0.0, 0.004618343160591265, 0.017674819153568368, 0.03691184466538491, 0.05900316322215746, 0.08012898515209145, 0.09663646392143418] + [0.10567130648522964] * 2 + [0.09663646392143418, 0.08012898515209148, 0.059003163222157505, 0.03691184466538492, 0.017674819153568357, 0.004618343160591259, 0.0],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.xy.EF_x90.wf.Q": {
            "type": "arbitrary",
            "samples": [0.0, -0.0008065355175237464, -0.0014736137169224855, -0.0018858907173049866, -0.001972080079886148, -0.0017172788808699087, -0.0011655445623624013, -0.0004122770003825005, 0.00041227700038250003, 0.001165544562362401, 0.001717278880869908, 0.001972080079886148, 0.0018858907173049866, 0.0014736137169224855, 0.0008065355175237464, 2.246887967757484e-18],
            "is_overridable": False,
            "max_allowed_error": 0.0001,
        },
        "q5.resonator.readout.wf.I": {
            "type": "constant",
            "sample": 0.136350332271279,
        },
        "q5.resonator.readout.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.resonator.const.wf.I": {
            "type": "constant",
            "sample": 0.125,
        },
        "q5.resonator.const.wf.Q": {
            "type": "constant",
            "sample": 0.0,
        },
        "q5.z.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "q5.z.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "q5.z.cz5_4.wf": {
            "type": "constant",
            "sample": -0.1732650364352627,
        },
        "coupler_q1_q2.const.wf": {
            "type": "constant",
            "sample": 0.25,
        },
        "coupler_q1_q2.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q1_q2.cz.wf": {
            "type": "constant",
            "sample": -0.04662292425886875,
        },
        "coupler_q2_q3.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "coupler_q2_q3.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q2_q3.cz.wf": {
            "type": "constant",
            "sample": -0.10043328056510244,
        },
        "coupler_q3_q4.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "coupler_q3_q4.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q3_q4.cz.wf": {
            "type": "constant",
            "sample": -0.11048567109,
        },
        "coupler_q4_q5.const.wf": {
            "type": "constant",
            "sample": 0.1,
        },
        "coupler_q4_q5.flux_pulse.wf": {
            "type": "constant",
            "sample": 0.2,
        },
        "coupler_q4_q5.cz.wf": {
            "type": "constant",
            "sample": -0.09938541439500001,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "q1.resonator.readout.iw1": {
            "cosine": [(-0.6829156716957551, 1040)],
            "sine": [(0.7304972178950004, 1040)],
        },
        "q1.resonator.readout.iw2": {
            "cosine": [(-0.7304972178950004, 1040)],
            "sine": [(-0.6829156716957551, 1040)],
        },
        "q1.resonator.readout.iw3": {
            "cosine": [(0.7304972178950004, 1040)],
            "sine": [(0.6829156716957551, 1040)],
        },
        "q2.resonator.readout.iw1": {
            "cosine": [(0.6031753077014853, 1040)],
            "sine": [(0.7976086434958052, 1040)],
        },
        "q2.resonator.readout.iw2": {
            "cosine": [(-0.7976086434958052, 1040)],
            "sine": [(0.6031753077014853, 1040)],
        },
        "q2.resonator.readout.iw3": {
            "cosine": [(0.7976086434958052, 1040)],
            "sine": [(-0.6031753077014853, 1040)],
        },
        "q3.resonator.readout.iw1": {
            "cosine": [(-0.3203852993033521, 1040)],
            "sine": [(-0.947287316493946, 1040)],
        },
        "q3.resonator.readout.iw2": {
            "cosine": [(0.947287316493946, 1040)],
            "sine": [(-0.3203852993033521, 1040)],
        },
        "q3.resonator.readout.iw3": {
            "cosine": [(-0.947287316493946, 1040)],
            "sine": [(0.3203852993033521, 1040)],
        },
        "q4.resonator.readout.iw1": {
            "cosine": [(-0.9722269651436221, 1040)],
            "sine": [(0.2340400142018932, 1040)],
        },
        "q4.resonator.readout.iw2": {
            "cosine": [(-0.2340400142018932, 1040)],
            "sine": [(-0.9722269651436221, 1040)],
        },
        "q4.resonator.readout.iw3": {
            "cosine": [(0.2340400142018932, 1040)],
            "sine": [(0.9722269651436221, 1040)],
        },
        "q5.resonator.readout.iw1": {
            "cosine": [(0.5461865405282237, 1040)],
            "sine": [(0.837663573844423, 1040)],
        },
        "q5.resonator.readout.iw2": {
            "cosine": [(-0.837663573844423, 1040)],
            "sine": [(0.5461865405282237, 1040)],
        },
        "q5.resonator.readout.iw3": {
            "cosine": [(0.837663573844423, 1040)],
            "sine": [(-0.5461865405282237, 1040)],
        },
    },
    "mixers": {},
}

