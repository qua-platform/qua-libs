import numpy as np

from qualang_tools.config.configuration import *
from qualang_tools.config.components import *
from qualang_tools.config.builder import ConfigBuilder

cnot_coupler = ArbitraryWaveform(
    "cnot_coupler",
    (
        2.0
        * np.pi
        * np.array(
            [
                -0.0022640274687561295,
                0.17980755683999214,
                0.390545132044947,
                0.35038324027031814,
                0.19550416635358,
                0.024658258821028484,
                -0.12822902284043466,
                -0.1752966912118911,
                -0.17921353908242568,
                -0.15096289009237282,
                -0.110322391784403,
                -0.08696095347060069,
                -0.0801578668706695,
                -0.07016164311782273,
                -0.053739705459874915,
                -0.035772043817697045,
            ]
        )
    ).tolist(),
)

rxpio2_I = ArbitraryWaveform(
    "rxpio2_I",
    (
        2.0
        * np.pi
        * np.array(
            [
                -0.016742025698504664,
                0.15475277233533596,
                0.3463070972101407,
                0.15726323682389567,
                0.10395169410442902,
                0.09411013144760073,
                0.09673361659417648,
                0.10203553196465118,
                0.10571243149509368,
                0.10553012236249742,
                0.10053283342900529,
                0.09062793974037282,
                0.07611822737705624,
                0.05743359982906475,
                0.03524864262704943,
                0.010738888245412562,
            ]
        )
    ).tolist(),
)
rxpio2_Q = ArbitraryWaveform(
    "rxpio2_Q",
    (
        2.0
        * np.pi
        * np.array(
            [
                0.0013861834828884864,
                0.45019309817567754,
                -0.1529850245143982,
                -0.2789586999181469,
                -0.2700135047261444,
                -0.24916771046102235,
                -0.22324002778043853,
                -0.19249833290909127,
                -0.15822974653694788,
                -0.1227810647445817,
                -0.08884979485369848,
                -0.05873179586049876,
                -0.03397619024264336,
                -0.01553821078627357,
                -0.004036814925800248,
                0.00027771463195008854,
            ]
        )
    ).tolist(),
)

cnot_I1 = ArbitraryWaveform(
    "cnot_I1",
    (
        2.0
        * np.pi
        * np.array(
            [
                0.004550030418978573,
                -0.06098227204421411,
                0.36957216345202887,
                0.29532398495147905,
                0.26477074638298104,
                0.3062366161339038,
                0.22502162570719328,
                0.1967487786023379,
                0.18801232158046996,
                0.17578459439160232,
                0.10929652583018863,
                0.0434285767839821,
                0.030467828001564386,
                0.02611384802414898,
                0.01982699565942382,
                0.012127833300662735,
            ]
        )
    ).tolist(),
)
cnot_Q1 = ArbitraryWaveform(
    "cnot_Q1",
    (
        -2.0
        * np.pi
        * np.array(
            [
                -9.897728585998266e-5,
                0.008034988293530495,
                -0.13374179184241197,
                -0.19647084682136146,
                -0.35146294139611556,
                -0.2528581565333187,
                -0.1250757901191075,
                -0.08132008047619026,
                -0.06529351116426077,
                -0.06738991440713618,
                -0.060797866422997615,
                -0.026609361647279173,
                -0.012496635056721881,
                -0.005936085984044713,
                -0.0021843702393362645,
                -0.00042162333083034455,
            ]
        )
    ).tolist(),
)

cnot_I2 = ArbitraryWaveform(
    "cnot_I2",
    (
        2.0
        * np.pi
        * np.array(
            [
                -0.006857635028247913,
                0.21253808896116097,
                -0.020088639336504545,
                -0.12584578959791412,
                -0.11347780637171505,
                -0.11644741135448118,
                -0.0823775265689421,
                -0.05812226313850513,
                -0.05239180082948136,
                -0.07733406271074851,
                -0.12683149973163313,
                -0.11575712587511292,
                -0.06723466515593918,
                -0.030842674949357687,
                -0.008387866578188683,
                0.006500701318770028,
            ]
        )
    ).tolist(),
)
cnot_Q2 = ArbitraryWaveform(
    "cnot_Q2",
    (
        -2.0
        * np.pi
        * np.array(
            [
                -0.00021459250417936717,
                -0.08378057074732273,
                -0.15420784656242895,
                0.020025773137018886,
                0.03817220075116107,
                0.0783367771695887,
                0.14219130502814858,
                0.16903873908509154,
                0.1808110064928055,
                0.17371305493458056,
                0.11472706694365807,
                0.045723392279059444,
                0.02312523234819297,
                0.009823882427570495,
                0.0019644568243603654,
                -0.0008139980959453693,
            ]
        )
    ).tolist(),
)

cb = ConfigBuilder()

con1 = Controller("con1")
con2 = Controller("con2")
con3 = Controller("con3")

cb.add(con1)
cb.add(con2)
cb.add(con3)

qb1 = Transmon(
    "qb1", I=con1.analog_output(1), Q=con1.analog_output(2), intermediate_frequency=50e6
)
qb1.lo_frequency = 4.8e9

qb2 = Transmon(
    "qb2", I=con2.analog_output(1), Q=con2.analog_output(2), intermediate_frequency=50e6
)
qb2.lo_frequency = 4.8e9

qb3 = Transmon(
    "qb3", I=con3.analog_output(1), Q=con3.analog_output(2), intermediate_frequency=50e6
)
qb3.lo_frequency = 4.8e9

cb.add(qb1)
cb.add(qb2)
cb.add(qb3)

rxpio2_pulse = ControlPulse("rxpio2", [rxpio2_I, rxpio2_Q], 16)
qb1.add(Operation(rxpio2_pulse))
qb2.add(Operation(rxpio2_pulse))
qb3.add(Operation(rxpio2_pulse))

cc12 = Coupler("cc12", p=con1.analog_output(5))
cc23 = Coupler("cc23", p=con2.analog_output(5))
cc31 = Coupler("cc31", p=con3.analog_output(5))

cb.add(cc12)
cb.add(cc23)
cb.add(cc31)

cnot_coupler_pulse = ControlPulse("cnot_coupler", [cnot_coupler], 16)
cc12.add(Operation(cnot_coupler_pulse))
cc23.add(Operation(cnot_coupler_pulse))
cc31.add(Operation(cnot_coupler_pulse))

cnot_c_pulse = ControlPulse("cnot_c", [cnot_I1, cnot_Q1], 16)
cnot_t_pulse = ControlPulse("cnot_t", [cnot_I2, cnot_Q2], 16)

qb1.add(Operation(cnot_c_pulse, "cnot_12"))
qb1.add(Operation(cnot_c_pulse, "cnot_13"))
qb1.add(Operation(cnot_t_pulse, "cnot_21"))
qb1.add(Operation(cnot_t_pulse, "cnot_31"))

qb2.add(Operation(cnot_c_pulse, "cnot_21"))
qb2.add(Operation(cnot_c_pulse, "cnot_23"))
qb2.add(Operation(cnot_t_pulse, "cnot_12"))
qb2.add(Operation(cnot_t_pulse, "cnot_32"))

qb3.add(Operation(cnot_c_pulse, "cnot_31"))
qb3.add(Operation(cnot_c_pulse, "cnot_32"))
qb3.add(Operation(cnot_t_pulse, "cnot_13"))
qb3.add(Operation(cnot_t_pulse, "cnot_23"))

res1 = ReadoutResonator(
    "rr1",
    intermediate_frequency=50e6,
    outputs=[con1.analog_output(3), con1.analog_output(4)],
    inputs=[con1.analog_input(1), con1.analog_input(2)],
)
res1.lo_frequency = 5e9
res1.time_of_flight = 24

res2 = ReadoutResonator(
    "rr2",
    intermediate_frequency=50e6,
    outputs=[con2.analog_output(3), con2.analog_output(4)],
    inputs=[con2.analog_input(1), con2.analog_input(2)],
)
res2.lo_frequency = 5e9
res2.time_of_flight = 24

res3 = ReadoutResonator(
    "rr3",
    intermediate_frequency=50e6,
    outputs=[con3.analog_output(3), con3.analog_output(4)],
    inputs=[con3.analog_input(1), con3.analog_input(2)],
)
res3.lo_frequency = 5e9
res3.time_of_flight = 24

cb.add(res1)
cb.add(res2)
cb.add(res3)

ro_I = ConstantWaveform("ro_I", 0.01)
ro_Q = ConstantWaveform("ro_Q", 0.0)

ro_pulse = MeasurePulse("ro_pulse", [ro_I, ro_Q], 100)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w1_I", cosine=1, sine=0, duration=100))
)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w1_Q", cosine=0, sine=-1, duration=100))
)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w2_I", cosine=0, sine=1, duration=100))
)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w2_Q", cosine=1, sine=0, duration=100))
)

res1.add(Operation(ro_pulse))
res2.add(Operation(ro_pulse))
res3.add(Operation(ro_pulse))

print(cb.build())
