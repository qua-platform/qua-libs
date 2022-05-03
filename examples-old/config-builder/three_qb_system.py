import numpy as np

from qualang_tools.config.configuration import *
from qualang_tools.config.components import *
from qualang_tools.config.builder import ConfigBuilder

# A sample configuration of a setup with 3 transmons connected to each other via capacitive couplers
# and 3 readout resonators

# All the waveform objects used in building the configuration
cnot_coupler = ArbitraryWaveform("cnot_coupler", np.random.rand(16))
rxpio2_I = ArbitraryWaveform("rxpio2_I", np.random.rand(16))
rxpio2_Q = ArbitraryWaveform("rxpio2_Q", np.random.rand(16))
cnot_I1 = ArbitraryWaveform("cnot_I1", np.random.rand(16))
cnot_Q1 = ArbitraryWaveform("cnot_Q1", np.random.rand(16))
cnot_I2 = ArbitraryWaveform("cnot_I2", np.random.rand(16))
cnot_Q2 = ArbitraryWaveform("cnot_Q2", np.random.rand(16))

cb = ConfigBuilder()

con1 = Controller("con1")
con2 = Controller("con2")
con3 = Controller("con3")

# Add the controllers
cb.add(con1)
cb.add(con2)
cb.add(con3)

# Define the qubits and the operations that they support
qb1 = Transmon("qb1", I=con1.analog_output(1), Q=con1.analog_output(2), intermediate_frequency=50e6)
qb1.lo_frequency = 4.8e9

qb2 = Transmon("qb2", I=con2.analog_output(1), Q=con2.analog_output(2), intermediate_frequency=50e6)
qb2.lo_frequency = 4.8e9

qb3 = Transmon("qb3", I=con3.analog_output(1), Q=con3.analog_output(2), intermediate_frequency=50e6)
qb3.lo_frequency = 4.8e9

# Adding the qubits to the ConfigBuilder
cb.add(qb1)
cb.add(qb2)
cb.add(qb3)

rxpio2_pulse = ControlPulse("rxpio2", [rxpio2_I, rxpio2_Q], 16)
qb1.add(Operation(rxpio2_pulse))
qb2.add(Operation(rxpio2_pulse))
qb3.add(Operation(rxpio2_pulse))

# The couplers connecting the qubits
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

# The readout resonators
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
ro_pulse.add(Weights(ConstantIntegrationWeights("cos", cosine=1, sine=0, duration=100)))
ro_pulse.add(Weights(ConstantIntegrationWeights("minus_sin", cosine=0, sine=-1, duration=100)))
ro_pulse.add(Weights(ConstantIntegrationWeights("sin", cosine=0, sine=1, duration=100)))

# Note that here we add the operations to the resonators, after adding resonators to the ConfigBuilder
# but these operations appear in the final configuration
res1.add(Operation(ro_pulse))
res2.add(Operation(ro_pulse))
res3.add(Operation(ro_pulse))

# Build the QUA configuration
print(cb.build())
