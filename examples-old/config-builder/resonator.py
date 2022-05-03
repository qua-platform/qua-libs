import numpy as np

from qualang_tools.config.configuration import *
from qualang_tools.config.components import *
from qualang_tools.config.builder import ConfigBuilder

# Initialize ConfigBuilder object
cb = ConfigBuilder()

# Adding controller
cont = Controller("con1")
cb.add(cont)

# Setting properties of readout resonator object and the operations it supports
res = ReadoutResonator(
    "res1",
    outputs=[cont.analog_output(0), cont.analog_output(1)],
    inputs=[cont.analog_input(0), cont.analog_input(1)],
    intermediate_frequency=2e6,
)
res.lo_frequency = 4e9

wfs = [
    ArbitraryWaveform("wf1", np.linspace(0, -0.5, 16)),
    ArbitraryWaveform("wf2", np.linspace(0, -0.5, 16)),
]

ro_pulse = MeasurePulse("ro_pulse", wfs, 16)
ro_pulse.add(Weights(ConstantIntegrationWeights("cos", cosine=1, sine=0, duration=16)))
ro_pulse.add(Weights(ConstantIntegrationWeights("minus_sin", cosine=0, sine=-1, duration=16)))
ro_pulse.add(Weights(ConstantIntegrationWeights("sin", cosine=0, sine=1, duration=16)))
res.add(Operation(ro_pulse))

# Adding resonator to the builder
cb.add(res)

# Build the QUA configuration
print(cb.build())
