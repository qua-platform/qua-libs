import numpy as np

from qualang_tools.config.configuration import *
from qualang_tools.config.components import *
from qualang_tools.config.builder import ConfigBuilder

cont = Controller("con1")

res = ReadoutResonator(
    "res1",
    outputs=[cont.analog_output(0), cont.analog_output(1)],
    inputs=[cont.analog_input(0), cont.analog_input(1)],
    intermediate_frequency=2e6,
)
res.lo_frequency = 4e9

wfs = [
    ArbitraryWaveform("wf1", np.linspace(0, -0.5, 16).tolist()),
    ArbitraryWaveform("wf2", np.linspace(0, -0.5, 16).tolist()),
]

ro_pulse = MeasurePulse("ro_pulse", wfs, 16)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w1_I", cosine=1, sine=0, duration=16))
)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w1_Q", cosine=0, sine=-1, duration=16))
)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w2_I", cosine=0, sine=1, duration=16))
)
ro_pulse.add(
    Weights(ConstantIntegrationWeights("integ_w2_Q", cosine=1, sine=0, duration=16))
)

res.add(Operation(ro_pulse))

cb = ConfigBuilder()
cb.add(cont)

# here the two wfs are already added to resonator object, but still it is possible to add them to the setup
# and they should appear in the config
# cb.add(wfs[0])
# cb.add(wfs[1])

cb.add(res)

print(cb.build())
