import numpy as np

from qualang_tools.config.configuration import *
from qualang_tools.config.components import *
from qualang_tools.config.builder import ConfigBuilder
from qualang_tools.config.parameters import ConfigVar

p = ConfigVar()

cb = ConfigBuilder()

cont = Controller("con1")
cb.add(cont)

wf1 = ArbitraryWaveform("wf1", p.parameter("wf1_samples"))
wf2 = ArbitraryWaveform("wf2", np.linspace(0, -0.5, 16))

qb1 = Transmon("qb1", I=cont.analog_output(0), Q=cont.analog_output(1), intermediate_frequency=5e6)
qb1.lo_frequency = 4e9
qb1.add(Operation(ControlPulse("pi_pulse", [wf1, wf2], 16)))

cb.add(qb1)

p.set(wf1_samples=np.linspace(0, -0.5, 16))

print(cb.build())
