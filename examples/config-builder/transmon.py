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
wf2 = ArbitraryWaveform("wf2", np.linspace(0, -0.5, 16).tolist())

qb1 = Transmon(
    "qb1", I=cont.analog_output(0), Q=cont.analog_output(1), intermediate_frequency=5e6
)
qb1.lo_frequency = 4e9
qb1.add(Operation(ControlPulse("pi_pulse", [wf1, wf2], 16)))

cb.add(qb1)

qb2 = FluxTunableTransmon(
    "qb2",
    I=cont.analog_output(2),
    Q=cont.analog_output(3),
    fl_port=cont.analog_output(4),
    intermediate_frequency=5e6,
)
qb2.lo_frequency = 4.5e9
qb2.add(Operation(ControlPulse("pi_pulse", [wf1, wf2], 16)))
qb2.add(Operation(ControlPulse("fl_pulse", [wf1], 16)))

cb.add(qb2)

p.set(wf1_samples=np.linspace(0, -0.5, 16).tolist())

qb1.mixer = Mixer(
    "mx1",
    intermediate_frequency=5e6,
    lo_frequency=4e9,
    correction=Matrix2x2([[1.0, 0.0], [1.0, 0.0]]),
)

print(cb.build())

for p in cb.ports:
    print(p)

objs = cb.find_users_of(cont.analog_output(0))
print([obj.name for obj in objs])

objs = cb.find_users_of(cont)
print([obj.name for obj in objs])

objs = cb.find_users_of(wf1)
print([obj.name for obj in objs])

objs = cb.find_users_of(ControlPulse("pi_pulse", [wf1, wf2], 16))
print([obj.name for obj in objs])
