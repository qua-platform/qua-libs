# check if config is correct by trying to open quantum machine
# add optionally your own simulator output for pulses

from quam import QuAM
from rich import print

machine = QuAM("quam_bootstrap_state.json")
print(machine.build_config())
