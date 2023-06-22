from quam import QuAM
from configuration import build_config

machine = QuAM("quam_bootstrap_state.json", flat_data=False)

# here do do some measurements ...
config = build_config(machine)
# ...

# ... and update value based on that
# machine.qubit1.freq = 5.033
#
# machine._save("quam_bootstrap_state.json", flat_data=False)
