# %%
from get_quam import QuAM
from quam_libs.quam_builder.superconducting import build_quam

path = "./quam_state"

machine = QuAM.load(path)


# Make the QuAM object and save it
quam = build_quam(machine, quam_state_path=path)

# %%
