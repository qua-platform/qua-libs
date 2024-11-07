# %%
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import build_quam

path = "./quam_state"

machine = QuAM.load(path)

# octave_settings = {"octave1": {"port": 11235} }  # externally configured: (11XXX where XXX are last three digits of oct ip)
# octave_settings = {"octave1": {"ip": "192.168.88.250"} }  # "internally" configured: use the local ip address of the Octave
octave_settings = {}

# Make the QuAM object and save it
quam = build_quam(machine, quam_state_path=path, octaves_settings=octave_settings)

# %%
