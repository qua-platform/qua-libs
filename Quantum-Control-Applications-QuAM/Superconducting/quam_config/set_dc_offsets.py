# %%

from quam_config import QuAM
from quam_builder.builder.superconducting.build_quam import save_machine
import json

with open("/Users/paul/QM/qw_statestore_QCC/quam_state/state.json", "r") as file:
    iqqc_machine = json.load(file)

# %%
machine = QuAM.load()


qubit_list = [qubit for qubit in iqqc_machine["qubits"]]

saturation_amps = {f"q{ii+1}":  iqqc_machine["qubits"][qubit]["xy"]["operations"]["saturation"]["amplitude"] for ii, qubit in enumerate(qubit_list)}
# %%


for key in saturation_amps:
    machine.qubits[key].xy.operations["saturation"].amplitude = saturation_amps[key]
    # print(machine.qubits[key].xy.operations["saturation"].amplitude)
# %%


save_machine(machine)

# %%


config = machine.generate_config()

with open("config_debug.json", "w") as outfile:
    json.dump(config, outfile, indent=4)

# %%
