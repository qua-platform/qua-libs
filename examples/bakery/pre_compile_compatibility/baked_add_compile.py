from qualang_tools import baking
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from config import config

# Create a first baked waveform, overridable
# and which will be inserted in the config
with baking(config, padding_method="right", override=True) as b_template:

    samples_I = [0.1, 0.1, 0.2, 0.1, 0.2]
    samples_Q = [0.2, 0.2, 0.3, 0.1, 0.0]
    b_template.add_op("Op", "qe1", [samples_I, samples_Q])
    b_template.play("Op", "qe1")

# Re-open the context manager with either same baking object (b_template) or a new one (b_new) to generate a
# new waveform
# Only important thing is to indicate which baking index it shall use to generate the right name to override waveform
# Note that override parameter and update_config are not relevant anymore (since program is already compiled with a
# previous config, as we only want to retrieve the waveforms out
# of this new baking object
with baking(
    config,
    padding_method="right",
    override=False,
    baking_index=b_template.get_baking_index(),
) as b_new:
    samples_I = [0.3, 0.3, 0.4]
    samples_Q = [0.0, 0.1, 0.2]
    b_new.add_op("Op", "qe1", [samples_I, samples_Q])
    b_new.play("Op", "qe1")

print(b_template.get_waveforms_dict())
print(b_new.get_waveforms_dict())
qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

with program() as prog:
    b_template.run()

pid = qm.queue.compile(prog)
pjob = qm.queue.add_compiled(prog, overrides={b_new.get_waveforms_dict()})
job = qm.queue.wait_for_execution(pjob)
job.results_handles.wait_for_all_values()
