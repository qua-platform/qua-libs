import pygsti
from pygsti.objects import Circuit, Model, DataSet, Label
from qm.qua import *
from pygsti.construction import make_lsgst_experiment_list
from pygsti.modelpacks import smq1Q_XYI
from encode_circuits import *

model = smq1Q_XYI

prep_fiducials, meas_fiducials, germs, basic_gates = (
    model.prep_fiducials(),
    model.meas_fiducials(),
    model.germs(),
    model.gates,
)

gate_sequences = list(
    {k.str.split("@")[0] for k in prep_fiducials + germs + meas_fiducials}
)
gate_sequences.sort(key=len, reverse=True)

GST_sequence_file = "Circuits_before_results.txt"
with open(file=GST_sequence_file, mode="r") as f:
    circuits = f.readlines()
    circ_list = encode_circuits(circuits, model)

with open(file=GST_sequence_file, mode="r") as f:
    circuits = f.readlines()
    for i, circ in enumerate(circuits):
        c = circ.rstrip()
        gates, qubit_measures = c.split("@")
        generated_gates = ""
        generated_gates += (
            gate_sequences[circ_list[i][0]] if circ_list[i][0] >= 0 else ""
        )
        generated_gates += (
            "(" + gate_sequences[circ_list[i][2]] + ")" if circ_list[i][2] >= 0 else ""
        )
        generated_gates += "^" + str(circ_list[i][3]) if circ_list[i][3] > 1 else ""
        generated_gates += (
            gate_sequences[circ_list[i][1]] if circ_list[i][1] >= 0 else ""
        )
        if generated_gates == "":
            generated_gates = "{}"
        assert gates == generated_gates
