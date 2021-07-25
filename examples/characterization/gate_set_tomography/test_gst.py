import pygsti
from pygsti.objects import Circuit, Model, DataSet, Label
from qm.qua import *
from pygsti.construction import make_lsgst_experiment_list
from pygsti.modelpacks import smq1Q_XYI
mdl_ideal = smq1Q_XYI.target_model()
# 2) get the building blocks needed to specify which circuits are needed
prep_fiducials, meas_fiducials = smq1Q_XYI.prep_fiducials(), smq1Q_XYI.meas_fiducials()
germs = smq1Q_XYI.germs()
maxLengths = [1, 2, 4]  # roughly gives the length of the sequences used by GST
# 3) generate "fake" data from a depolarized version of mdl_ideal
mdl_true = mdl_ideal.depolarize(op_noise=0.01, spam_noise=0.001)
listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
    mdl_ideal, prep_fiducials, meas_fiducials, germs, maxLengths)
circ_list = pygsti.protocols.StandardGSTDesign(mdl_ideal, prep_fiducials, meas_fiducials, germs, maxLengths)
ds = pygsti.construction.generate_fake_data(mdl_true, listOfExperiments, nSamples=1000,
                                            sampleError="binomial", seed=1234)
# Run GST
results = pygsti.do_stdpractice_gst(ds, mdl_ideal, prep_fiducials, meas_fiducials,
                                    germs, maxLengths, modes="TP,Target", verbosity=1)
mdl_estimate = results.estimates['TP'].models['stdgaugeopt']
print("2DeltaLogL(estimate, data): ", pygsti.tools.two_delta_logl(mdl_estimate, ds))
print("2DeltaLogL(true, data): ", pygsti.tools.two_delta_logl(mdl_true, ds))
print("2DeltaLogL(ideal, data): ", pygsti.tools.two_delta_logl(mdl_ideal, ds))

gates = {'I': 0, 'X': 1, 'Y': 2}
gate_list = []
startn_list = []
endn_list = []

# Add first line to test init+readout
gate_list.append(gates['I'])
startn_list.append(int(0))
endn_list.append(int(1))

GST_sequence_file = 'Circuits_before_results.txt'
circ_list = []
qubit_list = []
with open(file=GST_sequence_file, mode='r') as f:
    circuits = f.readlines()
    for circ in circuits:
        gates = []
        qubit_indices = []
        c = circ.rstrip()
        for char in range(len(c)):
            if c[char] == 0 and c[char] == 'G':
                gates.append(c[0: c.find(':')])
                qubit_indices.append(int(c[c.find(':') + 1]))

            else:
                if c[char] == 'G':
                    gates.append(c[char: c.find(':', char)])
                    qubit_indices.append(c[c.find(':', char) + 1])

        circ_list.append(gates)
        qubit_list.append(qubit_indices)
f.close()
# :
# :
# diction= {"Gxpi2": {
#     'macro_index': 0,
#     'macro_function': macro
# }}