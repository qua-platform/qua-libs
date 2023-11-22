from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_cavity_locking_ETHZ_OPX1 import *

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=9510)
# qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

def lookup_table(variables, indices):
    table = {}
    for key, value in zip(variables, indices):
        table[key] = value
    return table

def update_variable(table, variable, value):
    qm.set_io1_value(table[variable])
    qm.set_io2_value(value)
    print(f"{variable} updated to {value}")

# Set the correspondence table between IO value 1 and the parameters
correspondence_table = lookup_table(["bitshift_scale_factor", "gain_P", "gain_I", "gain_D", "alpha", "target"], [1, 2, 3, 4, 5, 6])
###########################
# Run or Simulate Program #
###########################
# Get the running quantum machine
qm_id = qmm.list_open_quantum_machines()[0]
qm = qmm.get_qm(qm_id)
# Update the PID parameters on the fly via command lines
# update_variable(correspondence_table, "bitshift_scale_factor", 2)
# update_variable(correspondence_table, "gain_P", 0e-4)
# update_variable(correspondence_table, "gain_I", 0e-4)
# update_variable(correspondence_table, "gain_D", 0e-4)
# update_variable(correspondence_table, "target", 0.0)
