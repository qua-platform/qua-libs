from configuration_with_lf_fem_and_mw_fem import *
from qm import QuantumMachinesManager

qop_ip = "192.168.50.136"
cluster_name = "Cluster_1"
qmm = QuantumMachinesManager(host=qop_ip,port=None, cluster_name=cluster_name)
print(qmm.capabilities)