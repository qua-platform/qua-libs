######################################
# pip install qcodes_contrib_drivers #
######################################
from qcodes_contrib_drivers.drivers.QDevil.QDAC1 import QDac, Mode

# then, e.g.,
# address = 'ASRL2::INSTR'
# qdac = QDac(name='qdac', address=address, update_currents=False)