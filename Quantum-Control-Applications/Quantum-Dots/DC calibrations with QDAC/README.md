# DC calibrations with QDAC

## Overview
This document provides guidance on establishing communication with the QDAC and performing various DC calibrations using its internal current sensor. These calibrations do not involve the OPX. 

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with caution. These files are given as-is with no guarantee.

## Establishing communication with QDAC
QDAC supports two communication methods:
1. Ethernet communication (recommended for faster performance)
2. USB communication

### 1. Ethernet communication
By default, the QDAC operates in DHCP mode. Once connected to the network, its IP address can be identified. If the IP is known, establish communication using the following code:
```commandline
import pyvisa as visa
from qcodes_contrib_drivers.drivers.QDevil import QDAC2

qdac_ipaddr = "172.0.0.1" # Write the QDAC IP address
# Open communication through Ethernet port
qdac = QDAC2.QDac2("QDAC", visalib="@py", address=f"TCPIP::{qdac_ipaddr}::5025::SOCKET")
```
### 2. USB communication
If the QDAC’s IP is unknown, or if USB communication is preferred, connect via USB and use the following code:
```commandline
import pyvisa as visa
from qcodes_contrib_drivers.drivers.QDevil import QDAC2

# List available VISA resources to find the relevant connection
rm = visa.ResourceManager('')
rm.list_resources()

# Insert the correct serial port from the listed resources
qdac_serial_addr = 'ASRL5::INSTR' # This is typical for windows, for mac users it will look something like 'ASRL/dev/cu.usbserial-14210::INSTR'
qdac = QDAC2.QDac2('QDAC', visalib='@py', address=qdac_serial_addr)
  ```

Note:  To determine QDAC’s IP address via USB and then switch to Ethernet communication, refer to the [QDAC communication](./00_QDAC_communication.py) script. 

## DC calibrations

1. [Leakage measurement](./01_Leakage_measurement.py) - Performs a leakage measurement to check if there is a leakage between different channels inside the device. This is done by applying a DC voltage to one channel and measure the current on all the channels using internal current sensors inside the QDAC. This is then repeated to all the channels.
2. [1D IV measurement](./02_1D_IV_measurement.py) - Performs a 1D IV measurement.  This is done by applying a DC voltage to either one of the sensor gates or a QD’s plunger or barrier gate and measuring the current on the SET using the internal QDAC sensor. The voltage on the gate is swept and the current is measured on the SET channel at every DC voltage step.
3. [2D IV measurement](./03_2D_IV_measurement.py) - Performs a 2D IV measurement. This is similar to the 1D IV measurement, but sweeping the voltage on two plunger gates. 
4. [Charge Stability virtual gates](./04_Charge_Stability_virtual_gates.py) - instead of applying DC voltages on the different gates, it is often more desired to change only the electrical potential of a quantum dot or tunnel barrier, which requires changing the voltages on multiple gates simultaneously. Virtual gates from the QDAC can correct for this. 

* Notes:
  * All calibrations use QDAC’s internal current sensor. If an external current sensor is preferred, please refer to the [QDAC documentation](https://docs.quantum-machines.co/latest/docs/Hardware/QDAC/).
  * To use those scripts, it is required to: 
    * `pip install pyvisa`
    * `pip install qcodes_contrib_drivers`