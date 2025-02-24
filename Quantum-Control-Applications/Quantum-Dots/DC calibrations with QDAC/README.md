# DC calibrations done with QDAC

## Basic Files
These files showcase various DC calibrations that are done with the QDAC. 
All those measurements are done with the current sensor of the QDAC and there is no OPX involved. 
These files were tested on real devices, but are given as-is with no guarantee.

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

0. [QDAC communication](./00_QDAC_communication.py) - This script helps to establish the communication with the QDAC. It shows how to find the DQAC's IP using the USB connection, and then communicate through Ethernet. 
1. [Leakage measurement](./01_Leakage_measurement.py) - Performs a leakage measurement to check  if there is a leakage between different channels inside the device. This is done by applying a DC voltage to one channel and measure the current on all the channels using internal current sensors inside the QDAC. This is then repeated to all the channels.
2. [1D IV measurement](./02_1D_IV_measurement.py) - Performs a 1D IV measurement.  This is done by applying a DC voltage to either one of the sensor gates or a QD’s plunger or barrier gate and measuring the current on the SET using the internal QDAC sensor. The voltage on the gate is swept and the current is measured on the SET channel at every DC voltage step.
3. [2D IV measurement](./03_2D_IV_measurement.py) - Performs a 2D IV measurement. This is similar to the 1D IV measurement, but sweeping the voltage on two plunger gates. 
4. [Charge Stability virtual gates](./04_Charge_Stability_virtual_gates.py) - instead of applying DC voltages on the different gates, it is often more desired to change only the electrical potential of a quantum dot or tunnel barrier, which requires changing the voltages on multiple gates simultaneously. Virtual gates from the QDAC can correct for this. 

* Note: scripts 1-4 are using the QDAC’s internal current sensor. If an external current sensor is preferred please refer to the [QDAC documentation](https://qm.quantum-machines.co/87kjeif6).
* Note: in order to use those scripts, it is required to: 
  * `pip install pyvisa`
  * `pip install qcodes_contrib_drivers`