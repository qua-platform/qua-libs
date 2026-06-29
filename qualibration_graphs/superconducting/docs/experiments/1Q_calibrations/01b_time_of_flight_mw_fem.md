# Time of Flight (MW-FEM)

[`01b_time_of_flight_mw_fem.py`](../../../../../calibrations/1Q_calibrations/01b_time_of_flight_mw_fem.py)

Align the ADC capture window and gain for MW-FEM readout hardware.


## Purpose

Same goal as the OPX+/LF-FEM time-of-flight experiment, adapted for MW-FEM modules where ADC routing and offset handling differ. The readout transient must sit fully inside the integration window before any resonator spectroscopy.

![Perfect calibration result](images/time_of_flight_mw_fem.png){ .calibration-result }

## Prerequisites

- MW-FEM readout hardware connected and configured.

## (Chosen) Input Parameters Effect

* Time of flight guess:
    * Initial delay — misalignment smears all later integrated IQ data.
* Readout power:
    * Higher power improves SNR but can distort the line or saturate the ADC; lower power may hide the pulse entirely.
* Readout length:
    * Must exceed the ring-up time of the resonator response.

## Output

* Calibrated time of flight for the MW-FEM readout path.


## Experiment Step-by-Step description

1. Send a readout pulse and stream ADC data through the MW-FEM path.
1. Average traces over many shots.
1. Fit the pulse arrival time from the ring-up onset.
1. Update the machine configuration with the calibrated delay.
