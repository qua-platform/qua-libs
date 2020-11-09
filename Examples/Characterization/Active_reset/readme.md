# Superconducting qubit active reset 

The active reset procedure is a good example of how the OPX can use feedback from
a measurement to feed-forward the state of a qubit. 
It tests the qubit state by performing a measurment and comparing the demodulated 
and itegrated in-phase signal to a threshold value. It then conditionally plays a 
pi pulse on that qubit. 

 
## config

The configuration dictionary is in the `configuration.py` file and is imported into the main program file 
`t1.py`. 
 
The configuration defines two elements: `qubit` and `rr` (the readout
resonator). 

The `qubit` quantum element defines the qubit we are measuring. The OPX is connected to 
a mixer via two analog output channels of the OPX, numbered 1 and 2. We also 
specify the LO frequency received by the mixer using the `lo_frequency` field of the `mixInputs`
dictionary, and a mixer correction matrix using the `mixer` field. 

The `qubit` element defines the operation: `pi` which plays a gaussian pulse to the 
I channel of the OPX - producing a rotation of the qubit about the X axis. 
This, of course, must be calibrated to product a proper $\pi$ pulse, e.g. with a 
Time-Rabi experiment.

The `rr` quantum element allows to measure the qubit state by measuring the resonant
I and Q components of a reflected microwave signal.
It defines a `readout` pulse which is read on input number 1 of the OPX, 
as set by the `output` entry of the `rr` element dictionary.
Note also the `time_of_flight` and `smearing` parameters which must 
be defined to perform a measurement. As for the `qubit` we define the associated
`lo_frequency` and `mixer` correction. 

> ⚠️Note that failing to declare a `digital_marker` will not fail program compilation, 
but will prevent data from being acquired. 

## program 

The QUA program `active_reset` is built an averaging `for_` loop. 
The body of the loops plays the `pi` operation if `I>th`.
It then perform an additional meausrmeent and saves the `I`,`Q` results to streams
`I_stream` and `Q_stream` which can then saved to the tags `I` and `Q` which can be 
manipulated by the user on the client side (in Python)

   
## post processing

No post processing is currently supplied for this script. 