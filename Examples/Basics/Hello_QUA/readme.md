#Hello QUA

This is the most basic introduction script in QUA. 
It plays a constant amplitude pulse to output 1 
of the OPX and displays the waveform.
  
##config

The config file declares 1 quantum element called `qe1`. 
The quantum element has one operation called `playOp`
The `playOp` operation plays a constant amplitude pulse c
called `constPulse` which has default duration 1 $\muS$.

##program 

The QUA program simply calls the play command.

We run the program on the simulator for 500 clock cycles. 

##post processing

We plot the simulated samples to observe a single cycle of the waveform (1 MHz for 1 micro-second).   


