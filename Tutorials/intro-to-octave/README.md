# Intro to octave
This folder contains scripts for checking the octave and integrating it into your running experiments. 

## This folder contains four python files 
1. `health_check_octave.py` runs some basics octave tests.
2. `hello_octave.py` shows an example of running a program with the octave.
3. `configuration.py` is the configuration file for `hello_octave.py`.
4. `set_octave.py` sets all octave parameters.


### `set_octave.py`

Let's talk about each function separately:

1. `get_elements_used_in_octave` function returns a list of all the elements that the octave is using for a specific program. This is for a later use. 


2. `octave_configuration` function:
   1. Creates a `calibration_db.json` file where the calibration parameters will be updated.
   2. Adds the octave devices to the octave_config object
   3. Sets the port mapping for each OPX-octave pair. 
   
   Note that the default is to use only one octave. If you have more than one you need set `more_than_one_octave=True`, enter the `set_octave.py` file and change the relevant parameters under this flag. 


3. `octave_settings` function:
   1. Sets the clock. The default is internal. 
      1. If you want to use an external clock with 10MHz set `external_clock=`10MHz``
      2. If you want to use an external clock with 10MHz set `external_clock=`100MHz``
      3. If you want to use an external clock with 10MHz set `external_clock=`1GHz``
   2. Sets the up-converters modules.
      1. LO  - The default LO source is internal. If you want to use an external LO you need to enter the `set_octave.py` file and change the LO to the relevant external one (LO1, LO2, LO3, LO4 or LO5).
      2. Gain - The default is zero gain. You can change it by entering the `set_octave.py` file and modify the relevant command.
      3. Trigger - The default is on, You can change it by entering the `set_octave.py` file and modify the relevant command.
   3. Sets the down-converters modules.
      1. The default is: 
         1. connecting RF1 -> RF1in, RF2 -> RF2in. 
         2. The LO source for down converter module 1 is internal. You can change it by entering the `set_octave.py` file and modify the relevant command.
      2. If using down-converter module 2 don't forget to connect the LO to Dmd2LO in the back panel.  
   4. Calibration:
      1. It is set to True by default. 
