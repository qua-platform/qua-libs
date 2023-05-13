# Intro to octave
This folder contains scripts for checking the octave and integrating it into your running experiments. 

## You can find here four python files 
1. `octave_introduction.py` contains the basic octave commands. This file is a step-by-step guide on how to configure the Octave, test the RF outputs and calibrate the mixers. 
2. `hello_octave.py` shows an example of running a program with the octave.
3. `configuration.py` is the configuration file for `hello_octave.py`.
4. `set_octave.py` sets all octave parameters.

### `octave_introduction.py`
In this file you can find the basic octave commands.
It contains the configuration and its parameters, adds information about the octave and runs octave commands. 
It is organized in the following sections:
   1. Configuring the clock
   2. Configuring the lo source, lo frequency, rf output gain and rf output mode
   3. Looking at the octave RF outputs
   4. Configuring the digital switches
   5. Configuring the down converter modules
   6. Calibration

### `hello_octave.py` 
This file shows an example of how to run a program with a setup contains OPX + octave. It uses the `configuration.py` file and `set_octave.py`  file for all the octave commands.

### `configuration.py` 
This file contains the config dictionary that the `hello_octave.py` file uses.
Note that while using your configuration file, there are two things to pay attention to:
   1. In the `elements` section, in `mixInputs` key, the mixer name should have a specific structure, as can be seen in this file.
   2. In the `mixers` section, the keys should correspond to the mixers names given in the `elements` section. 

### `set_octave.py`
This file contains all the relevant octave commands and should make it easier for you to integrate the Octave in your running experiments!

Let's talk about each function separately:

1. `get_elements_used_in_octave` function returns a list of all the elements that the octave is using for a specific program. This is for a later use. 


2. `octave_configuration` function:
   1. Creates a `calibration_db.json` file where the calibration parameters will be updated
   2. Adds the octave devices to the octave_config object
   3. Sets the port mapping for each OPX-octave pair
   
   * Note: the default is to use only one octave. If you have more than one use the flag `more_than_one_octave=True`, enter the `set_octave.py` file and modify the relevant parameters under this flag. 


3. `octave_settings` function:
   1. Sets the clock. The default is internal. 
      1. If you want to use an external, use the flag `external_clock=True` and set the relevant external clock frequency.
         2. If you want to use the clock from the OPT use the `ClockType.Buffered`.
      * Note: the above is related to configuring clock in. The Octave's clock out is fixed to 1GHz. 
   2. Sets the up-converters modules.
      1. LO  - The default LO source is internal. If you want to use an external LO you need to enter `set_octave.py` file and change the LO to the relevant external one (LO1, LO2, LO3, LO4 or LO5)
         Note: setting the LO frequency may be done only if LO source is internal.
      2. Gain - The default is zero gain. You can change it by entering `set_octave.py` file and modify the relevant command
      3. Trigger - The default is on. You can change it by entering `set_octave.py` file and modify the relevant command
        * Note: the 4 options for the trigger are: on, off, trig_normal and trig_inverse. 
   3. Sets the down-converters modules
      1. The default is: 
         1. connecting RF1 -> RF1in, RF2 -> RF2in
         2. The LO source for down converter module 1 is internal. You can change it by entering `set_octave.py` file and modify the relevant command.
      2. If using down-converter module 2 don't forget to connect the LO to Dmd2LO in the back panel
   4. Calibration:
      1. It is set to True by default
