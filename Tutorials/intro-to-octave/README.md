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
   1. In `Set octave configuration` section, you initiate a data class `OctavesSettings` and set the octave parameters:
      1. name - the name of the octave, for example "octave1"
      2. con - the name of the controller that is connected to this octave, for example "con1"
      3. ip - the ip of the octave, for example "192.168.88.50"
      4. port - the port of the octave, for example 50.
      5. clock - the clock settings of the octave. Can be "Internal", "External_10MHz", "External_100MHz", "External_1000MHz". 
         * Note: the default is "Internal".
   2. In `Octave settings` section, you initiate a data class "ElementsSettings" and define all the parameters for the up-converter and down-converter modules:
      1. name - element's name as defined in the configuration, for example "qe1"
      2. LO_source - LO source for the up-converter module. Cen be "Internal", "LO1", "LO2", "LO3", "LO4", "LO5".
         * Note: the default is "Internal".
      3. gain - octave's gain. The default is 0
      4. switch_mode -  fast switch's mode. Can be "on", "trig_normal", "trig_inverse", "off".
         * Note: the default is "on".
      5. RF_in_port - list of the octave and the port of the down converter module. for example ["octave1", 1]. 
      6. Down_convert_LO_source - LO source for the down-converter module. Cen be "Internal"
         * Note: The default for down converter 1 is "Internal".
         * Note: There is no internal LO source for down converter 2. If one wants to use down converter 2, inputting a signal to Dmd2LO is needed.
      7. IF_mode - the mode of the IF module. Can be "direct", "envelope", "mixer".
         * Note: The default is "direct".
      * Note: if you want to define a different port mapping that is not the default one, you need to add 
 
      * Note: If you don't initiate this class for an element used in the program, the following parameters will be set:
        * LO_source - "Internal"
        * gain - 0
        * switch_mode - "on"
        * RF_in_port - 1 for element with analog outputs 1,2 and 2 for element with analog outputs 3,4.
        * Down_convert_LO_source - "Internal" for down converter 1 (that RF_in_port 1 is using), "Dmd2LO" for down converter 2 (that RF_in_port 2 is using).
        * IF_mode - "direct"
        
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


3. `octave_settings` function:
   1. Sets the octave's clock in.
      * Note: The Octave's clock out is fixed to 1GHz. 
   2. Sets the up-converters modules.
      * Note: the 4 options for the trigger are: on, off, trig_normal and trig_inverse. 
   3. Sets the down-converters modules
      * Note: If using down-converter module 2 don't forget to connect the LO to Dmd2LO in the back panel
   4. Calibration:
      * Note: It is set to True by default
