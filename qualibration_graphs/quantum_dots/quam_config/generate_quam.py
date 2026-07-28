"""
General purpose script to generate the wiring and the QUAM that corresponds to your experiment for the first time.
The workflow is as follows:
    - Copy the content of the wiring example corresponding to your architecture and paste it here.
    - Modify the statis parameters to match your network configuration.
    - Update the instrument setup section with the available hardware.
    - Define which qubit ids are present in the system.
    - Define any custom/hardcoded channel addresses.
    - Allocate the wiring to the connectivity object based on the available instruments.
    - Visualize and validate the resulting connectivity.
    - Build the wiring and QUAM.
    - Populate the generated quam with initial values by modifying and running populate_quam_xxx.py
"""
