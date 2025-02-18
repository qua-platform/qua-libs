mw_in_channel_ports = ["opx_input"]
mw_out_channel_ports = ["opx_output"]
mw_in_out_channel_ports = mw_in_channel_ports + mw_out_channel_ports

iq_in_channel_ports = ["opx_input_I", "opx_input_Q", "frequency_converter_down"]
iq_out_channel_ports = ["opx_output_I", "opx_output_Q", "frequency_converter_up"]
iq_in_out_channel_ports = iq_in_channel_ports + iq_out_channel_ports

digital_out_ports = ["digital_output"]

valid_ports = mw_in_out_channel_ports + iq_in_out_channel_ports + digital_out_ports
