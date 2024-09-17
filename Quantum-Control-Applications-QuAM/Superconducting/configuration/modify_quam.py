# %%
import json
# from pathlib import Path

# from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import save_machine

path = r"C:\Users\daveh\Documents\Cloned Repos\CS_installations\configuration\quam_state"

machine = QuAM.load(path)

# u = unit(coerce_to_integer=True)

# Change active qubits
# machine.active_qubit_names = ["q0"]

# %%
# FLux-Line sampling_rate, upsampling_mode, output_mode

qubits = machine.active_qubits

qubit_fluxes_dict = {
    1: {"sampling_rate": 1e9, "upsampling_mode": "pulse", "output_mode": "direct", "delay": 0},
    2: {"sampling_rate": 1e9, "upsampling_mode": "pulse", "output_mode": "direct", "delay": 0},
    3: {"sampling_rate": 1e9, "upsampling_mode": "pulse", "output_mode": "direct", "delay": 0},
    4: {"sampling_rate": 1e9, "upsampling_mode": "pulse", "output_mode": "direct", "delay": 0},
    5: {"sampling_rate": 1e9, "upsampling_mode": "pulse", "output_mode": "direct", "delay": 0},
    6: {"sampling_rate": 1e9, "upsampling_mode": "pulse", "output_mode": "direct", "delay": 0}
}

for qb_flux in qubit_fluxes_dict.keys():
    fem_id = qubits[qb_flux - 1].z.opx_output.fem_id
    port_id = qubits[qb_flux - 1].z.opx_output.port_id

    machine.ports.analog_outputs.con1[fem_id][port_id].sampling_rate = qubit_fluxes_dict[qb_flux]["sampling_rate"]
    machine.ports.analog_outputs.con1[fem_id][port_id].upsampling_mode = qubit_fluxes_dict[qb_flux]["upsampling_mode"]
    machine.ports.analog_outputs.con1[fem_id][port_id].output_mode = qubit_fluxes_dict[qb_flux]["output_mode"]
    machine.ports.analog_outputs.con1[fem_id][port_id].delay = qubit_fluxes_dict[qb_flux]["delay"]


# %%
# XY-drive full_scale_power_dbm, upconverter_frequency, band

# NOTE: be aware of coupled ports for bands

qubits = machine.active_qubits

qubits_dict = {
    1: {"power_dbm": 1, "up_freq": 5700000000.0, "band": 2, "delay": 0},
    2: {"power_dbm": 1, "up_freq": 5700000000.0, "band": 2, "delay": 0},
    3: {"power_dbm": 1, "up_freq": 5700000000.0, "band": 2, "delay": 0},
    4: {"power_dbm": 1, "up_freq": 5700000000.0, "band": 2, "delay": 0},
    5: {"power_dbm": 1, "up_freq": 5700000000.0, "band": 2, "delay": 0},
    6: {"power_dbm": 1, "up_freq": 5700000000.0, "band": 2, "delay": 0}
}


for qb in qubits_dict.keys():
        
    fem_id = qubits[qb - 1].xy.opx_output.fem_id
    port_id = qubits[qb - 1].xy.opx_output.port_id
    machine.ports.mw_outputs.con1[fem_id][port_id].full_scale_power_dbm = qubits_dict[qb]["power_dbm"]
    machine.ports.mw_outputs.con1[fem_id][port_id].upconverter_frequency = qubits_dict[qb]["up_freq"]
    machine.ports.mw_outputs.con1[fem_id][port_id].band = qubits_dict[qb]["band"]
    machine.ports.mw_outputs.con1[fem_id][port_id].delay = qubits_dict[qb]["delay"]


# %%
# Resonator-drive full_scale_power_dbm, upconverter_frequency, band

# NOTE: be aware of coupled ports for bands

resonators_dict = {
    1: {
        "output_full_scale_power_dbm": 10,
        "output_upconverter_frequency": 6.2e9,
        "intermediate_frequency": -260e6,
        "output_band": 2,
        "input_band": 2,
        "input_downconverter_frequency": 6.2e9,
        "delay": 0
    },
    2: {
        "output_full_scale_power_dbm": 10,
        "output_upconverter_frequency": 6.2e9,
        "intermediate_frequency": -170e6,
        "output_band": 2,
        "input_band": 2,
        "input_downconverter_frequency": 6.2e9,
        "delay": 0
    },
    3: {
        "output_full_scale_power_dbm": 10,
        "output_upconverter_frequency": 6.2e9,
        "intermediate_frequency": -80e6,
        "output_band": 2,
        "input_band": 2,
        "input_downconverter_frequency": 6.2e9,
        "delay": 0
    },
    4: {
        "output_full_scale_power_dbm": 10,
        "output_upconverter_frequency": 6.2e9,
        "intermediate_frequency": +20e6,
        "output_band": 2,
        "input_band": 2,
        "input_downconverter_frequency": 6.2e9,
        "delay": 0
    },
    5: {
        "output_full_scale_power_dbm": 10,
        "output_upconverter_frequency": 6.2e9,
        "intermediate_frequency": 110e6,
        "output_band": 2,
        "input_band": 2,
        "input_downconverter_frequency": 6.2e9,
        "delay": 0
    },
    6: {
        "output_full_scale_power_dbm": 10,
        "output_upconverter_frequency": 6.2e9,
        "intermediate_frequency": 220e6,
        "output_band": 2,
        "input_band": 2,
        "input_downconverter_frequency": 6.2e9,
        "delay": 0
    }
}

resonators_to_change = [1, 2, 3, 4, 5, 6]

for rr in resonators_to_change:

    opx_output_fem_id = qubits[rr - 1].resonator.opx_output.fem_id
    opx_output_port_id = qubits[rr - 1].resonator.opx_output.port_id
    opx_input_fem_id = qubits[rr - 1].resonator.opx_input.fem_id
    opx_input_port_id = qubits[rr - 1].resonator.opx_input.port_id

    machine.active_qubits[rr - 1].resonator.intermediate_frequency = resonators_dict[rr]["intermediate_frequency"]

    machine.ports.mw_outputs.con1[opx_output_fem_id][opx_output_port_id].full_scale_power_dbm = resonators_dict[rr]["output_full_scale_power_dbm"]
    machine.ports.mw_outputs.con1[opx_output_fem_id][opx_output_port_id].upconverter_frequency = resonators_dict[rr]["output_upconverter_frequency"]
    machine.ports.mw_outputs.con1[opx_output_fem_id][opx_output_port_id].band = resonators_dict[rr]["output_band"]
    machine.ports.mw_outputs.con1[opx_output_fem_id][opx_output_port_id].delay = resonators_dict[rr]["delay"]

    machine.ports.mw_inputs.con1[opx_input_fem_id][opx_input_port_id].band = resonators_dict[rr]["input_band"]
    machine.ports.mw_inputs.con1[opx_input_fem_id][opx_input_port_id].downconverter_frequency = resonators_dict[rr]["input_downconverter_frequency"]

# %%
# Resonator Operations amplitude
for qubit in machine.active_qubits:
    # NOTE: in MW-FEM it scales from [-1, 1] with respect to the full_scale_power_dbm of the corresponding output
    qubit.resonator.operations['readout'].amplitude = 0.5
    qubit.resonator.operations['const'].amplitude = 0.5

# %%
# save into state.json
save_machine(machine, path)

# %%
# View the corresponding "raw-QUA" config
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)

# %%
