from qua_dashboards.virtual_gates import virtual_layer_adder
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.run_video_mode.video_mode_specific_parameters import (
    VideoModeCommonParameters,
)

from typing import List, Literal, Dict, Union, Callable, Optional

from cloverleaf_tunup_bayb.setup_script import KDAC_params, dc_gates, KDACS_voltage_limits
from gatemanager.manager import GateManager
from qmtcodes.metadrivers.QMT import KDAC

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 10
    """Number of averages to perform. Default is 100."""
    sensor_names: List[str] = None
    """List of sensor dot names to measure in your measurement."""
    x_axis_name: str = None
    """The name of the swept element in the X axis."""
    y_axis_name: str = None
    """The name of the swept element in the Y axis."""
    x_points: int = 101
    """Number of measurement points in the X axis."""
    y_points: int = 101
    """Number of measurement points in the Y axis."""
    x_span: float = 0.01
    """The X axis span in volts"""
    y_span: float = 0.01
    """The Y axis span in volts"""
    ramp_duration: int = 1000
    """The ramp duration for the entire sweep."""
    per_line_compensation: bool = True
    """Whether to send a compensation pulse at the end of each scan line."""
    max_compensation_voltage: float = 0.05
    """The maximum voltage for the compensation pulse."""
    x_center: Optional[float] = None
    """The center of the X axis sweep. If dc_control = True, then this will be applied to the external source. Else, it will be applied by the OPX."""
    y_center: Optional[float] = None
    """The center of the Y axis sweep. If dc_control = True, then this will be applied to the external source. Else, it will be applied by the OPX."""
    plot_points: bool = False
    """Plots the existing points saved in the VirtualGateSet. Default True."""
    perform_edge_analysis: bool = False
    """Whether to perform edge analysis on the data."""
    per_line_wait: int = 0
    """Wait time at the start of each line, in order to allow the electrostatics to settle."""
    use_validation: bool = True
    """Whether to use validation with simulated data."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    VideoModeCommonParameters,
    NodeSpecificParameters,
):
    plot_pca: bool = True
    pixel_hold_duration: int = 1000


class OPXQDACParameters(
    NodeParameters,
    VideoModeCommonParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    x_from_qdac: bool = False
    "Check to perform 2D map using the QDAC instead of the OPX"
    y_from_qdac: bool = False
    "Check to perform 2D map using the QDAC instead of the OPX"
    post_trigger_wait_ns: int = 10000
    """A pause in the QUA programme to allow the QDAC to get to the correct level."""
    qdac_dwell_time_us: float = 200
    """The dwell time in microseconds for the QDAC."""


class SimulationParameters(
    NodeParameters,
    VideoModeCommonParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass


import numpy as np


# Backward-compatible aliases kept for existing nodes/tests that still use the
# legacy OPX/OPXQDAC parameter class names.
OPXParameters = Parameters
# OPXQDACParameters = Parameters

def get_axis_names(node): 
    """In the case that x_axis_name or y_axis_name is None, assign the first and second elements of QDs."""
    quantum_dots = list(node.machine.quantum_dots.keys())
    x_axis_name = node.parameters.x_axis_name
    y_axis_name = node.parameters.y_axis_name

    if node.parameters.x_axis_name is None:
        x_axis_name = quantum_dots[0]
    
    if node.parameters.y_axis_name is None: 
        y_axis_name = quantum_dots[1]
    return x_axis_name, y_axis_name

def get_voltage_arrays(node):
    """Extract the X and Y voltage arrays from a given node."""
    x_span, x_center, x_points = node.parameters.x_span, 0, node.parameters.x_points
    y_span, y_center, y_points = node.parameters.y_span, 0, node.parameters.y_points
    x_volts, y_volts = np.linspace(
        x_center - x_span / 2, x_center + x_span / 2, x_points
    ), np.linspace(y_center - y_span / 2, y_center + y_span / 2, y_points)
    return x_volts, y_volts


from .scan_modes import ScanMode


def _find_physical_dc_lists(
    virtual_dc_set: "VirtualDCSet",
    axis_name: str,
    axis_values: List[float],
) -> Dict[str, Union[List, np.ndarray]]:
    """Use the VirtualDCSet to yield a dictionary of physical dc_lists to use for the Qdac"""

    full_physical_dicts = {name: [] for name in virtual_dc_set.channels.keys()}

    for value in axis_values:
        virtual_dict = {axis_name: float(value)}
        physical_dict = virtual_dc_set.resolve_voltages(virtual_dict)

        for physical_gate in virtual_dc_set.channels.keys():
            full_physical_dicts[physical_gate].append(physical_dict[physical_gate])

    # Check if the physical list is constant or not
    physical_lists = {
        name: arr
        for name, arr in full_physical_dicts.items()
        if len(arr) > 1 and not np.allclose(arr, arr[0], atol=1e-8)
    }
    return physical_lists


def prepare_dc_lists(
    node,
    virtual_dc_set_id: str,
    axis_name: str,
    axis_values: List[float],
) -> None:
    """
    Prepares the DC list attributes for the QDAC channel. This function assumes the use of the
    Qdac2 driver from qcodes_contrib_drivers. This also assumes that the VoltageGate objects have
    their QdacSpec objects configured with the qdac_output_port and opx_trigger_out.
    """
    virtual_dc_set = node.machine.virtual_dc_sets[virtual_dc_set_id]
    physical_dc_lists = _find_physical_dc_lists(virtual_dc_set, axis_name, axis_values)

    trigger = None
    for ch_name in physical_dc_lists.keys():
        spec = getattr(virtual_dc_set.channels[ch_name], "qdac_spec", None)
        if spec is not None:
            trig = getattr(spec, "qdac_trigger_in", None)
            if trig is not None:
                trigger = trig
                break
    if trigger is None:
        raise ValueError(
            f"No trigger found for the physical outputs associated with the axis {axis_name}"
        )

    for name, voltages in physical_dc_lists.items():
        dc_list = node.machine.qdac.channel(
            virtual_dc_set.channels[name].qdac_spec.qdac_output_port
        ).dc_list(
            voltages=voltages,
            dwell_s=node.parameters.qdac_dwell_time_us / 1e6,
            stepped=True,
        )
        dc_list.start_on_external(trigger=trigger)



def prepare_dacs_and_gatemanager():

    dac1 = KDAC(address="COM3", reconnect=True) # Change COM PORT to the actual COM PORT of the KDAC 1
    dac2 = KDAC(address="COM4", reconnect=True) 
    ggs, chan_map = KDAC_params(r'C:/Users/siqec.project/src/cloverleaf_tunup_bayb/cloverleaf_tunup_bayb/TRES_bonding.xlsx')
    dacs = [dac1, dac2]
    dc = dc_gates(dacs, chan_map)
    KDACS_voltage_limits(dc, ggs)
    gate_list = ['P_x_1', 'P_x_2', 'P_x_3', 'P_x_4', 'B_x_12', 'B_x_23', 'B_x_34', 'C_i_sd', 'P_y_1', 'P_y_4', 'B_y_s1', 'B_y_12', 'B_y_34', 'B_y_4d', 'A_x_s', 'A_x_d', 'B_x_s1', 'B_x_4d', 'A_y_s', 'MA_y_23', 'MO_y_23', 'C_y_3d', 'A_y_d', 'C_x_sd', 'C_y_s2']
    g1 = []
    special_gates = ['P_x_1', 'P_x_2', 'P_x_3']
    detuning_gates = ['eps_12', 'eps_23', 'eps_34', 'delta_1234']
    for ind, ii in enumerate(gate_list):
        if ii in special_gates:
            index = special_gates.index(ii)
            ii = detuning_gates[index]
            g1.append(ii)
        elif ii == 'P_x_4':
            ii = detuning_gates[-1]
            g1.append(ii)
        else:
            g1.append(f'v{ii}')


    g2 = []
    for ind, ii in enumerate(g1):
        if ii in ['vB_x_12', 'vB_x_23', 'vB_x_34']:
            ii = ii.replace('vB','J')
            g2.append(ii)
        else:
            g2.append(ii)

    g3 = []
    special_gates = ['P_x_1', 'P_x_2', 'P_x_3', 'P_x_4']
    detuning_gates = ['eps23_p_eps12', 'eps23_m_eps12', 'eps34_p_eps23', 'eps34_m_eps23']
    for ind, ii in enumerate(gate_list):
        if ii in special_gates:
            index = special_gates.index(ii)
            ii = detuning_gates[index]
            g3.append(ii)
        elif ii == 'P_x_4':
            ii = detuning_gates[-1]
            g3.append(ii)
        else:
            g3.append(f'v{ii}')

    vgm = np.array([[ 1.49696232,  0.5218    ,  0.3689126 ,  1.        , -0.15019481,
         0.11476931,  0.02468148,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        , -0.05987849,
         0.        , -0.04490887,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.0133225 ,  0.        ],
       [-0.54456232,  0.4782    ,  0.3380874 ,  1.        ,  0.01011277,
         0.04221069,  0.09136917,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.02178249,
         0.        ,  0.01633687,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        , -0.0510333 ,  0.        ],
       [-0.552392  , -0.58      ,  0.58994   ,  1.        ,  0.08124758,
        -0.14961983,  0.02271895,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.02209568,
         0.        ,  0.01657176,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.00187226,  0.        ],
       [-0.400008  , -0.42      , -1.29694   ,  1.        ,  0.05883446,
        -0.00736017, -0.13876961,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.01600032,
         0.        ,  0.01200024,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.03583854,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.017     ,
        -0.007653  ,  0.        ,  0.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        , -0.01785   ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  1.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  1.        ]])
    
    layer_config_list = [('eps0', None, g1), ('J', None, g2), ('top', vgm, g2)]
    gates = GateManager(dc, gate_list, layer_configs=layer_config_list, use_cache=True)
    #gates = GateManager(dc, gate_list, layer_configs=[('eps0', None, g1), ('top', None, g3)], use_cache=True)
    #gates['eps0'].update_vgm('P_x_2', 'eps_12', -1 )
    #gates['eps0'].update_vgm('P_x_3', 'eps_23', -1 )
    #gates['eps0'].update_vgm('P_x_4', 'eps_34', -1 )
    #gates['top'].update_vgm('eps_23', 'eps23_p_eps12', 1 )
    #gates['top'].update_vgm('eps_12', 'eps23_m_eps12', -1 )

    gates.P_x_1.dac_channel.set_voltage_limits(min_voltage=-0.4, max_voltage=1.4)
    gates.P_x_2.dac_channel.set_voltage_limits(min_voltage=-0.4, max_voltage=1.4)
    gates.P_x_3.dac_channel.set_voltage_limits(min_voltage=-0.4, max_voltage=1.4)
    gates.P_x_4.dac_channel.set_voltage_limits(min_voltage=-0.4, max_voltage=1.4)

    for ch in gate_list:
        gg = getattr(gates, ch)
        gg.dac_channel.output_mode = 'ALL'
    #gates.C_i_sd.dac_channel.set_voltage_limits(min_voltage=-0.4, max_voltage=0.15)

    return dac1, dac2, gates, gates['top']

def close_dacs(dac1, dac2):
    dac1.close()
    dac2.close()