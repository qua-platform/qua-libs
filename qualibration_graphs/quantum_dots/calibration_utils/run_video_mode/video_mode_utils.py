from typing import Dict, List, Optional
import signal
import psutil
import time
import requests


from quam.core import QuamRoot
from qua_dashboards.video_mode import VideoModeComponent, OPXDataAcquirer, scan_modes
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.core import build_dashboard
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qm import QuantumMachinesManager
from qcodes.parameters import DelegateParameter
from qua_dashboards.utils import setup_logging
import threading


_DASHBOARD_THREAD: Optional[threading.Thread] = None
_DASHBOARD_APP = None

def is_port_in_use(port: int = 8050): 
    for connection in psutil.net_connections():
        if connection.laddr.port == port and connection.status == "LISTEN": 
            return True
    return False

def stop_existing_dashboard(port: int = 8050, timeout: float = 15.0): 
    """Kill any process listening on the specified port."""
    logger = setup_logging(__name__)
    logger.info(f"Checking for existing dashboard on port {port}...")
    
    killed = False
    for process in psutil.process_iter(['pid', 'name']):
        try:
            # Get connections for this process
            connections = process.net_connections(kind='inet')
            for conn in connections:
                # Check if this process is listening on our port
                if conn.status == 'LISTEN' and conn.laddr.port == port:
                    logger.info(f"Found process {process.name()} (PID: {process.pid}) using port {port}")
                    logger.info(f"Terminating process...")
                    process.terminate()
                    try:
                        process.wait(timeout=timeout)
                        logger.info(f"Successfully terminated process on port {port}")
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process did not terminate gracefully, killing...")
                        process.kill()
                        process.wait(timeout=5)
                    killed = True
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        
        if killed:
            break
    
    if not killed:
        logger.info(f"No process found using port {port}")
    
    return killed


def launch_video_mode(
    machine: QuamRoot,
    x_axis_name: str, 
    y_axis_name: str, 
    virtual_gate_id: str, 
    readout_pulses: List, 
    dc_control: bool, 
    save_path: str, 
    x_span: float = None, 
    y_span: float = None, 
    x_points: int = None, 
    y_points: int = None,
    x_mode: str = "Voltage", 
    y_mode: str = "Voltage", 
    scan_modes_dict: Dict = None, 
    result_type: str = "I",
    port: int = 8050, 
    auto_kill_existing: bool = True
) -> threading.Thread: 
    global _DASHBOARD_THREAD, _DASHBOARD_APP
    if scan_modes_dict is None: 
        scan_modes_dict = {
            "Switch_Raster_Scan": scan_modes.SwitchRasterScan(), 
            "Raster_Scan": scan_modes.RasterScan(), 
            "Spiral_Scan": scan_modes.SpiralScan(),
        }

    if auto_kill_existing: 
        stop_existing_dashboard(port = port)
        time.sleep(1)

    logger = setup_logging(__name__)


    qmm = machine.connect()
    virtual_gate_set = machine.virtual_gate_sets[virtual_gate_id]
    data_acquirer = OPXDataAcquirer(
        qmm=qmm,
        machine=machine,
        gate_set=virtual_gate_set,  # Replace with your GateSet instance
        x_axis_name=x_axis_name,  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        y_axis_name=y_axis_name,  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        scan_modes=scan_modes_dict,
        result_type=result_type,  # "I", "Q", "amplitude", or "phase"
        available_readout_pulses=readout_pulses # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI. 
    )
    data_acquirer.x_mode = x_mode if x_mode is not None else "Voltage"
    data_acquirer.y_mode = y_mode if y_mode is not None else "Voltage"

    if x_span is not None or x_points is not None:
        x_sweepaxis = data_acquirer.find_sweepaxis(x_axis_name, mode = x_mode)
        x_sweepaxis.span = x_span if x_span is not None else 0.03
        x_sweepaxis.points = x_points if x_points is not None else 51

    if y_axis_name is not None and (y_span is not None or y_points is not None): 
        y_sweepaxis = data_acquirer.find_sweepaxis(y_axis_name, y_mode)
        y_sweepaxis.span = y_span if y_span is not None else 0.03
        y_sweepaxis.points = y_points if y_points is not None else 51

    video_mode_component = VideoModeComponent(
        data_acquirer = data_acquirer, 
        data_polling_interval_s = 0.2, 
        save_path = save_path
    )

    virtual_gates_component = VirtualLayerEditor(gateset = virtual_gate_set, component_id = 'Virtual_Gates')

    components = [video_mode_component, virtual_gates_component]
    if dc_control: 
        voltage_parameters = []
        physical_channels = machine.physical_channels
        for ch in list(physical_channels.values()): 
            voltage_parameters.append(DelegateParameter(
                name = ch.id, label = ch.id, source = ch.offset_parameter
            ))
        voltage_control_component = VoltageControlComponent(component_id="Voltage_Control",voltage_parameters=voltage_parameters,update_interval_ms=1000)
        components.append(voltage_control_component)

    app = build_dashboard(
        components=components,
        title="OPX Video Mode Dashboard",  # Title for the web page
    )

    ui_update(app, video_mode_component)

    _DASHBOARD_APP = app
    

    def run_server(): 
        logger.info(f"Dashboard built. Starting Dash server on http://localhost:{port}")
        app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)


    server_thread = threading.Thread(target=run_server, daemon=True, name="VideoModeDashboard")
    server_thread.start()
    _DASHBOARD_THREAD = server_thread
    logger.info("Dashboard server started in background thread")
    logger.info("Node will complete but dashboard remains accessible at http://localhost:8050")


    return server_thread


def create_video_mode(
    machine: QuamRoot,
    x_axis_name: str,
    y_axis_name: str,
    virtual_gate_id: str,
    readout_pulses: List,
    dc_control: bool,
    save_path: str,
    **kwargs
) -> None:
    """
    Convenience function that automatically refreshes video mode.
    Call this from your measurement nodes to keep video mode in sync.
    
    This will:
    1. Kill any existing video mode instance
    2. Launch a fresh instance with the current configuration
    3. Return immediately (runs in background)
    """
    launch_video_mode(
        machine=machine,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        virtual_gate_id=virtual_gate_id,
        readout_pulses=readout_pulses,
        dc_control=dc_control,
        save_path=save_path,
        auto_kill_existing=True,
        **kwargs
    )
