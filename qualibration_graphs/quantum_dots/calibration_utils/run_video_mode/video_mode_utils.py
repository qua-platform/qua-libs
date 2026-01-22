from typing import Dict, List, Optional, Union
import time
from werkzeug.serving import make_server

from quam.core import QuamRoot
from qua_dashboards.video_mode import VideoModeComponent, OPXDataAcquirer, scan_modes
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.core import build_dashboard
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qcodes.parameters import DelegateParameter
import threading
import webbrowser

_DASHBOARD_THREAD: Optional[threading.Thread] = None
_DASHBOARD_SERVER = None


def stop_dashboard():
    """Gracefully shutdown the dashboard server without killing the process."""
    global _DASHBOARD_THREAD, _DASHBOARD_SERVER

    if _DASHBOARD_SERVER is not None:
        print("Shutting down existing dashboard...")
        _DASHBOARD_SERVER.shutdown()
        _DASHBOARD_SERVER = None

    if _DASHBOARD_THREAD is not None and _DASHBOARD_THREAD.is_alive():
        _DASHBOARD_THREAD.join(timeout=3)
        _DASHBOARD_THREAD = None

    time.sleep(1)


def launch_video_mode(
    machine: QuamRoot,
    log,
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
    num_software_averages: int = 1,
    x_mode: str = "Voltage",
    y_mode: str = "Voltage",
    scan_modes_dict: Dict = None,
    result_type: str = "I",
    port: int = 8050,
) -> None:
    global _DASHBOARD_THREAD, _DASHBOARD_SERVER

    if _DASHBOARD_THREAD is not None and _DASHBOARD_THREAD.is_alive():
        log("Stopping existing dashboard")
        stop_dashboard()

    if scan_modes_dict is None:
        scan_modes_dict = {
            "Switch_Raster_Scan": scan_modes.SwitchRasterScan(),
            "Raster_Scan": scan_modes.RasterScan(),
            "Spiral_Scan": scan_modes.SpiralScan(),
        }

    qmm = machine.connect()
    virtual_gate_set = machine.virtual_gate_sets[virtual_gate_id]
    data_acquirer = OPXDataAcquirer(
        qmm=qmm,
        machine=machine,
        gate_set=virtual_gate_set,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        scan_modes=scan_modes_dict,
        result_type=result_type,
        available_readout_pulses=readout_pulses,
        num_software_averages=num_software_averages,
        x_mode=x_mode,
        y_mode=y_mode,
    )

    def find_default(mode):
        if mode == "Voltage":
            points, span = 51, 0.03
        if mode == "Frequency":
            points, span = 51, int(10e6)
        if mode == "Amplitude":
            points, span = 51, 0.01
        return (points, span)

    if x_span is not None or x_points is not None:
        x_sweepaxis = data_acquirer.find_sweepaxis(x_axis_name, mode=x_mode)
        x_sweepaxis.span = x_span if x_span is not None else find_default(x_mode)[1]
        x_sweepaxis.points = x_points if x_points is not None else find_default(x_mode)[0]

    if y_axis_name is not None and (y_span is not None or y_points is not None):
        y_sweepaxis = data_acquirer.find_sweepaxis(y_axis_name, mode=y_mode)
        y_sweepaxis.span = y_span if y_span is not None else find_default(x_mode)[1]
        y_sweepaxis.points = y_points if y_points is not None else find_default(x_mode)[0]

    video_mode_component = VideoModeComponent(
        data_acquirer=data_acquirer, data_polling_interval_s=0.2, save_path=save_path, shutdown_callback=stop_dashboard
    )

    virtual_gates_component = VirtualLayerEditor(gateset=virtual_gate_set, component_id="Virtual_Gates")

    components = [video_mode_component, virtual_gates_component]
    if dc_control:
        voltage_parameters = []
        physical_channels = machine.physical_channels
        for ch in list(physical_channels.values()):
            voltage_parameters.append(DelegateParameter(name=ch.id, label=ch.id, source=ch.offset_parameter))
        voltage_control_component = VoltageControlComponent(
            component_id="Voltage_Control", voltage_parameters=voltage_parameters, update_interval_ms=1000
        )
        components.append(voltage_control_component)

    app = build_dashboard(
        components=components,
        title="OPX Video Mode Dashboard",  # Title for the web page
    )

    ui_update(app, video_mode_component)

    def run_server():
        global _DASHBOARD_SERVER
        log(f"Starting new dashboard on http://localhost:{port}")
        _DASHBOARD_SERVER = make_server("0.0.0.0", port, app.server)
        _DASHBOARD_SERVER.serve_forever()

    _DASHBOARD_THREAD = threading.Thread(target=run_server, daemon=True, name="VideoMode")
    _DASHBOARD_THREAD.start()
    time.sleep(0.5)
    webbrowser.open(f"http://localhost:{port}")
    
    log("Dashboard running at http://localhost:8050")


def create_video_mode(
    machine: QuamRoot,
    x_axis_name: str,
    y_axis_name: str,
    virtual_gate_id: str,
    readout_pulses: List,
    num_software_averages: int,
    dc_control: bool,
    save_path: str,
    **kwargs,
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
        num_software_averages=num_software_averages,
        save_path=save_path,
        **kwargs,
    )
