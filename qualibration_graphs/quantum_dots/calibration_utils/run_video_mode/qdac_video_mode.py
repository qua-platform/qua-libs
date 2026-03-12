from typing import Dict, List, Optional, Union
import time
from werkzeug.serving import make_server

from quam.core import QuamRoot
from qua_dashboards.video_mode import VideoModeComponent, HybridOPXQDACDataAcquirer, scan_modes
from qua_dashboards.voltage_control import VoltageControlComponent
from qua_dashboards.core import build_dashboard
from qua_dashboards.virtual_gates import VirtualLayerEditor, ui_update
from qcodes.parameters import DelegateParameter
import threading
import webbrowser
import subprocess

_DASHBOARD_THREAD: Optional[threading.Thread] = None
_DASHBOARD_SERVER = None


def stop_dashboard(port: int = 8050):
    """Nuclear option: kill anything on the given port."""
    global _DASHBOARD_THREAD, _DASHBOARD_SERVER

    # 1. Try graceful werkzeug shutdown first
    if _DASHBOARD_SERVER is not None:
        try:
            _DASHBOARD_SERVER.shutdown()
        except Exception:
            pass
        _DASHBOARD_SERVER = None

    # 2. Kill any process using the port (works even if server ref is lost)
    result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
    pids = result.stdout.strip().split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed PID {pid} on port {port}")
        except ProcessLookupError:
            pass

    # 3. Clean up thread ref
    if _DASHBOARD_THREAD is not None:
        _DASHBOARD_THREAD = None

    print(f"Port {port} is free.")


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
    settle_time: int = 200_000,
    qdac_ext_trigger_input_port=1,
    mid_scan_compensation: bool = True,
    use_buffered_stream: bool = False,
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

    dc_set = None
    if dc_control:
        external_qdac = "qdac_ip" in machine.network
        machine.connect_to_external_source(external_qdac=external_qdac)
        dc_set = machine.virtual_dc_sets.get(virtual_gate_id, None)

    voltage_control_tab, voltage_control_component = None, None
    if dc_set is not None:
        voltage_control_component = VoltageControlComponent(
            component_id="Voltage_Control",
            dc_set=dc_set,
            update_interval_ms=1000,
        )
        from qua_dashboards.video_mode.tab_controllers import (
            VoltageControlTabController,
        )

        voltage_control_tab = VoltageControlTabController(voltage_control_component=voltage_control_component)

    qmm = machine.connect()
    virtual_gate_set = machine.virtual_gate_sets[virtual_gate_id]

    data_acquirer = HybridOPXQDACDataAcquirer(
        qmm=qmm,
        machine=machine,
        gate_set=virtual_gate_set,
        x_axis_name=x_axis_name,  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        y_axis_name=y_axis_name,  # Must appear in gate_set.valid_channel_names; Virtual gate names also valid
        scan_modes=scan_modes_dict,
        result_type=result_type,  # "I", "Q", "amplitude", or "phase"
        available_readout_pulses=readout_pulses,  # Input a list of pulses. The default only reads out from the first pulse, unless the second one is chosen in the UI.
        voltage_control_component=voltage_control_component,
        dc_set=dc_set,
        qdac=machine.qdac,
        mid_scan_compensation=mid_scan_compensation,
        qdac_settle_delay_ns=settle_time,
        num_software_averages=num_software_averages,
        qdac_ext_trigger_input_port=qdac_ext_trigger_input_port,
        use_buffered_stream=use_buffered_stream,
        acquisition_interval_s=0.05,
    )

    def find_default(mode):
        if mode == "Voltage":
            points, span = 51, 0.03
        return (points, span)

    if x_span is not None or x_points is not None:
        x_sweepaxis = data_acquirer.find_sweepaxis(x_axis_name, mode=x_mode)
        x_sweepaxis.span = x_span if x_span is not None else find_default(x_mode)[1]
        x_sweepaxis.points = x_points if x_points is not None else find_default(x_mode)[0]

    if y_axis_name is not None and (y_span is not None or y_points is not None):
        y_sweepaxis = data_acquirer.find_sweepaxis(y_axis_name, mode=y_mode)
        y_sweepaxis.span = y_span if y_span is not None else find_default(x_mode)[1]
        y_sweepaxis.points = y_points if y_points is not None else find_default(x_mode)[0]

    virtual_gates_component = VirtualLayerEditor(gateset=virtual_gate_set, component_id="Virtual_Gates", dc_set=dc_set)

    video_mode_component = VideoModeComponent(
        data_acquirer=data_acquirer,
        data_polling_interval_s=0.05,
        save_path=save_path,
        shutdown_callback=stop_dashboard,
        voltage_control_tab=voltage_control_tab,
    )

    components = [video_mode_component, virtual_gates_component]

    app = build_dashboard(
        components=components,
        title="OPX Video Mode Dashboard",  # Title for the web page
    )

    ui_update(app, video_mode_component)

    def run_server():
        global _DASHBOARD_SERVER
        log(f"Starting new dashboard on http://localhost:{port}")
        _DASHBOARD_SERVER = make_server("0.0.0.0", port, app.server, threaded=True)
        _DASHBOARD_SERVER.serve_forever()

    _DASHBOARD_THREAD = threading.Thread(target=run_server, daemon=True, name="VideoMode")
    _DASHBOARD_THREAD.start()
    time.sleep(0.5)
    webbrowser.open(f"http://localhost:{port}")

    log("Dashboard running at http://localhost:8050")


def wait_for_dashboard():
    """Block the main thread until the dashboard is shut down.
    Call this at the end of a script to keep the process alive.
    In Jupyter, this is not needed since the kernel stays alive."""
    global _DASHBOARD_THREAD
    if _DASHBOARD_THREAD is not None and _DASHBOARD_THREAD.is_alive():
        try:
            _DASHBOARD_THREAD.join()
        except KeyboardInterrupt:
            stop_dashboard()


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
    try:
        get_ipython()  # noqa: F821
    except NameError:
        wait_for_dashboard()
