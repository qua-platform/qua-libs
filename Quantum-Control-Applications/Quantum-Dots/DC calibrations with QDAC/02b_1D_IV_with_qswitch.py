"""
02b — 1D IV (QDAC current vs voltage) + QSwitch routing

Companion to 02_1D_IV_measurement.py. Same QDAC sweep/sense, but routes
channels through a QSwitch first (always 0 V before changing relays).

02_1D_IV_measurement.py is typically 3+ terminal on a fridge board:

    Sweep V on QDAC ch 1 (plunger / ohmic)
    Measure I on QDAC ch 5 (a different ohmic / SET)

This script also supports a 2-terminal room-temp check (same QDAC channel
for V and I) via MODE = "same_line".

QSwitch firmware 2.0 speaks SCPI over UDP :5025 (not TCP VISA SOCKET).
Do not use `open (@1!1:1!9)` — list relays explicitly.

Install (once):
    python -m pip install pyvisa pyvisa-py ipykernel numpy matplotlib
    python -m pip install qcodes qcodes-contrib-drivers

VS Code: Python + Jupyter extensions → Select Interpreter → Run Cell on # %%

Run order: 1 → 2 (connect) → 3 (ROUTES) → 4 (zero + QSwitch) → 5 (IV) → 6 shutdown
"""

# %% ========================================================================
# 1. IMPORTS
# ===========================================================================

from time import sleep
import socket
import numpy as np
import matplotlib.pyplot as plt
from qcodes_contrib_drivers.drivers.QDevil import QDAC2

VISA_LIB = "@py"

# Write the instrument IPs
QDAC_IP = "127.0.0.1"  # QDAC TCP :5025
QSWITCH_IP = "127.0.0.1"  # QSwitch UDP :5025


# %% ========================================================================
# 1b. QSWITCH (UDP)
# ===========================================================================


class QSwitch_LAN:
    def __init__(self, ip_addr, port=5025, timeout=3.0):
        self._ip = ip_addr
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)

    def query(self, cmd):
        self._sock.sendto(f"{cmd}\n".encode(), (self._ip, self._port))
        data, _ = self._sock.recvfrom(2048)
        return data.decode(errors="replace").strip()

    def write(self, cmd, wait_opc=True):
        self._sock.sendto(f"{cmd}\n".encode(), (self._ip, self._port))
        if wait_opc:
            response = self.query("*OPC?")
            if response != "1":
                raise RuntimeError(f"QSwitch OPC failed: {response}")

    def closed_relays(self):
        return self.query("clos:stat?")

    def close(self):
        self._sock.close()


def open_line_routes(qsw, line):
    relays = ",".join(f"{line}!{p}" for p in range(1, 10))
    qsw.write(f"open (@{relays})")


def soft_ground_line(qsw, line):
    open_line_routes(qsw, line)
    qsw.write(f"close (@{line}!0)")


def connect_line_to_input(qsw, line):
    """QDAC IN -> DUT only."""
    open_line_routes(qsw, line)
    qsw.write(f"open (@{line}!0)")
    qsw.write(f"close (@{line}!9)")


def monitor_qdac_on_bnc(qsw, line, bnc):
    """QDAC IN -> DUT + BNC tap."""
    open_line_routes(qsw, line)
    qsw.write(f"open (@{line}!0)")
    qsw.write(f"close (@{line}!9)")
    qsw.write(f"close (@{line}!{bnc})")


# %% ========================================================================
# 2. CONNECT QDAC (TCP) + QSWITCH (UDP)
# ===========================================================================

qdac = QDAC2.QDac2(
    "QDAC",
    visalib=VISA_LIB,
    address=f"TCPIP::{QDAC_IP}::5025::SOCKET",
)
qsw = QSwitch_LAN(QSWITCH_IP)

print("QDAC IDN:", qdac.IDN())
print("QDAC errors:", qdac.errors())
print("QSwitch IDN:", qsw.query("*IDN?"))
print("QSwitch relays:", qsw.closed_relays())


# %% ========================================================================
# 3. IV + QSWITCH ROUTES
#
# Pick ONE mode:
#
# "same_line"
#     One BNC to GND (resistor).
#     QDAC ch 1 sources V and measures I on the SAME channel.
#     Wiring: BNC 1 centre -- resistor -- BNC GND
#     Use R_load >> 50 Ω (QDAC series output R is 50 Ω). A 50 Ω load
#     or 50 Ω scope termination halves the voltage at the BNC.
#
# "pcb"
#     DUT / PCB between TWO BNCs:
#         BNC 1 = high potential  (QDAC ch 1 voltage)
#         BNC 5 = low potential   (QDAC ch 5 current sense, usually ~0 V)
#     Wiring: BNC 1 -- DUT -- BNC 5
# ===========================================================================

MODE = "same_line"  # "same_line" | "pcb"

if MODE == "same_line":
    IV_CHANNEL = 1
    MONITOR_BNC = 1

    CONTACTS = {"dut": IV_CHANNEL}
    SWEEP_CONTACT = "dut"
    SENSOR_CHANNEL = IV_CHANNEL

    QSWITCH_LINES = [
        {"line": IV_CHANNEL, "bnc": MONITOR_BNC},
    ]

elif MODE == "pcb":
    V_CHANNEL = 1  # high potential  -> QSwitch line 1 -> BNC 1
    I_CHANNEL = 5  # low potential   -> QSwitch line 5 -> BNC 5
    V_BNC = 1
    I_BNC = 5

    CONTACTS = {"bias": V_CHANNEL}
    SWEEP_CONTACT = "bias"
    SENSOR_CHANNEL = I_CHANNEL

    QSWITCH_LINES = [
        {"line": V_CHANNEL, "bnc": V_BNC},  # high
        {"line": I_CHANNEL, "bnc": I_BNC},  # low / current
    ]

else:
    raise ValueError('MODE must be "same_line" or "pcb"')

print(f"MODE={MODE}  sweep V on ch {CONTACTS[SWEEP_CONTACT]}  sense I on ch {SENSOR_CHANNEL}")
print("QSwitch lines:", QSWITCH_LINES)

# Sweep
n_steps = 101
V_list = np.linspace(-1, 1, n_steps)
step_time = 20e-3  # s per voltage step
sensor_integration_time = 15e-3  # must be <= step_time

# Current meter (NOT voltage):
#   "low"  ~ ±150 nA   "high" ~ ±10 mA
sensing_range = "low"

# Voltage output range (set at 0 V):
#   "low"  ±2 V    "high" ±10 V
# |V| > 2 V requires "high"
OUTPUT_RANGE = "high" if np.max(np.abs(V_list)) > 2.0 else "low"

# Analog LP filter (bandwidth only; series R stays 50 Ω):
#   "dc"   ~10 Hz     quietest, slow
#   "med"  ~10 kHz
#   "high" ~300 kHz   power-up default
OUTPUT_FILTER = "high"  # "dc" | "med" | "high"
if OUTPUT_FILTER not in ("dc", "med", "high"):
    raise ValueError('OUTPUT_FILTER must be "dc", "med", or "high"')

print(
    f"OUTPUT_RANGE={OUTPUT_RANGE}  (voltage)   "
    f"OUTPUT_FILTER={OUTPUT_FILTER}  "
    f"sensing_range={sensing_range}  (current)"
)


# %% ========================================================================
# 4. ZERO QDAC, THEN ROUTE QSWITCH
# ===========================================================================

print("All QDAC outputs -> 0 V")
for ch in range(1, 25):
    qdac.channel(ch).dc_constant_V(0.0)

# Voltage range and filter must be set at ~0 V
v_ch = CONTACTS[SWEEP_CONTACT]
for ch in {v_ch, SENSOR_CHANNEL}:
    qdac.channel(ch).output_range(OUTPUT_RANGE)
    qdac.channel(ch).output_filter(OUTPUT_FILTER)
print(f"QDAC ch {v_ch} and ch {SENSOR_CHANNEL}: " f"output_range={OUTPUT_RANGE}  output_filter={OUTPUT_FILTER}")

sleep(0.2)

for item in QSWITCH_LINES:
    line = item["line"]
    if "bnc" in item:
        monitor_qdac_on_bnc(qsw, line=line, bnc=item["bnc"])
        print(f"QSwitch line {line} -> QDAC IN + BNC {item['bnc']}")
    else:
        connect_line_to_input(qsw, line=line)
        print(f"QSwitch line {line} -> QDAC IN")

print("Closed relays:", qsw.closed_relays())


# %% ========================================================================
# 5. 1D IV SWEEP (same logic as 02_1D_IV_measurement.py)
# ===========================================================================

arrangement = qdac.arrange(
    contacts=CONTACTS,
    internal_triggers={"inner"},
)

# PCB mode: low-side current channel stays at 0 V
if MODE == "pcb":
    qdac.channel(SENSOR_CHANNEL).dc_constant_V(0.0)

sweep = arrangement.virtual_sweep(
    contact=SWEEP_CONTACT,
    voltages=V_list,
    step_time_s=step_time,
    step_trigger="inner",
)

sensor = qdac.channel(SENSOR_CHANNEL)
sensor.measurement_aperture_s(sensor_integration_time)
sensor.measurement_range(sensing_range)
sensor.clear_measurements()
measurement = sensor.measurement()
measurement.start_on(arrangement.get_trigger_by_name("inner"))

arrangement.set_virtual_voltage(SWEEP_CONTACT, V_list[0])
sleep(0.5)

sweep.start()
sleep(n_steps * step_time + 0.5)

arrangement.set_virtual_voltage(SWEEP_CONTACT, 0)

raw = measurement.available_A()
available = list(map(lambda x: float(x), raw[-n_steps:]))

currents_mA = np.array(available) * 1000.0
fig, ax = plt.subplots()
ax.plot(V_list, currents_mA)
ax.set_title("1D IV (QDAC + QSwitch)")
ax.set_xlabel("Voltage [V]")
ax.set_ylabel("Current [mA]")
plt.show()

qdac.free_all_triggers()
print("IV done. Voltages returned to 0 on sweep contact.")


# %% ========================================================================
# 6. SAFE SHUTDOWN
# ===========================================================================

for ch in range(1, 25):
    qdac.channel(ch).dc_constant_V(0.0)

for item in QSWITCH_LINES:
    soft_ground_line(qsw, line=item["line"])
    print(f"QSwitch line {item['line']} -> soft ground")

qsw.close()
qdac.close()
print("QDAC + QSwitch closed.")
