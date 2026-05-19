import pyvisa as visa
from typing import Tuple, Sequence, List, Dict, Set, Union, Optional
from dataclasses import dataclass
import socket
from datetime import datetime
from time import sleep as sleep_s
import re
import itertools
from packaging.version import parse
import serial.tools.list_ports as list_ports
from platform import system as platform_system

# version 1.1.4

"""
Python driver for the QSwitch.


Use like this for UDP ethernet communication (Firmware version >= 1.9):

import qswitch_driver
qswitch = qswitch_driver.QSwitch( qswitch.UDPconfig(ip="192.168.8.100"))


Use like this on USB:

import qswitch_driver
qswitch = qswitch_driver.QSwitch( qswitch.VISAconfig(visaAddress='ASRL12::INSTR'))

or for automatic USB detection:

device = qswitch_driver.find_qswitch_on_usb()
qswitch = qswitch_driver.QSwitch( qswitch.VISAConfig(visaAddress=device))


Use like this for TCP/IP ethernet communication (Firmware version < 1.9):

import qswitch_driver
qswitch = qswitch_driver.QSwitch( qswitch.VISAconfig(visaAddress="TCPIP::192.168.8.100::5025::SOCKET"))
"""


@dataclass
class UDPConfig:
    ip: str
    timeout_ms: float = 2000
    delay_s: float = 0.01
    query_attempts: int = 5
    write_attempts: int = 5


@dataclass
class VISAConfig:
    visaAddress: str
    timeout_ms: float = 5000
    delay_s: float = 0.01


class QSwitch:

    def __init__(self, config: VISAConfig | UDPConfig):
        """
        Connect to a QSwitch

        Args:
            resource:  Visa / UDP configuration
        """

        self.log = print
        self.verbose = False
        self._config = config

        if isinstance(config, UDPConfig):
            # Setup UDP configuration for ethernet port
            self._udp_mode = True
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(self._config.timeout_ms / 1000)  # Convert ms to seconds
            if self.verbose:
                self.log(f"{datetime.now()} Connected UDP: {self._config.ip}:5025, timeout:{self._config.timeout_ms}ms")
        elif isinstance(config, VISAConfig):
            # Setup VISA configuration for USB or TCP/IP
            self._udp_mode = False
            self._switch = visa.ResourceManager("@py").open_resource(self._config.visaAddress)
            self._switch.write_termination = "\n"
            self._switch.read_termination = "\n"
            self._switch.timeout = self._config.timeout_ms
            self._switch.query_delay = self._config.delay_s
            self._switch.baud_rate = 9600
            if self.verbose:
                self.log(
                    f"{datetime.now()} Connected VISA: timeout: {self._switch.timeout} ms, query_delay: {self._switch.query_delay} s"
                )

        self._set_default_names()
        self._set_up_debug_settings()

        self._state = self.query("stat?")

        self._check_for_wrong_model()
        self._check_for_incompatible_firmware()
        self._state_force_update()

    OneOrMore = Union[str, Sequence[str]]
    State = Sequence[Tuple[int, int]]

    # -----------------------------------------------------------------------
    # Instrument-wide functions
    # -----------------------------------------------------------------------

    def auto_save(self, val: int | str) -> None:
        """
        Set the autosave function on or off.

        Args:
            val: 1,'1','on' or 0,'0','off'
        """
        if val in [0, 1]:
            self.write(f"aut {str(val)}")
        elif val.lower() in ["0", "1", "on", "off"]:
            self.write(f"aut {val}")
        else:
            raise ValueError(f"Unknown autosave setting {val}")

    def get_auto_save(self) -> str:
        """
        Get the autosave status.
        """
        return self.query("aut?")

    def errors(self) -> str:
        """
        Retrieve and clear all previous errors

        Returns:
            str: Comma separated list of errors or '0, "No error"'
        """
        return self.query("all?")

    def error(self) -> str:
        """
        Retrieve next error

        Returns:
            str: The next error or '0, "No error"'
        """
        return self.query("next?")

    def error_indicator(self, val: int | str) -> None:
        """
        Set the error indicator function on or off.

        Args:
            val: 1,'1','on' or 0,'0','off'
        """
        if val in [0, 1]:
            self.write(f"beep:stat {str(val)}")
        elif val.lower() in ["0", "1", "on", "off"]:
            self.write(f"beep:stat {val}")
        else:
            raise ValueError(f"Unknown autosave setting {val}")

    def reset(self) -> None:
        """ "
        Reset the QSwitch to power-on conditions and then update the known state
        """
        self._write("*rst")
        sleep_s(0.6)
        self._state_force_update()

    def restart(self) -> None:
        """ "
        Restart the QSwitch firmware, including the LAN interface, resets to power-on conditions, and then update the known state
        """
        self._write("rest")
        sleep_s(5)
        self._state_force_update()

    # -----------------------------------------------------------------------
    # Direct manipulation of the relays
    # -----------------------------------------------------------------------

    def close_relays(self, relays: State) -> None:
        """
        Close a set of relays

        Args:
            relays: sets of channel and breakout numbers
        """
        currently = self._channel_list_to_state(self._state)
        union = list(itertools.chain(currently, relays))
        self._effectuate(union)

    def close_relay(self, line: int, tap: int) -> None:
        """
        Close a single relay

        Args:
            line: Fischer channel number
            tap: BNC breakout number
        """
        self.close_relays([(line, tap)])

    def open_relays(self, relays: State) -> None:
        """
        Open a set of relays

        Args:
            relays: sets of channel and breakout numbers
        """
        currently = frozenset(self._channel_list_to_state(self._state))
        subtraction = frozenset(relays)
        self._effectuate(list(currently - subtraction))

    def open_relay(self, line: int, tap: int) -> None:
        """
        Open a single relay

        Args:
            line: Fischer channel number
            tap: BNC breakout number
        """
        self.open_relays([(line, tap)])

    # -----------------------------------------------------------------------
    # Manipulation functions - close/open relays in a fixed order
    # ----------------------------------------------------------------------

    def ground_and_release(self, lines: OneOrMore) -> None:
        """
        Soft ground one or more channels and then disconnect them from the input and breakout connectors.

        Args:
            lines: One or more channels to ground
        """
        connections: List[Tuple[int, int]] = []
        if isinstance(lines, str):
            line = self._to_line(lines)
            self.close_relay(line, 0)
            taps = range(1, 10)
            connections = list(itertools.zip_longest([], taps, fillvalue=line))
            self.open_relays(connections)
        else:
            numbers = map(self._to_line, lines)
            grounds = list(itertools.zip_longest(numbers, [], fillvalue=0))
            self.close_relays(grounds)
            for tap in range(1, 10):
                connections += itertools.zip_longest(map(self._to_line, lines), [], fillvalue=tap)
            self.open_relays(connections)

    def ground_and_release_all(self) -> None:
        """
        Soft ground all channels and then disconnect them from the input and breakout connectors.
        """
        grounds = list(itertools.zip_longest(range(1, 25), [], fillvalue=0))
        self.close_relays(grounds)
        for tap in range(1, 10):
            connections = itertools.zip_longest(range(1, 25), [], fillvalue=tap)
            self.open_relays(connections)

    def connect_and_unground(self, lines: OneOrMore) -> None:
        """
        Connect one or more channels to the input Fischer connector and then disconnect them from the soft ground.

        Args:
            lines: One or more channels to connect to the input Fischer connector
        """
        if isinstance(lines, str):
            self.close_relay(self._to_line(lines), 9)
            self.open_relay(self._to_line(lines), 0)
        else:
            numbers = map(self._to_line, lines)
            pairs = list(itertools.zip_longest(numbers, [], fillvalue=9))
            self.close_relays(pairs)
            numbers = map(self._to_line, lines)
            connections = list(itertools.zip_longest(numbers, [], fillvalue=0))
            self.open_relays(connections)

    def connect_and_unground_all(self) -> None:
        """
        Connect all channels to the input Fischer connector and then disconnect them from the soft ground.
        """
        connects = list(itertools.zip_longest(range(1, 25), [], fillvalue=9))
        self.close_relays(connects)
        ungrounds = list(itertools.zip_longest(range(1, 25), [], fillvalue=0))
        self.open_relays(ungrounds)

    def breakout(self, line: str, tap: str) -> None:
        """
        Connect a channel to a BNC breakout connector and then disconnect them from the soft ground.

        Args:
            line (str): Channel to connect to the BNC breakout connector
            tap (str): BNC breakout connector
        """
        self.close_relay(self._to_line(line), self._to_tap(tap))
        self.open_relay(self._to_line(line), 0)

    # -----------------------------------------------------------------------
    # Naming
    # ----------------------------------------------------------------------

    def arrange(self, breakouts: Optional[Dict[str, int]] = None, lines: Optional[Dict[str, int]] = None) -> None:
        """
        An arrangement of names for lines and breakouts

        Args:
            breakouts (Dict[str, int]): Name/breakout pairs
            lines (Dict[str, int]): Name/line pairs
        """
        if lines:
            for name, line in lines.items():
                self._line_names[name] = line
        if breakouts:
            for name, tap in breakouts.items():
                self._tap_names[name] = tap

    def _set_default_names(self) -> None:
        """
        Set names of lines and breakouts to the default numbering
        """
        lines = range(1, 25)
        taps = range(1, 10)
        self._line_names = dict(zip(map(str, lines), lines))
        self._tap_names = dict(zip(map(str, taps), taps))

    # -----------------------------------------------------------------------
    # Overview functions
    # ----------------------------------------------------------------------

    def overview(self) -> dict[str, List[str]]:
        """
        Give an overview list of all channels with their connections
        """
        self._state_force_update()
        result = self._channel_list_to_overview(self._state)
        return result

    def state(self) -> str:
        """
        Gives the state of the QSwitch in the channel list notation
        """
        self._state_force_update()
        result = self._state_to_compressed_list(self._channel_list_to_state(self._state))
        return result

    def closed_relays(self) -> str:
        """
        Gives the state of the QSwitch in the State notation (Python array)
        """
        self._state_force_update()
        result = self._channel_list_to_state(self._state)
        return result

    def expand_channel_list(self, channel_list: str) -> str:
        """
        Expand the channel list notation to note all individualy closed relays
        Args:
            channel_list (str): channel list notation of closed relays
        """
        return self._state_to_expanded_list(self._channel_list_to_state(channel_list))

    def compress_channel_list(self, channel_list: str) -> str:
        """
        Compress the channel list notation as much as possible
        Args:
            channel_list (str): channel list notation of closed relays
        """
        return self._state_to_compressed_list(self._channel_list_to_state(channel_list))

    # -----------------------------------------------------------------------
    # Instrument communication
    # -----------------------------------------------------------------------

    def write(self, cmd: str):
        """
        Send SCPI command to instrument

        Args:
            cmd (str): SCPI command

        Send a SCPI command to the QSwitch, and check when ready for a new command by sending a query
        UDP connection: For relay open/close and *rst commands, only, it is checked if the command was well received
        """
        if self._udp_mode:  # UDP (ethernet) commands
            cmd_lower = cmd.lower()
            is_open_close_cmd = (
                cmd_lower.find("clos ", 0, 12) != -1
                or (cmd_lower.find("close ", 0, 12) != -1)
                or (cmd_lower.find("open ", 0, 12) != -1)
            )
            is_rst_cmd = cmd_lower == "*rst"
            counter = 0
            while True:
                self._write(cmd)
                # Check that relay command was well received
                if is_open_close_cmd or is_rst_cmd:
                    if (counter > 0) and self.verbose:
                        self.log(
                            f"{datetime.now()} UDP write repeat {counter} [{cmd}]"
                        )  # log repetition if verbose=True
                    if is_open_close_cmd:
                        splitcmd = cmd.split(" ")  # split command name and channel representation
                        reply = self.query(
                            splitcmd[0] + "? " + splitcmd[1] if len(splitcmd) == 2 else ""
                        )  # use the written command as a query to verify state
                        if (len(reply) > 0) and (reply.find("0") == -1):  # verify that the relays have switched
                            return
                    elif is_rst_cmd:
                        reply = self.query("clos:stat?")
                        if reply == "(@1!0:24!0)":  # verify that the relays are in the default state
                            return
                    counter += 1
                    if self.verbose:
                        self.log(f"{datetime.now()} UDP: {counter} failed check of [{cmd_lower}], result: {reply}")
                    if counter >= self._config.write_attempts:  # throw error when max attempts is reached
                        raise ValueError(
                            f"QSwitch {self._config.ip} (UDP): Command check failure [{cmd_lower}] after {self._config.write_attempts} attempts"
                        )
                else:
                    self.query("*opc?")
                    return
        else:  # VISA (USB) commands
            try:
                self._write(cmd)
                self._query("*opc?")
            except Exception as e:
                if self.verbose:
                    self.log(f"{datetime.now()} VISA error: {repr(e)}")
                raise ValueError(f"QSwitch VISA error: {repr(e)}")
            return

    def query(self, cmd: str) -> str:
        """
        Send a SCPI query to the QSwitch

        Args:
            cmd (str): SCPI query command

        UDP: Repeat query until a reply is received
        """
        if self._udp_mode:  # UDP (ethernet) queries
            if self._record_commands:
                self._scpi_sent.append(cmd)
            counter = 0
            time_before_next = 0.1
            while True:
                try:
                    self.clear()
                    time_before = datetime.now()
                    self._sock.sendto(f"{cmd}\n".encode(), (self._config.ip, 5025))
                    sleep_s(self._config.delay_s)
                    # Wait for response
                    data, _ = self._sock.recvfrom(1024)
                    answer = data.decode().strip()
                    if (counter > 0) and self.verbose:
                        self.log(f"{datetime.now()} UDP query repeat {counter} [{cmd}]")
                    return answer
                except Exception as error:
                    counter += 1
                    if self.verbose:
                        self.log(
                            f"{time_before} - {datetime.now().time()} UDP query error {counter} [{cmd}]: {repr(error)}"
                        )
                    if counter >= self._config.query_attempts:
                        raise ValueError(
                            f"QSwitch {self._config.ip} (UDP): Query timeout [{cmd}] after {self._config.query_attempts} attempts"
                        )
                    sleep_s(time_before_next)
                    time_before_next += (
                        0.5  # Next time we wait even longer so that we do not quickly run out of retries
                    )
        else:  # VISA (USB) queries
            answer = self._query(cmd)
        return answer

    def clear(self) -> None:
        """
        Function to reset the connection state for TCPIP (FW <= 1.3)
        or flush the input buffer for a UDP connection (FW >= 1.9).
        In USB-serial mode, do nothing.
        """
        if self._udp_mode:  # UDP (ethernet) clear
            self._sock.settimeout(0.0001)
            while True:
                try:
                    data, _ = self._sock.recvfrom(1024)
                except:
                    break
            self._sock.settimeout(self._config.timeout_ms / 1000)
        else:  # VISA (USB) clear
            if self._switch.resource_class == "SOCKET":
                self._switch.clear()

    def close(self):
        """
        Close the QSwitch instrument
        """
        if self._udp_mode:
            self._sock.close()
        else:
            self._switch.close()

    # ----------------------------------------------------------------------
    # Debugging and testing
    # ----------------------------------------------------------------------

    def start_recording_scpi(self) -> None:
        """
        Record all SCPI commands sent to the instrument

        Any previous recordings are removed.  To inspect the SCPI commands sent
        to the instrument, call get_recorded_scpi_commands().
        """
        self._scpi_sent: List[str] = []
        self._record_commands = True

    def get_recorded_scpi_commands(self) -> Sequence[str]:
        """
        Returns the SCPI commands sent to the instrument
        """
        commands = self._scpi_sent
        self._scpi_sent = []
        return commands

    def _set_up_debug_settings(self) -> None:
        """
        Initialize the debugging settings
        """
        self._record_commands = False
        self._scpi_sent = list()
        self._message_flush_timeout_ms = 1
        self._round_off = None

    def _check_for_wrong_model(self) -> None:
        """
        Check if the instrument is a QSwitch
        """
        model = self.query("*IDN?").split(",")[1]
        if model != "QSwitch":
            raise ValueError(f"Unknown model {model}. Are you using the right" " driver for your instrument?")

    def _check_for_incompatible_firmware(self) -> None:
        """
        Check if the firmware is 0.178 or above
        """
        firmware = self.query("*IDN?").split(",")[3]
        least_compatible_fw = "0.178"
        if parse(firmware) < parse(least_compatible_fw):
            raise ValueError(f"Incompatible firmware {firmware}. You need at " f"least {least_compatible_fw}")

    # ----------------------------------------------------------------------
    # Supporting functions
    # ----------------------------------------------------------------------

    def _write(self, cmd: str) -> None:
        """
        Write SCPI command to QSwitch

        Args:
            cmd (str): SCPI command
        """
        if self._record_commands:
            self._scpi_sent.append(cmd)

        if self._udp_mode:  # UDP (ethernet) write
            try:
                self._sock.sendto(f"{cmd}\n".encode(), (self._config.ip, 5025))
            except Exception as e:
                self.log(f"{datetime.now()} UDP write Error: {repr(e)}")  # raise?
                raise ValueError(f"QSwitch {self._config.ip} (UDP): Write Error [{cmd}]: {repr(e)}")
        else:  # VISA (USB) write
            self._switch.write(cmd)

    def _query(self, cmd: str) -> str:
        """
        Send SCPI query to QSwitch and receive answer

        Args:
            cmd (str): SCPI query command
        """
        if self._record_commands:
            self._scpi_sent.append(cmd)

        if self._udp_mode:  # UDP (ethernet) query
            try:
                self.clear()
                self._sock.sendto(f"{cmd}\n".encode(), (self._config.ip, 5025))
                sleep_s(self._config.delay_s)
                # Wait for response
                data, _ = self._sock.recvfrom(1024)
                answer = data.decode().strip()
            except Exception as error:
                if self.verbose:
                    self.log(f"{datetime.now()} QSwitch failed UDP query [{cmd}] (1st try): {repr(error)}")
                raise ValueError(f"QSwitch failed UDP query [{cmd}] (1st try): {repr(error)}")
        else:  # VISA (USB) query
            try:
                answer = self._switch.query(cmd)
            except visa.errors.VisaIOError as error:
                if self.verbose:
                    self.log(f"{datetime.now()} QSwitch failed VISA query [{cmd}] (1st try): {repr(error)}")
                raise ValueError(f"QSwitch failed VISA query [{cmd}] (1st try): {repr(error)}")
        return answer

    def _channel_list_to_overview(self, channel_list: str) -> dict[str, List[str]]:
        """
        Convert a channel list notation into a printable overview list of all channels
        Args:
            channel_list (str):
        """
        state = self._channel_list_to_state(channel_list)
        line_names: dict[int, str] = dict()
        for name, line in self._line_names.items():
            line_names[line] = name
        tap_names: dict[int, str] = dict()
        for name, tap in self._tap_names.items():
            tap_names[tap] = name
        result: dict[str, List[str]] = dict()
        for _, line in self._line_names.items():
            line_name = line_names[line]
            result[line_name] = list()
        for line, tap in state:
            line_name = line_names[line]
            if tap == 0:
                result[line_name].append("grounded")
            elif tap == 9:
                result[line_name].append("connected")
            else:
                tap_name = f"breakout {tap_names[tap]}"
                result[line_name].append(tap_name)
        return result

    def _to_line(self, name: str) -> int:
        """
        Convert a Fischer channel name to the Fischer channel number
        Args:
            name (str): name of the channel
        """
        try:
            return self._line_names[name]
        except KeyError:
            raise ValueError(f'Unknown line "{name}"')

    def _to_tap(self, name: str) -> int:
        """
        Convert a BNC breakout name to the BNC breakout number
        Args:
            name (str): name of the breakout
        """
        try:
            return self._tap_names[name]
        except KeyError:
            raise ValueError(f'Unknown tap "{name}"')

    def _get_state(self) -> str:
        """
        Get the current state from the QSwitch
        """
        self._state_force_update()
        return self._state

    def _set_state(self, channel_list: str) -> None:
        """
        Set the QSwitch to a new state
        Args:
            channel_list (str): channel list notation of closed relays
        """
        self._effectuate(self._channel_list_to_state(channel_list))

    def _state_force_update(self) -> None:
        """
        Set the current known state (self._state) to the current state from the QSwitch
        """
        self._set_state_raw(self.query("stat?"))

    def _set_state_raw(self, channel_list: str) -> None:
        """
        Update the current known state (self._state)
        Args:
            channel_list (str): channel list notation of closed relays
        """
        self._state = channel_list

    def _effectuate(self, state: State) -> None:
        """
        Compares the current state to the requested state, and opens and closes the required relays. Then updates the known state.
        Args:
            state (State): state of the relays
        """
        currently = self._channel_list_to_state(self._state)
        positive, negative, total = self._state_diff(currently, state)
        if positive:
            self.write(f"clos {self._state_to_compressed_list(positive)}")
        if negative:
            self.write(f"open {self._state_to_compressed_list(negative)}")
        self._set_state_raw(self._state_to_compressed_list(total))

    def _line_tap_split(self, input: str) -> Tuple[int, int]:
        """
        Splits the line and tap commands from the channel list
        Args:
            input (str): line and tap numbers separated by !
        """
        pair = input.split("!")
        if len(pair) != 2:
            raise ValueError(f"Expected channel pair, got {input}")
        if not pair[0].isdecimal():
            raise ValueError(f"Expected channel, got {pair[0]}")
        if not pair[1].isdecimal():
            raise ValueError(f"Expected channel, got {pair[1]}")
        return int(pair[0]), int(pair[1])

    def _channel_list_to_state(self, channel_list: str) -> State:
        """
        Converts channel list notation to the State notation
        Args:
            channel_list (str): channel list notation of closed relays
        """
        outer = re.match(r"\(@([0-9,:! ]*)\)", channel_list)
        result: List[Tuple[int, int]] = []
        if len(channel_list) == 0:
            return result
        elif not outer:
            raise ValueError(f"Expected channel list, got {channel_list}")
        sequences = outer[1].split(",")
        for sequence in sequences:
            limits = sequence.split(":")
            if limits == [""]:
                raise ValueError(f"Expected channel sequence, got {limits}")
            line_start, tap_start = self._line_tap_split(limits[0])
            line_stop, tap_stop = line_start, tap_start
            if len(limits) == 2:
                line_stop, tap_stop = self._line_tap_split(limits[1])
            if len(limits) > 2:
                raise ValueError(f"Expected channel sequence, got {limits}")
            if tap_start != tap_stop:
                raise ValueError(f"Expected same breakout in sequence, got {limits}")
            for line in range(line_start, line_stop + 1):
                result.append((line, tap_start))
        return result

    def _state_to_expanded_list(self, state: State) -> str:
        """
        Converts state notation to a long channel list of closed relays
        Args:
            state (State): state of the relays
        """
        return "(@" + ",".join([f"{line}!{tap}" for (line, tap) in state]) + ")"

    def _state_to_compressed_list(self, state: State) -> str:
        """
        Converts state notation to a short channel list of closed relays
        Args:
            state (State): state of the relays
        """
        tap_to_line: Dict[int, Set[int]] = dict()
        for line, tap in state:
            tap_to_line.setdefault(tap, set()).add(line)
        taps = list(tap_to_line.keys())
        taps.sort()
        intervals = []
        for tap in taps:
            start_line = None
            end_line = None
            lines = list(tap_to_line[tap])
            lines.sort()
            for line in lines:
                if not start_line:
                    start_line = line
                    end_line = line
                    continue
                if line == end_line + 1:
                    end_line = line
                    continue
                if start_line == end_line:
                    intervals.append(f"{start_line}!{tap}")
                else:
                    intervals.append(f"{start_line}!{tap}:{end_line}!{tap}")
                start_line = line
                end_line = line
            if start_line == end_line:
                intervals.append(f"{start_line}!{tap}")
            else:
                intervals.append(f"{start_line}!{tap}:{end_line}!{tap}")
        return "(@" + ",".join(intervals) + ")"

    def _state_diff(self, before: State, after: State) -> Tuple[State, State, State]:
        """
        Find the differences between the current state and the required state
        Args:
            before (State): current state of closed relays
            after (State): required state of closed relays
        """
        initial = frozenset(before)
        target = frozenset(after)
        return list(target - initial), list(initial - target), list(target)


# ----------------------------------------------------------------------
# USB detection
# ----------------------------------------------------------------------


def find_qswitch_on_usb() -> str:
    signature = "04D8:00DD"
    candidates = list(list_ports.grep(signature))
    if len(candidates) == 1:
        handle = candidates[0].device
    elif len(candidates) > 1:
        raise ValueError(f"More than one device with signature {signature} found")
    else:
        raise ValueError(f"No device with signature {signature} found")
    if platform_system() == "Windows":
        if handle[:3].lower() == "com":
            handle = handle[3:]
    return f"ASRL{handle}::INSTR"
