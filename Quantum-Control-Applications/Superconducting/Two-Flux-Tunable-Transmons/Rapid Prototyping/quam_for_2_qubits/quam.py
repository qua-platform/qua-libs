# QuAM class automatically generated using QuAM SDK (ver 0.11.0)
# open source code and documentation is available at
# https://github.com/entropy-lab/quam-sdk

from typing import List, Union
import sys
import os
from quam_sdk.classes import QuamComponent, quam_data, quam_tags


__all__ = ["QuAM"]


class _add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


@quam_data
class Network(QuamComponent):
    qop_ip: str
    cluster_name: str
    save_dir: str


@quam_data
class Qubit(QuamComponent):
    freq: float
    power: int


@quam_data
class Readout(QuamComponent):
    freq: float
    power: int


@quam_data
class Local_oscillators(QuamComponent):
    qubits: List[Qubit]
    readout: List[Readout]


@quam_data
class Mixer_correction(QuamComponent):
    offset_I: float
    offset_Q: float
    gain: float
    phase: float


@quam_data
class Wiring(QuamComponent):
    controller: str
    I: int
    Q: int
    mixer_correction: Mixer_correction


@quam_data
class Xy(QuamComponent):
    LO_index: int
    f_01: float
    anharmonicity: float
    drag_coefficient: float
    ac_stark_detuning: float
    pi_length: int
    pi_amp: float
    wiring: Wiring


@quam_data
class Filter(QuamComponent):
    iir_taps: List[Union[str, int, float, bool, list]]
    fir_taps: List[Union[str, int, float, bool, list]]


@quam_data
class Wiring2(QuamComponent):
    controller: str
    port: int
    filter: Filter


@quam_data
class Iswap(QuamComponent):
    length: int
    level: float


@quam_data
class Cz(QuamComponent):
    length: int
    level: float


@quam_data
class Z(QuamComponent):
    flux_pulse_length: int
    flux_pulse_amp: float
    max_frequency_point: float
    min_frequency_point: float
    wiring: Wiring2
    iswap: Iswap
    cz: Cz


@quam_data
class Qubit2(QuamComponent):
    name: str
    ge_threshold: float
    T1: int
    T2: int
    T2echo: int
    xy: Xy
    z: Z


@quam_data
class Mixer_correction2(QuamComponent):
    offset_I: float
    offset_Q: float
    gain: float
    phase: float


@quam_data
class Wiring3(QuamComponent):
    controller: str
    I: int
    Q: int
    mixer_correction: Mixer_correction2


@quam_data
class Opt_weights(QuamComponent):
    weights_real: List[Union[str, int, float, bool, list]]
    weights_minus_imag: List[Union[str, int, float, bool, list]]
    weights_imag: List[Union[str, int, float, bool, list]]
    weights_minus_real: List[Union[str, int, float, bool, list]]


@quam_data
class Resonator(QuamComponent):
    name: str
    LO_index: int
    f_res: float
    f_opt: float
    depletion_time: int
    readout_pulse_length: int
    readout_pulse_amp: float
    rotation_angle: float
    readout_fidelity: float
    wiring: Wiring3
    opt_weights: Opt_weights


@quam_data
class Crosstalk(QuamComponent):
    z: List[Union[str, int, float, bool, list]]
    xy: List[Union[str, int, float, bool, list]]


@quam_data
class Global_parameters(QuamComponent):
    time_of_flight: int
    downconversion_offset_I: float
    downconversion_offset_Q: float


@quam_data
class QuAM(QuamComponent):
    qubits: List[Qubit2]
    resonators: List[Resonator]
    network: Network
    local_oscillators: Local_oscillators
    crosstalk: Crosstalk
    global_parameters: Global_parameters
