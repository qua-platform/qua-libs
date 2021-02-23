"""bakary.py: Generating shorter arbitrary waveforms to be inserted into QUA program
Author: Arthur Strauss - Quantum Machines
Created: 23/02/2021
"""

from typing import Set, List, Union
import numpy as np
from typing import Iterable
from qm import qua


def flatten(items):
    """Yield items from any nested iterable"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def baking(config):
    return Baking(config)


def get_baking_index(config: dict):
    index = 0
    for pulse in list(config["pulses"].keys()):
        if pulse.find("baked") != -1:  # Check if pulse comes from a previous baking
            index += 1
    return index


def generate_samples_dict(config: dict):
    sample_dict = {}
    for qe in config["elements"].keys():
        if "mixInputs" in config["elements"][qe]:
            sample_dict[qe] = {"I": [],
                               "Q": []}
        elif "singleInput" in config["elements"][qe]:
            sample_dict[qe] = []
    return sample_dict


class Baking:

    def __init__(self, config):
        self._config = config
        self._local_config = {}
        self._local_config.update(config)
        self._qe_dict = {qe: {"time": 0,
                              "phase": 0
                              } for qe in self._local_config["elements"].keys()
                         }
        self._seq = []
        # self._qe_set = set()
        self._samples_dict = generate_samples_dict(self._local_config)
        self._ctr = get_baking_index(self._local_config)  # unique name counter

        print('started bake object')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Updates the configuration dictionary upon exit
        """
        if exc_type:
            return

        # my goal here is to build arb wf per qe

        # collect QEs (make a list of QE) -> done while building the seq

        # dictionary of lists of samples per QE

        # align (not trivial) need to figure out how many zeros to pad and where

        # add samples to each QE to make a multiple of 4

        for qe in self._local_config["elements"].keys():
            if not self._qe_dict[qe]["time"] % 4 == 0:
                self.wait(4 - self._qe_dict[qe]["time"] % 4, qe)

        for qe in self._samples_dict.keys():
            qe_samples = self._samples_dict[qe]
            if self._qe_dict[qe]["time"] > 0:
                self._config["elements"][qe]["operations"][f"baked_Op_{self._ctr}"] = f"{qe}_baked_pulse_{self._ctr}"
                if "I" in qe_samples:
                    self._config["pulses"][f"{qe}_baked_pulse_{self._ctr}"] = {'operation': 'control',
                                                                               'length': len(qe_samples["I"]),
                                                                               'waveforms': {
                                                                                    'I': f"{qe}_baked_wf_I_{self._ctr}",
                                                                                    'Q': f"{qe}_baked_wf_Q_{self._ctr}"
                                                                                }
                                                                               }
                    self._config["waveforms"][f"{qe}_baked_wf_I_{self._ctr}"] = {
                        "type": "arbitrary",
                        "samples": qe_samples["I"]
                    }
                    self._config["waveforms"][f"{qe}_baked_wf_Q_{self._ctr}"] = {
                        "type": "arbitrary",
                        "samples": qe_samples["Q"]
                    }

                elif type(qe_samples) == list:
                    self._config["pulses"][f"{qe}_baked_pulse_{self._ctr}"] = {'operation': 'control',
                                                                               'length': len(qe_samples["I"]),
                                                                               'waveforms': {
                                                                                   'single': f"{qe}_baked_wf_{self._ctr}"
                                                                                }
                                                                               }
                    self._config["waveforms"][f"{qe}_baked_wf_{self._ctr}"] = {
                        "type": "arbitrary",
                        "samples": qe_samples
                    }

    def play(self, Op: str, qe: str) -> None:
        """
        Add a pulse to the bake sequence
        :param Op: operation to play to quantum element
        :param qe: Quantum element to play to
        :return:
        """
        try:
            if Op in self._local_config['elements'][qe]["operations"].keys():
                pulse = self._local_config["elements"][qe]["operations"][Op]
                samples = self._get_samples(pulse)
                phi = self._qe_dict[qe]["phase"]
                if "mixInputs" in self._local_config["elements"][qe].keys():
                    if type(samples[0]) != list:
                        print(f"Error : samples given do not correspond to mixInputs for element {qe} ")
                    I = samples[0]
                    Q = samples[1]
                    I2 = [None] * len(I)
                    Q2 = [None] * len(Q)

                    for i in range(len(I)):
                        I2[i] = np.cos(phi)*I[i] - np.sin(phi)*Q[i]
                        Q2[i] = np.sin(phi)*I[i] + np.cos(phi)*Q[i]
                        self._samples_dict[qe]["I"].append(I2[i])
                        self._samples_dict[qe]["Q"].append(Q2[i])

                    self._update_qe_time(qe, len(I))

                elif "singleInput" in self._local_config["elements"][qe].keys():
                    if type(samples[0]) == list:
                        print(f"Error : samples given do not correspond to singleInput for element {qe} ")
                    for sample in samples:
                        self._samples_dict[qe].append(np.cos(phi)*sample)
                    self._update_qe_time(qe, len(samples))

        except KeyError:
            raise KeyError(f'Op:"{Op}" does not exist in configuration and not manually added (use add_pulse)')

    def frame_rotation(self, angle, qe):
        """
        Shift the phase of the oscillator associated with a quantum element by the given angle.
        This is typically used for virtual z-rotations.
        """
        self._update_qe_phase(qe, angle)

    def reset_frame(self, qe_set: Set[str]):
        """
        Used to reset all of the frame updated made up to this statement.
        :param qe_set: Set[str] of quantum elements
        """
        for qe in qe_set:
            self._update_qe_phase(qe, 0.)

    def ramp(self, amp: float, duration: int, qe: str):
        """
        Analog of ramp function in QUA
        :param amp: slope
        :param duration: duration of ramping
        :param qe: quantum element
        """
        ramp_samp = [amp*t for t in range(duration)]
        if "singleInput" in self._local_config["elements"][qe]:
            self._samples_dict[qe] += ramp_samp
        elif "mixInputs" in self._local_config["elements"][qe]:
            self._samples_dict[qe]["Q"] += ramp_samp
        self._update_qe_time(qe, duration)

    def _update_qe_time(self, qe: str, dt: int):
        self._qe_dict[qe]["time"] += dt

    def _update_qe_phase(self, qe: str, phi: float):
        self._qe_dict[qe]["phase"] = phi

    def _get_samples(self, pulse: str) -> Union[List[float], List[List]]:
        """
        Returns samples associated with a pulse
        :param pulse:
        :returns: Python list containing samples, [samples_I, samples_Q] in case of mixInputs
        """
        try:
            if 'single' in self._local_config['pulses'][pulse]['waveforms'].keys():
                wf = self._local_config['pulses'][pulse]['waveforms']['single']
                return self._local_config['waveforms'][wf]['samples']
            elif 'I' in self._local_config['pulses'][pulse]['waveforms'].keys():
                wf_I = self._local_config['pulses'][pulse]['waveforms']['I']
                wf_Q = self._local_config['pulses'][pulse]['waveforms']['Q']
                samples_I = self._local_config['waveforms'][wf_I]['samples']
                samples_Q = self._local_config['waveforms'][wf_Q]['samples']
                return [samples_I, samples_Q]

        except KeyError:
            raise KeyError(f'No waveforms found for pulse {pulse}')

    def wait(self, duration: int, qe: str):
        """
        Wait for the given duration on all provided elements.
        During the wait command the OPX will output 0.0 to the elements.

        :param duration: waiting duration
        :param qe: quantum element

        """
        if qe in self._samples_dict.keys():
            if "mixInputs" in self._local_config["elements"][qe].keys():
                self._samples_dict[qe]["I"] = self._samples_dict[qe]["I"] + [0] * duration
                self._samples_dict[qe]["Q"] = self._samples_dict[qe]["Q"] + [0] * duration

            elif "singleInput" in self._local_config["elements"][qe].keys():
                self._samples_dict[qe] = self._samples_dict[qe] + [0] * duration

        self._update_qe_time(qe, duration)

    def align(self, *qe_set: Set[str]):
        """
        Align several quantum elements together.
        All of the quantum elements referenced in *elements will wait for all
        the others to finish their currently running statement.

        :param qe_set : set of quantum elements to be aligned altogether
        """
        # self._seq.append(AlignBop(qe_set))
        last_qe = ''
        last_t = 0
        for qe in qe_set:
            qe_t = self._qe_dict[qe]["time"]

            if qe_t > last_t:
                last_qe = qe
                last_t = qe_t

        for qe in qe_set:
            qe_t = self._qe_dict[qe]["time"]
            if qe != last_qe:
                self.wait(last_t-qe_t, qe)

    def add_pulse(self, input_type: bool, samples: list):
        """
        Adds in the configuration file a pulse element.
        :param input_type: Set as True if pulse is to be applied on mixInputs,
            False for singleInput
        :param  samples: arbitrary waveform to be inserted into pulse definition
        """

        if input_type:
            pulse = {
                    f"baked_{self._ctr}": {
                        "operation": "control",
                        "length": len(samples),
                        "waveforms": {"I": f"baked_{self._ctr}_wf_I",
                                      "Q": f"baked_{self._ctr}_wf_Q"}
                    }
                }

            waveform = {
                    f"baked_{self._ctr}_wf_I": {
                        'type':
                            'arbitrary',
                        'samples': samples[0]
                    },
                    f"baked_{self._ctr}_wf_Q": {
                        'type':
                            'arbitrary',
                        'samples': samples[1]
                    }
                }

        else:
            pulse = {
                    f"baked_{self._ctr}": {
                            "operation": "control",
                            "length": len(samples),
                            "waveforms": {"single": f"baked_{self._ctr}_wf"}
                          }
                          }

            waveform = {

                            f"baked_{self._ctr}_wf": {
                                'type':
                                    'arbitrary',
                                'samples': samples
                             }
                            }

        self._local_config["pulses"].update(pulse)
        self._local_config["waveforms"].update(waveform)

    def run(self) -> None:
        """
        Plays the baked waveform
        :return: None
        """

        qe_set = set()
        for qe in self._qe_dict.keys():
            # number of QEs: if >1 we need an align between all of them. if =1, no align
            if self._qe_dict[qe]["time"] > 0:
                qe_set.add(qe)

        if len(qe_set) == 1:
            for qe in qe_set:
                qua.play(f"baked_Op_{self._ctr}", qe)

        else:
            qua.align(qe_set)
            for qe in qe_set:
                qua.play(f"baked_Op_{self._ctr}", qe)


if __name__ == '__main__':
    conf = {'elements':{},'waveforms':{},'pulses':{}}
    with baking(config=conf) as b:
        s = (np.random.random_sample(53) - 0.5).tolist()
        b.add_pulse('my_pulse', s)
        # b.add_pulse('my_pulse', [1])
        b.play('my_pulse', 'that')
        b.play('my_pulse', 'this')
        b.wait(20, 'that')
        b.wait(100, 'this')
        # b.add_pulse('my_pulse2', [1])
        # b.play('my_pulse2', 'that')
        b.align('this', 'that')
        # b.bake()
    b.run()

