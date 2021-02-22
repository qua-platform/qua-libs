from abc import ABC

from dataclasses import dataclass
from typing import Set, List, Union
import numpy as np
from typing import Iterable


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
        '''
        Updates the configuration dictionary upon exit
        '''
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
            if len(qe_samples) > 0:
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


            # if "mixInputs" in self._local_config["elements"][qe].keys():
            #     self._config['waveforms'][f"{qe}_arb_{self._ctr}"] = {"type": "arbitrary", "samples": qe_samps}
            #     self._config['pulses'][f"{qe}_baked_{self._ctr}"] = {
            #     'operation': 'control',
            #     'length': len(qe_samps),
            #     'waveforms': {
            #         'I': 'x90_wf',
            #         'Q': 'x90_der_wf'
            #     }
            # },
            # TODO: update pulse
        # TODO: update ops for each QE


        # update config with uniquely named baked waveforms

        # remember pulse name per QE: which pulse name is played per QE

        print('entered exit')

    def play(self, pulse: str, qe: str) -> None:
        """
        Add a pulse to the bake sequence
        :param pulse: pulse to play
        :param qe: Quantum element to play to
        :return:
        """
        try:
            if pulse in self._local_config['pulses'].keys():
                # self._seq.append(PlayBop(pulse, qe))
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
                    for sample in samples:
                        self._samples_dict[qe].append(np.cos(phi)*sample)
                    self._update_qe_time(qe, len(samples))



                # print(self._samples_dict)
                # print(self._qe_time_dict)
                # self._update_qe_set() #TODO: do i need this on every play?
        except KeyError:
            raise KeyError(f'Pulse:"{pulse}" does not exist in configuration and not manually added (use add_pulse)')

    def frame_rotation(self, angle, qe):
        """
        Shift the phase of the oscillator associated with a quantum element by the given angle.
        This is typically used for virtual z-rotations.
        """
        self._update_qe_phase(qe, angle)

    def reset_frame(self, qe):
        self._update_qe_phase(qe, 0.)

    # def _gen_pulse_name(self) -> str:
    #     return f"b_wf_{self._ctr}"

    # def _update_qe_set(self):
    #     self._qe_set = self._get_qe_set()

    # def _get_qe_set(self) -> Set[str]:
    #
    #     return set(flatten([el.qe for el in self._seq]))

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
        # self._seq.append(WaitBop(duration, qe))
        if qe in self._samples_dict.keys():
            if "mixInputs" in self._local_config["elements"][qe].keys():
                self._samples_dict[qe]["I"] = self._samples_dict[qe]["I"] + [0] * duration
                self._samples_dict[qe]["Q"] = self._samples_dict[qe]["Q"] + [0] * duration

            elif "singleInput" in self._local_config["elements"][qe].keys():
                self._samples_dict[qe] = self._samples_dict[qe] + [0] * duration

            self._qe_set.add(qe)

        self._update_qe_time(qe, duration)
        # print(self._samples_dict)
        # print(self._qe_dict["time"])
        # self._qe_set = self._get_qe_set()

    def align(self, *qe_set: Set[str]):
        """
        Align several quantum elements together.
        All of the quantum elements referenced in *elements will wait for all
        the others to finish their currently running statement.

        Args: *qe_set : set of quantum elements to be aligned altogether
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

        # self._qe_set = self._get_qe_set()

    def add_pulse(self, input_type: bool, samples: list):
        """
        Adds in the configuration file a pulse element.
        Args : input_type: Set as True if pulse is to be applied on mixInputs,
            False for singleInput
            name: pulse name
            samples: list of arbitrary waveforms to be inserted into pulse defintion
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

    # def bake(self):
    #     '''
    #     update the configuration with the arbitrary baked waveforms
    #     :return:
    #     '''
    #     for qe in self._get_qe_set():
    #         if not self._qe_time_dict[qe] % 4 ==0:
    #             self.wait(4-self._qe_time_dict[qe] % 4,qe)
    #
    #     for qe in self._samples_dict.keys():
    #         qe_samps = self._samples_dict[qe]
    #         self._config['waveforms'][f"{qe}_arb_{Baking._ctr}"] = {"type": "arbitrary", "samples": qe_samps}
    #         # TODO: update pulse
    #     # TODO: update ops for each QE
    #
    #
    #     # for wf in self._local_config['waveforms'].keys():
    #     #     wfl = len(self._samples_dict[wf]['samples'])
    #     #     print(f"{wf}: {wfl}")
    #
    #     # self._config.update(self._local_config)

    def run(self) -> None:
        """
        Plays the baked waveform
        :return: None
        """

        # number of QEs: if >1 we need an align between all of them. if =1, no align
        if len(self._qe_set) == 1:

            for qe in self._qe_set:
                print(f'play(arb_{qe},{qe})')

        else:
            print('aligns!')
            qeset = list(self._qe_set)
            print(f"align(*{qeset})")
            for qe in self._qe_set:
                print(f'play(arb_{qe},{qe})')
        # qua.play on arb pulse per QE in the qe list
        # print(self._get_qe_set())


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

