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


class Baking:
    _ctr = 0  # unique name counter

    def __init__(self, config):
        # self._config = config
        self._local_config = {}
        self._local_config.update(config)
        self._qe_time_dict = {}  # a dictionary to hold the latest play time per QE
        self._seq = []
        self._qe_set = set()
        self._samples_dict = {}

        print('started bake')

    def __enter__(self):
        print('entered start')
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

        # update config with uniquely named baked waveforms

        # remember pulse name per QE: which pulse name is played per QE

        print('entered exit')

    def play(self, pulse: str, qe: str) -> None:
        '''
        Add a pulse to the bake sequence
        :param pulse: pulse to play
        :param qe: Quantum element to play to
        :return:
        '''
        try:
            if pulse in self._local_config['pulses'].keys():
                self._seq.append(PlayBop(pulse, qe))
                samples = self._get_samples(pulse)
                self._samples_dict[qe] = samples
                self._update_qe_time(qe, len(samples))
                print(self._samples_dict)
                print(self._qe_time_dict)
                self._update_qe_set()
        except KeyError:
            raise KeyError(f'Pulse:"{pulse}" does not exist in configuration and not manually added (use add_pulse)')

    def _gen_pulse_name(self) -> str:
        Baking._ctr = Baking._ctr + 1
        return f"b_wf_{Baking._ctr}"

    def _update_qe_set(self):
        self._qe_set = self._get_qe_set()

    def _get_qe_set(self) -> Set[str]:

        return set(flatten([el.qe for el in self._seq]))

    def _update_qe_time(self, qe: str, dt: int):
        if qe in self._qe_time_dict.keys():
            self._qe_time_dict[qe] = self._qe_time_dict[qe] + dt
        else:
            self._qe_time_dict[qe] = dt

    def _get_samples(self, pulse: str) -> Union[List[float], List[List]]:
        '''
        returns samples associated with a pulse
        :param pulse:
        :return:
        '''
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
        self._seq.append(WaitBop(duration, qe))
        if qe in self._samples_dict.keys():
            self._samples_dict[qe] = self._samples_dict[qe] + [0] * duration
        else:
            self._qe_set.add(qe)

        self._update_qe_time(qe, duration)
        print(self._samples_dict)
        print(self._qe_time_dict)
        self._qe_set = self._get_qe_set()

    def align(self, *qe_set: Set[str]):
        self._seq.append(AlignBop(qe_set))
        last_qe = ''
        last_t = 0
        for qe in qe_set:
            qe_t = self._qe_time_dict[qe]

            if qe_t > last_t:
                last_qe = qe
                last_t = qe_t

        for qe in qe_set:
            qe_t = self._qe_time_dict[qe]
            if qe != last_qe:
                self.wait(last_t-qe_t,qe)

        self._qe_set = self._get_qe_set()

    def add_pulse(self, name: str, samples: list):

        pulse = {'pulses': {name:
                                {"operation": "control",
                                 "length": len(samples),
                                 "waveforms": {"single": f"{name}_wf"}
                                 }
                            }
                 }

        waveform = {'waveforms':
                        {f"{name}_wf":
                             {'type':
                                  'arbitrary',
                              'samples': samples
                              }
                         }
                    }
        self._local_config.update(pulse)
        self._local_config.update(waveform)

    def run(self) -> None:
        '''
        Plays the baked waveform
        :return: None
        '''

        # number of QEs: if >1 we need an align between all of them. if =1, no align
        if len(self._qe_set) == 1:

            print(self._seq)
        else:
            self.align(*self._get_qe_set())
            print('aligns!')
            print(self._seq)
        # qua.play on arb pulse per QE in the qe list
        # print(self._get_qe_set())


class BOp(ABC):
    pass


@dataclass
class PlayBop(BOp):
    pulse: str
    qe: str


@dataclass
class WaitBop(BOp):
    dur: int
    qe: str


@dataclass
class AlignBop(BOp):
    qe: Set[str]


if __name__ == '__main__':
    conf = {}
    with baking(config=conf) as b:
        s = (np.random.random_sample(50) - 0.5).tolist()
        b.add_pulse('my_pulse', s)
        # b.add_pulse('my_pulse', [1])
        b.play('my_pulse', 'that')
        # b.play('my_pulse', 'this')
        b.wait(20, 'that')
        b.wait(100, 'this')
        # b.add_pulse('my_pulse2', [1])
        # b.play('my_pulse2', 'that')
        b.align('this', 'that')
        b.run()
