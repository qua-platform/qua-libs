import numpy as np


class baking:
    def __init__(self, config):
        self._config = config
        self._seq = []
        print('started bake')

    def __enter__(self):
        print('entered start')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        '''
        Updates the configuration dictionary upon exit
        '''
        waveform = {'waveforms':
                        {'arb_qe1':
                             {'type':
                                  'arbitrary',
                              'samples': (np.random.random_sample(100) - 0.5).tolist()
                              }
                         }
                    }
        self._config.update(waveform)
        print(self._config)
        print('entered exit')

    def play(self, pulse: str, qe: str) -> None:
        '''
        Add a pulse to the bake sequence
        :param pulse: pulse to play
        :param qe: Quantum element to play to
        :return:
        '''

        if pulse in self._config['pulses'].keys():
            self._seq.append((pulse, qe))
        else:
            raise KeyError(f'Pulse:"{pulse}" does not exist in configuration')

    def add_pulse(self, name: str, samples: list):
        pulse = {'pulses':{name:
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
        self._config.update(pulse)
        self._config.update(waveform)

    def run(self) -> None:
        '''
        Plays the baked waveform
        :return: None
        '''
        print(self._seq)


if __name__ == '__main__':
    config = {}
    with baking(config=config) as b:
        s = (np.random.random_sample(100) - 0.5).tolist()
        b.add_pulse('my_pulse',s)
        b.play('my_pulse', 'that')
    # b.run()
