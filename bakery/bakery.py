"""bakery.py: Framework to generate arbitrary waveforms to be played into QUA program
Author: Arthur Strauss - Quantum Machines
Created: 23/02/2021
"""

from typing import List, Union, Tuple
import numpy as np
from qm import qua
import copy


def baking(
    config,
    padding_method="right",
    override=False,
    baking_index: int = None,
):
    """
    Opens a context manager to synthesize samples for arbitrary waveforms. The input config is updated, unless a
    baking_index is given, in which case the generated waveforms can be retrieved using the get_waveform_dict() method,
    in a format readily pluggable as an overrides argument
    :param config: config file
    :param padding_method: Method to pad 0s to format the waveform to match hardware constraint
    (>16 ns and multiple of 4), can be set to "right", "left", "symmetric_r" or "symmetric_l"
    :param override: Define if baked waveforms are overridable when using add_compiled feature (default set to False)
    :param baking_index: index of a reference baking object to impose length constraint on new baked waveform
    (default set to None). If integer is provided, input config is not updated with the waveform to be generated
    (useful for matching lengths when using waveform overriding in add_compiled feature)
    """
    return Baking(config, padding_method, override, baking_index)


class Baking:
    def __init__(
        self,
        config,
        padding_method: str = "right",
        override: bool = False,
        baking_index: int = None,
    ):
        self._config = config
        if baking_index is not None:
            self.update_config = False
        else:
            self.update_config = True
        self._padding_method = padding_method
        self._local_config = copy.deepcopy(config)
        (
            self._samples_dict,
            self._qe_dict,
            self._digital_samples_dict,
        ) = self._init_dict()
        self._ctr = self._find_baking_index(baking_index)  # unique name counter
        self._qe_set = set()
        self.override = override
        self.length_constraint = self._retrieve_constraint_length(baking_index)
        self.override_waveforms_dict = {"waveforms": {}}
        self._out = True

    def __enter__(self):
        (
            self._samples_dict,
            self._qe_dict,
            self._digital_samples_dict,
        ) = self._init_dict()
        self._out = False
        return self

    @property
    def elements(self):
        """
        Return the set of quantum elements involved in the baking
        """
        return self._qe_set

    @property
    def operations(self):
        """
        Access operations defined by the baking environment
        """
        return BakingOperations(self)

    @property
    def config(self):
        return self._config

    def _find_baking_index(self, baking_index: int = None):
        if baking_index is None:
            max_index = [-1]
            for qe in self._config["elements"].keys():
                index = [-1]
                for op in self._config["elements"][qe]["operations"]:
                    if op.find("baked") != -1:
                        index.append(int(op.split("_")[-1]))
                max_index.append(max(index))
            return max(max_index) + 1
        else:
            return baking_index

    def _init_dict(self):
        sample_dict = {}
        qe_dict = {}
        digit_samples_dict = {}
        for qe in self._config["elements"].keys():
            qe_dict[qe] = {
                "time": 0,
                "phase": 0.0,
                "freq": 0,
                "time_track": 0,
                "phase_track": [0],
                "freq_track": [0],
            }
            if "mixInputs" in self._local_config["elements"][qe]:
                sample_dict[qe] = {"I": [], "Q": []}

            elif "singleInput" in self._local_config["elements"][qe]:
                sample_dict[qe] = {"single": []}
            digit_samples_dict[qe] = []

        return sample_dict, qe_dict, digit_samples_dict

    def _update_config(self, qe, qe_samples):

        # Generates new Op, pulse, and waveform for each qe to be added in the original config file

        self._config["elements"][qe]["operations"][
            f"baked_Op_{self._ctr}"
        ] = f"{qe}_baked_pulse_{self._ctr}"
        if "I" in qe_samples:
            self._config["pulses"][f"{qe}_baked_pulse_{self._ctr}"] = {
                "operation": "control",
                "length": len(qe_samples["I"]),
                "waveforms": {
                    "I": f"{qe}_baked_wf_I_{self._ctr}",
                    "Q": f"{qe}_baked_wf_Q_{self._ctr}",
                },
            }
            self._config["waveforms"][f"{qe}_baked_wf_I_{self._ctr}"] = {
                "type": "arbitrary",
                "samples": qe_samples["I"],
                "is_overridable": self.override,
            }
            self._config["waveforms"][f"{qe}_baked_wf_Q_{self._ctr}"] = {
                "type": "arbitrary",
                "samples": qe_samples["Q"],
                "is_overridable": self.override,
            }

        elif "single" in qe_samples:
            self._config["pulses"][f"{qe}_baked_pulse_{self._ctr}"] = {
                "operation": "control",
                "length": len(qe_samples["single"]),
                "waveforms": {"single": f"{qe}_baked_wf_{self._ctr}"},
            }
            self._config["waveforms"][f"{qe}_baked_wf_{self._ctr}"] = {
                "type": "arbitrary",
                "samples": qe_samples["single"],
                "is_overridable": self.override,
            }
        if len(self._digital_samples_dict[qe]) != 0:
            self._config["pulses"][f"{qe}_baked_pulse_{self._ctr}"][
                "digital_marker"
            ] = f"{qe}_baked_digital_wf_{self._ctr}"
            if "digital_waveforms" in self._config:
                self._config["digital_waveforms"][
                    f"{qe}_baked_digital_wf_{self._ctr}"
                ] = {"samples": self._digital_samples_dict[qe]}
            else:
                self._config["digital_waveforms"] = {}
                self._config["digital_waveforms"][
                    f"{qe}_baked_digital_wf_{self._ctr}"
                ] = {"samples": self._digital_samples_dict[qe]}

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Updates the configuration dictionary upon exit
        """
        if exc_type:
            return
        self._out = True
        elements = self._local_config["elements"]
        for qe in elements:
            wait_duration = 0  # Stores the duration that has to be padded with 0s to make a valid sample for QUA
            # in original config file

            if (
                self._qe_dict[qe]["time"] > 0
            ):  # Check if a sample was added to the quantum element
                # otherwise we do not add any Op
                self._qe_set.add(qe)
                if self.length_constraint is not None:
                    assert self._qe_dict[qe]["time"] <= self.length_constraint, (
                        f"Provided length constraint (={self.length_constraint}) "
                        f"smaller than actual baked samples length ({self._qe_dict[qe]['time']})"
                    )
                    wait_duration += self.length_constraint - self._qe_dict[qe]["time"]
                    self.wait(self.length_constraint - self._qe_dict[qe]["time"], qe)
                if (
                    self._qe_dict[qe]["time"] < 16
                ):  # Sample length must be at least 16 ns long
                    wait_duration += 16 - self._qe_dict[qe]["time"]
                    self.wait(16 - self._qe_dict[qe]["time"], qe)
                if (
                    not self._qe_dict[qe]["time"] % 4 == 0
                ):  # Sample length must be a multiple of 4
                    wait_duration += 4 - self._qe_dict[qe]["time"] % 4
                    self.wait(4 - self._qe_dict[qe]["time"] % 4, qe)

                qe_samples = self._samples_dict[qe]
                end_samples = 0
                if "mixInputs" in elements[qe]:
                    end_samples = len(qe_samples["I"]) - wait_duration
                elif "singleInput" in elements[qe]:
                    end_samples = len(qe_samples["single"]) - wait_duration

                # Padding done according to desired method, can be either right, left, symmetric left or symmetric right
                if self._padding_method == "right":
                    pass
                elif self._padding_method == "left":
                    if "mixInputs" in elements[qe]:
                        qe_samples["I"] = (
                            qe_samples["I"][end_samples:]
                            + qe_samples["I"][0:end_samples]
                        )
                        qe_samples["Q"] = (
                            qe_samples["Q"][end_samples:]
                            + qe_samples["Q"][0:end_samples]
                        )
                    elif "singleInput" in elements[qe]:
                        qe_samples["single"] = (
                            qe_samples["single"][end_samples:]
                            + qe_samples["single"][0:end_samples]
                        )

                elif self._padding_method == "symmetric_l":
                    if "mixInputs" in elements[qe]:
                        qe_samples["I"] = (
                            qe_samples["I"][end_samples + wait_duration // 2 :]
                            + qe_samples["I"][0 : end_samples + wait_duration // 2]
                        )

                        qe_samples["Q"] = (
                            qe_samples["Q"][end_samples + wait_duration // 2 :]
                            + qe_samples["Q"][0 : end_samples + wait_duration // 2]
                        )

                    elif "singleInput" in elements[qe]:
                        qe_samples["single"] = (
                            qe_samples["single"][end_samples + wait_duration // 2 :]
                            + qe_samples["single"][0 : end_samples + wait_duration // 2]
                        )

                elif self._padding_method == "symmetric_r":
                    if "mixInputs" in elements[qe]:
                        qe_samples["I"] = (
                            qe_samples["I"][end_samples + wait_duration // 2 + 1 :]
                            + qe_samples["I"][0 : end_samples + wait_duration // 2 + 1]
                        )
                        qe_samples["Q"] = (
                            qe_samples["Q"][end_samples + wait_duration // 2 + 1 :]
                            + qe_samples["Q"][0 : end_samples + wait_duration // 2 + 1]
                        )
                    elif "singleInput" in elements[qe]:
                        qe_samples["single"] = (
                            qe_samples["single"][end_samples + wait_duration // 2 + 1 :]
                            + qe_samples["single"][
                                0 : end_samples + wait_duration // 2 + 1
                            ]
                        )

                if self.update_config:
                    self._update_config(qe, qe_samples)

                if "mixInputs" in elements[qe]:
                    self.override_waveforms_dict["waveforms"][
                        f"{qe}_baked_wf_I_{self._ctr}"
                    ] = qe_samples["I"]
                    self.override_waveforms_dict["waveforms"][
                        f"{qe}_baked_wf_Q_{self._ctr}"
                    ] = qe_samples["Q"]

                elif "singleInput" in elements[qe]:
                    self.override_waveforms_dict["waveforms"][
                        f"{qe}_baked_wf_{self._ctr}"
                    ] = qe_samples["single"]

    def is_out(self):
        return self._out

    def _get_samples(self, pulse: str) -> Union[List[float], List[List]]:
        """
        Returns samples associated with a pulse
        :param pulse:
        :returns: Python list containing samples, [samples_I, samples_Q] in case of mixInputs
        """
        try:
            if "single" in self._local_config["pulses"][pulse]["waveforms"]:
                wf = self._local_config["pulses"][pulse]["waveforms"]["single"]
                if self._local_config["waveforms"][wf]["type"] == "constant":
                    return [
                        self._local_config["waveforms"][wf]["sample"]
                    ] * self._local_config["pulses"][pulse]["length"]
                else:
                    return list(self._local_config["waveforms"][wf]["samples"])
            elif "I" in self._local_config["pulses"][pulse]["waveforms"]:
                wf_I = self._local_config["pulses"][pulse]["waveforms"]["I"]
                wf_Q = self._local_config["pulses"][pulse]["waveforms"]["Q"]
                if self._local_config["waveforms"][wf_I]["type"] == "constant":
                    samples_I = [
                        self._local_config["waveforms"][wf_I]["sample"]
                    ] * self._local_config["pulses"][pulse]["length"]
                else:
                    samples_I = list(self._local_config["waveforms"][wf_I]["samples"])
                if self._local_config["waveforms"][wf_Q]["type"] == "constant":
                    samples_Q = [
                        self._local_config["waveforms"][wf_Q]["sample"]
                    ] * self._local_config["pulses"][pulse]["length"]
                else:
                    samples_Q = list(self._local_config["waveforms"][wf_Q]["samples"])
                return [samples_I, samples_Q]

        except KeyError:
            raise KeyError(f"No waveforms found for pulse {pulse}")

    def get_baking_index(self):
        return self._ctr

    def get_current_length(self, qe: str):
        """
        Retrieve within the baking the current length of the waveform being created (within the baking)

        :param qe quantum element
        """
        if "mixInputs" in self._local_config["elements"][qe]:
            return len(self._samples_dict[qe]["I"])
        elif "singleInput" in self._local_config["elements"][qe]:
            return len(self._samples_dict[qe]["single"])
        else:
            raise KeyError("quantum element not in the config")

    def _get_pulse_index(self, qe):
        index = 0
        for pulse in self._local_config["pulses"]:
            if pulse.find(f"{qe}_baked_pulse_b{self._ctr}") != -1:
                index += 1
        return index

    def get_qe_set(self):
        return self._qe_set

    def get_waveforms_dict(self):
        return self.override_waveforms_dict

    def delete_baked_Op(self, *qe_set: str):
        """
        Delete in the input config of the baking object the associated baked operation and
        its associated pulse and waveform(s) for the specified quantum element
        :param qe: tuple of selected quantum elements, if no element is provided, operations are deleted for every element
        involved within the baking object
        """

        def remove_Op(q):
            if self.length_constraint is None:
                if self._out:
                    if (
                        f"baked_Op_{self._ctr}"
                        in self.config["elements"][q]["operations"]
                    ):
                        del self.config["elements"][q]["operations"][
                            f"baked_Op_{self._ctr}"
                        ]
                        del self.config["pulses"][f"{q}_baked_pulse_{self._ctr}"]
                        if "mixInputs" in self.config["elements"][q]:
                            del self.config["waveforms"][f"{q}_baked_wf_I_{self._ctr}"]
                            del self.config["waveforms"][f"{q}_baked_wf_Q_{self._ctr}"]
                        elif "singleInput" in self.config["elements"][q]:
                            del self.config["waveforms"][f"{q}_baked_wf_{self._ctr}"]
                        if "digital_waveforms" in self._config:
                            if (
                                f"{q}_baked_digital_wf_{self._ctr}"
                                in self._config["digital_waveforms"]
                            ):
                                del self.config["digital_waveforms"][
                                    f"{q}_baked_digital_wf_{self._ctr}"
                                ]

                else:
                    raise KeyError(
                        "delete_baked_Op only available outside of the context manager "
                        "(config is updated at the exit)"
                    )
            else:
                raise Warning(
                    "Operation could not be deleted because baking object does not update the config"
                )

        if len(qe_set) != 0:
            for qe in qe_set:
                remove_Op(qe)
        else:
            for qe in self._qe_dict.keys():
                remove_Op(qe)

    def get_Op_name(self, qe: str):
        """
        Get the baked operation issued from the baking object for quantum element qe
        :param qe: quantum element for which the baked operation is intended to be played on
        """
        if not (qe in self._qe_set):
            raise KeyError(
                f"{qe} is not in the set of quantum elements of the baking object "
            )
        else:
            return f"baked_Op_{self._ctr}"

    def get_Op_length(self, qe: str):
        """
        Retrieve the length of the finalized baked waveform associated to quantum element qe (outside the baking)
        :param qe: quantum element
        """
        if self.update_config:
            if not (qe in self._qe_set):
                raise KeyError(
                    f"{qe} is not in the set of quantum elements of the baking object "
                )
            else:
                if "mixInputs" in self._config["elements"][qe]:
                    return len(
                        self._config["waveforms"][f"{qe}_baked_wf_I_{self._ctr}"][
                            "samples"
                        ]
                    )
                else:
                    return len(
                        self._config["waveforms"][f"{qe}_baked_wf_{self._ctr}"][
                            "samples"
                        ]
                    )
        else:
            return self.get_current_length(qe)

    def add_digital_waveform(self, name: str, digital_samples: List[Tuple]):
        """
        Adds a digital waveform to be attached to a baked operation created using the add_Op method
        :param name: name of the digital waveform
        :param digital_samples: samples used to generate digital_waveform
        """
        if "digital_waveforms" in self._local_config:
            self._local_config["digital_waveforms"][name] = {"samples": digital_samples}
        else:
            self._local_config["digital_waveforms"] = {}
            self._local_config["digital_waveforms"][name] = {"samples": digital_samples}

    def add_Op(
        self,
        name: str,
        qe: str,
        samples: Union[List[float], List[List[float]]],
        digital_marker: str = None,
    ):
        """
        Adds an operation playable within the baking context manager.
        :param name: name of the Operation to be added for the quantum element (to be used only within the context manager)
        :param qe: targeted quantum element
        :param  samples: arbitrary waveforms to be inserted into pulse definition
        :param digital_marker: name of the digital marker sample associated to the generated pulse (assumed to be in the original config)
        """

        index = self._get_pulse_index(qe)
        Op = {name: f"{qe}_baked_pulse_b{self._ctr}_{index}"}
        pulse = {}
        waveform = {}
        if "mixInputs" in self._local_config["elements"][qe]:
            assert (
                len(samples) == 2
            ), f"{qe} is a mixInput element, two lists should be provided"
            assert len(samples[0]) == len(
                samples[1]
            ), "Error : samples provided for I and Q do not have the same length"

            pulse = {
                f"{qe}_baked_pulse_b{self._ctr}_{index}": {
                    "operation": "control",
                    "length": len(samples[0]),
                    "waveforms": {
                        "I": f"{qe}_baked_b{self._ctr}_{index}_wf_I",
                        "Q": f"{qe}_baked_b{self._ctr}_{index}_wf_Q",
                    },
                }
            }
            if digital_marker is not None:
                pulse[f"{qe}_baked_pulse_b{self._ctr}_{index}"][
                    "digital_marker"
                ] = digital_marker

            waveform = {
                f"{qe}_baked_b{self._ctr}_{index}_wf_I": {
                    "type": "arbitrary",
                    "samples": samples[0],
                },
                f"{qe}_baked_b{self._ctr}_{index}_wf_Q": {
                    "type": "arbitrary",
                    "samples": samples[1],
                },
            }

        elif "singleInput" in self._local_config["elements"][qe]:
            for i in range(len(samples)):
                assert (
                    type(samples[i]) == float or type(samples[i]) == int
                ), f"{qe} is a singleInput element, list of numbers (int or float) should be provided "

            pulse = {
                f"{qe}_baked_pulse_b{self._ctr}_{index}": {
                    "operation": "control",
                    "length": len(samples),
                    "waveforms": {"single": f"{qe}_baked_b{self._ctr}_{index}_wf"},
                }
            }

            if digital_marker is not None:
                pulse[f"{qe}_baked_pulse_b{self._ctr}_{index}"][
                    "digital_marker"
                ] = digital_marker
            waveform = {
                f"{qe}_baked_b{self._ctr}_{index}_wf": {
                    "type": "arbitrary",
                    "samples": samples,
                }
            }

        self._local_config["pulses"].update(pulse)
        self._local_config["waveforms"].update(waveform)
        self._local_config["elements"][qe]["operations"].update(Op)

    def play(self, Op: str, qe: str, amp: Union[float, Tuple[float]] = 1.0) -> None:
        """
        Add a pulse to the baked sequence
        :param Op: operation to play to quantum element
        :param qe: targeted quantum element
        :param amp: amplitude of the pulse, can be either a float or a tuple of 4 variables (similar to amp(a) or amp(v00, v01, v10, v11) in QUA)
        :return:
        """
        try:
            if self._qe_dict[qe]["time_track"] == 0:
                pulse = self._local_config["elements"][qe]["operations"][Op]
                samples = self._get_samples(pulse)
                freq = self._qe_dict[qe]["freq"]
                phi = self._qe_dict[qe]["phase"]

                if "mixInputs" in self._local_config["elements"][qe]:
                    assert isinstance(samples, list)
                    assert (
                        len(samples) == 2
                    ), f"{qe} is a mixInput element, two lists should be provided"
                    assert type(samples[0] == list) and type(samples[1] == list), (
                        f"{qe} is a mixInput element, " f"two lists should be provided"
                    )

                    assert len(samples[0]) == len(
                        samples[1]
                    ), "Error : samples provided for I and Q do not have the same length"

                    I = samples[0]
                    Q = samples[1]
                    I2 = [None] * len(I)
                    Q2 = [None] * len(Q)
                    I3 = [None] * len(I)
                    Q3 = [None] * len(Q)

                    for i in range(len(I)):
                        if type(amp) == float or type(amp) == int:
                            I2[i] = amp * I[i]
                            Q2[i] = amp * Q[i]
                        elif len(amp) != 4 or type(amp) != tuple:
                            raise IndexError(
                                "Amplitudes provided must be stored in a tuple (v00, v01, v10, v11)"
                            )
                        else:
                            I2[i] = amp[0] * I[i] + amp[1] * Q[i]
                            Q2[i] = amp[2] * I[i] + amp[3] * Q[i]
                        I3[i] = (
                            np.cos(freq * i * 1e-9 + phi) * I2[i]
                            - np.sin(freq * i * 1e-9 + phi) * Q2[i]
                        )
                        Q3[i] = (
                            np.sin(freq * i * 1e-9 + phi) * I2[i]
                            + np.cos(freq * i * 1e-9 + phi) * Q2[i]
                        )

                        self._samples_dict[qe]["I"].append(I3[i])
                        self._samples_dict[qe]["Q"].append(Q3[i])
                        self._qe_dict[qe]["phase_track"].append(phi)
                        self._qe_dict[qe]["freq_track"].append(freq)
                    self._update_qe_time(qe, len(I))

                elif "singleInput" in self._local_config["elements"][qe]:
                    for i in range(len(samples)):
                        assert (
                            type(samples[i]) == float or type(samples[i]) == int
                        ), f"{qe} is a singleInput element, list of numbers (int or float) should be provided "
                        self._samples_dict[qe]["single"].append(
                            amp * np.cos(freq * i * 1e-9 + phi) * samples[i]
                        )
                        self._qe_dict[qe]["phase_track"].append(phi)
                        self._qe_dict[qe]["freq_track"].append(freq)
                    self._update_qe_time(qe, len(samples))
                # Update of digital waveform
                if "digital_marker" in self._local_config["pulses"][pulse]:
                    digital_marker = self._local_config["pulses"][pulse][
                        "digital_marker"
                    ]
                    digital_waveform = self._local_config["digital_waveforms"][
                        digital_marker
                    ]["samples"]
                    self._digital_samples_dict[qe] += digital_waveform
            else:
                self.play_at(Op, qe, self._qe_dict[qe]["time_track"], amp)
                self._qe_dict[qe]["time_track"] = 0

        except KeyError:
            raise KeyError(
                f'Op:"{Op}" does not exist in configuration and not manually added (use add_pulse)'
            )

    def play_at(
        self, Op: str, qe: str, t: int, amp: Union[float, Tuple[float]] = 1.0
    ) -> None:
        """
        Add a waveform to the sequence at the specified time index.
        If indicated time is higher than the pulse duration for the specified quantum element,
        a wait command followed by the given waveform at indicated time (in ns) occurs.
        Otherwise, waveform is added (addition of samples) to the pre-existing sequence.
        Finally, providing a negative index starts adding the sample with a prior negative wait of t
        Note that the phase played for the newly formed sample is the one that was set before adding the new waveform
        :param Op: operation to play to quantum element
        :param qe: targeted quantum element
        :param t: Time tag in ns where the pulse should be added
        :param amp: amplitude of the pulse, can be either a float or a tuple of 4 variables (similar to amp(a) or amp(v00, v01, v10, v11) in QUA)
        :return:
        """
        if type(t) != int:
            if type(t) == float:
                t = int(t)
            else:
                raise TypeError("Provided time is not an integer")
        elif t < 0:
            self.wait(t, qe)  # Negative wait
            self.play(Op, qe, amp)
        elif t > self._qe_dict[qe]["time"]:
            self.wait(t - self._qe_dict[qe]["time"], qe)
            self.play(Op, qe, amp)
        else:
            try:
                pulse = self._local_config["elements"][qe]["operations"][Op]
                samples = self._get_samples(pulse)
                new_samples = 0
                if "mixInputs" in self._local_config["elements"][qe]:
                    assert isinstance(samples, list)
                    assert (
                        len(samples) == 2
                    ), f"{qe} is a mixInput element, two lists should be provided"
                    assert type(samples[0] == list) and type(samples[1] == list), (
                        f"{qe} is a mixInput element, " f"two lists should be provided"
                    )

                    assert len(samples[0]) == len(
                        samples[1]
                    ), "Error : samples provided for I and Q do not have the same length"

                    I, Q = samples[0], samples[1]
                    I2, Q2, I3, Q3 = (
                        [None] * len(I),
                        [None] * len(I),
                        [None] * len(I),
                        [None] * len(I),
                    )
                    phi = self._qe_dict[qe]["phase_track"]
                    freq = self._qe_dict[qe]["freq_track"]
                    for i in range(len(I)):
                        if type(amp) == float or type(amp) == int:
                            I2[i] = amp * I[i]
                            Q2[i] = amp * Q[i]
                        else:
                            if len(amp) != 4 or type(amp) != tuple:
                                raise IndexError(
                                    "Amplitudes provided must be stored in a tuple (v00, v01, v10, v11)"
                                )
                            else:
                                I2[i] = amp[0] * I[i] + amp[1] * Q[i]
                                Q2[i] = amp[2] * I[i] + amp[3] * Q[i]
                        if t + i < len(self._samples_dict[qe]["I"]):
                            I3[i] = (
                                np.cos(freq[t + i] * (t + i) * 1e-9 + phi[t + i])
                                * I2[i]
                                - np.sin(freq[t + i] * (t + i) * 1e-9 + phi[t + i])
                                * Q2[i]
                            )
                            Q3[i] = (
                                np.sin(freq[t + i] * (t + i) * 1e-9 + phi[t + i])
                                * I2[i]
                                + np.cos(freq[t + i] * (t + i) * 1e-9 + phi[t + i])
                                * Q2[i]
                            )

                            self._samples_dict[qe]["I"][t + i] += I3[i]
                            self._samples_dict[qe]["Q"][t + i] += Q3[i]
                        else:
                            phi = self._qe_dict[qe]["phase"]
                            freq = self._qe_dict[qe]["freq"]
                            I3[i] = (
                                np.cos(freq * i * 1e-9 + phi) * I2[i]
                                - np.sin(freq * i * 1e-9 + phi) * Q2[i]
                            )
                            Q3[i] = (
                                np.sin(freq * i * 1e-9 + phi) * I2[i]
                                + np.cos(freq * i * 1e-9 + phi) * Q2[i]
                            )

                            self._samples_dict[qe]["I"].append(I3[i])
                            self._samples_dict[qe]["Q"].append(Q3[i])
                            self._qe_dict[qe]["phase_track"].append(phi)
                            self._qe_dict[qe]["freq_track"].append(freq)
                            new_samples += 1

                elif "singleInput" in self._local_config["elements"][qe]:
                    if type(amp) != float and type(amp) != int:
                        raise IndexError("Amplitude must be a number")

                    for i in range(len(samples)):
                        assert (
                            type(samples[i]) == float or type(samples[i]) == int
                        ), f"{qe} is a singleInput element, list of numbers (int or float) should be provided "
                        if t + i < len(self._samples_dict[qe]):
                            phi = self._qe_dict[qe]["phase_track"][t + i]
                            freq = self._qe_dict[qe]["freq_track"][t + i]
                            self._samples_dict[qe]["single"][t + i] += (
                                amp * np.cos(freq * (t + i) * 1e-9 + phi) * samples[i]
                            )
                        else:
                            phi = self._qe_dict[qe]["phase"]
                            freq = self._qe_dict[qe]["freq"]
                            self._samples_dict[qe]["single"].append(amp * samples[i])
                            self._qe_dict[qe]["phase_track"].append(phi)
                            self._qe_dict[qe]["freq_track"].append(freq)
                            new_samples += 1

                self._update_qe_time(qe, new_samples)

            except KeyError:
                raise KeyError(
                    f'Op:"{Op}" does not exist in configuration and not manually added (use add_pulse)'
                )

    def frame_rotation(self, angle: float, qe: str):
        """
        Shift the phase of the oscillator associated with a quantum element by the given angle.
        This is typically used for virtual z-rotations. Frame rotation done within the baking sticks to the rest of the
        QUA program after its execution.
        :param angle: phase parameter
        :param qe: quantum element
        """

        self._update_qe_phase(qe, angle)

    def frame_rotation_2pi(self, angle: float, qe: str):
        """
        Shift the phase of the oscillator associated with a quantum element by the given angle.
        This is typically used for virtual z-rotations. This performs a frame rotation of 2*π*angle
        :param angle: phase parameter
        :param qe: quantum element
        """

        self._update_qe_phase(qe, 2 * np.pi * angle)

    def set_detuning(self, qe: str, freq: int):
        """Update frequency by adding detuning to original IF set in the config.
        Unlike frame rotation, the detuning will only affect the baked operation and will not stick in the element
        :param qe quantum element
        :param freq frequency of the detuning (in Hz)
        """
        self._qe_dict[qe]["freq"] = freq

    def reset_frame(self, *qe_set: str):
        """
        Used to reset all of the frame updated made up to this statement.
        :param qe_set: Set[str] of quantum elements
        """
        for qe in qe_set:
            self._update_qe_phase(qe, -self._qe_dict[qe]["phase"])

    def ramp(self, amp: float, duration: int, qe: str):
        """
        Analog of ramp function in QUA
        :param amp: slope
        :param duration: duration of ramping
        :param qe: quantum element
        """
        ramp_sample = [amp * t for t in range(duration)]
        if "singleInput" in self._local_config["elements"][qe]:
            self._samples_dict[qe]["single"] += ramp_sample
        elif "mixInputs" in self._local_config["elements"][qe]:
            self._samples_dict[qe]["Q"] += ramp_sample
            self._samples_dict[qe]["I"] += [0] * duration
        self._update_qe_time(qe, duration)

    def _update_qe_time(self, qe: str, dt: int):
        self._qe_dict[qe]["time"] += dt

    def _update_qe_phase(self, qe: str, phi: float):
        self._qe_dict[qe]["phase"] += phi

    def wait(self, duration: int, *qe_set: str):
        """
        Wait for the given duration on all provided elements.
        Here, the wait is simply adding 0 to the existing sample for a given duration.

        :param duration: waiting duration
        :param qe_set: set of quantum elements

        """
        if duration >= 0:
            for qe in qe_set:
                if qe in self._samples_dict.keys():

                    self._qe_dict[qe]["phase_track"] += [
                        self._qe_dict[qe]["phase"]
                    ] * duration
                    self._qe_dict[qe]["freq_track"] += [
                        self._qe_dict[qe]["freq"]
                    ] * duration
                    if "mixInputs" in self._local_config["elements"][qe].keys():
                        self._samples_dict[qe]["I"] += [0] * duration
                        self._samples_dict[qe]["Q"] += [0] * duration

                    elif "singleInput" in self._local_config["elements"][qe].keys():
                        self._samples_dict[qe]["single"] += [0] * duration

                self._update_qe_time(qe, duration)
        else:
            for qe in qe_set:
                # Duration is negative so just add for subtraction
                self._qe_dict[qe]["time_track"] = self._qe_dict[qe]["time"] + duration

    def align(self, *qe_set: str):
        """
        Align several quantum elements together.
        All of the quantum elements referenced in *elements will wait for all
        the others to finish their currently running statement.

        :param qe_set : set of quantum elements to be aligned altogether, if no element is passed, alignment is done
        all elements within the baking
        """

        def alignment(qe_set2):
            last_qe = ""
            last_t = 0
            for qe in qe_set2:
                qe_t = self._qe_dict[qe]["time"]
                if qe_t > last_t:
                    last_qe = qe
                    last_t = qe_t

            for qe in qe_set2:
                qe_t = self._qe_dict[qe]["time"]
                if qe != last_qe:
                    self.wait(last_t - qe_t, qe)

        if len(qe_set) == 0:
            alignment(self._qe_dict.keys())
        else:
            alignment(qe_set)

    def run(
        self, amp_array: List[Tuple] = None, trunc_array: List[Tuple] = None
    ) -> None:
        """
        Plays the baked waveform
        This method must be used within a QUA program
        :param amp_array list of tuples for amplitudes (e.g [(qe1, amp1), (qe2, amp2)] ), each amplitude must be a scalar
        :param trunc_array list of tuples for truncations (e.g [(qe1, amp1), (qe2, amp2)] ), each truncation must be a
         int or QUA int
        :return None
        """

        qe_set = self.get_qe_set()
        if len(qe_set) > 1:
            qua.align(*qe_set)
        if trunc_array is None:
            if amp_array is None:
                for qe in qe_set:
                    qua.play(f"baked_Op_{self._ctr}", qe)
                    qua.frame_rotation(self._qe_dict[qe]["phase"], qe)

            else:
                for qe in qe_set:
                    if not (qe in list(zip(*amp_array))[0]):
                        qua.play(f"baked_Op_{self._ctr}", qe)

                    else:
                        index2 = list(zip(*amp_array))[0].index(qe)
                        amp = list(zip(*amp_array))[1][index2]
                        if type(amp) == list:
                            raise TypeError(
                                "Amplitude can only be a number (either Python or QUA variable)"
                            )
                        qua.play(f"baked_Op_{self._ctr}" * qua.amp(amp), qe)
                    qua.frame_rotation(self._qe_dict[qe]["phase"], qe)

        else:
            for qe in qe_set:
                if qe not in list(zip(*trunc_array))[0]:
                    trunc_array.append((qe, None))
                index = list(zip(*trunc_array))[0].index(qe)
                trunc = list(zip(*trunc_array))[1][index]
                if amp_array is None:
                    qua.play(f"baked_Op_{self._ctr}", qe, truncate=trunc)

                else:
                    if not (qe in list(zip(*amp_array))[0]):
                        qua.play(f"baked_Op_{self._ctr}", qe, truncate=trunc)

                    else:
                        index2 = list(zip(*amp_array))[0].index(qe)
                        amp = list(zip(*amp_array))[1][index2]
                        if type(amp) == list:
                            raise TypeError(
                                "Amplitude can only be a number (either Python or QUA variable)"
                            )
                        qua.play(
                            f"baked_Op_{self._ctr}" * qua.amp(amp), qe, truncate=trunc
                        )

                qua.frame_rotation(self._qe_dict[qe]["phase"], qe)

    def _retrieve_constraint_length(self, baking_index: int = None):
        if baking_index is not None:
            for pulse in self._local_config["pulses"]:
                if pulse.find(f"baked_pulse_{baking_index}") != -1:
                    return self._local_config["pulses"][pulse]["length"]
        else:
            return None


def deterministic_run(baking_list, j):
    """
    Generates a QUA macro for a binary tree ensuring a synchronized play of operations
    listed in the various baking objects
    :param baking_list: Python list of Baking objects
    :param j: QUA int
    """

    with qua.switch_(j):
        for i in range(len(baking_list)):
            with qua.case_(i):
                baking_list[i].run()


class BakingOperations:
    def __init__(self, b: Baking) -> None:
        super().__init__()
        self._baking = b

    def __getitem__(self, qe: str):
        return self._baking.get_Op_name(qe)

    def length(self, qe):
        return self._baking.get_Op_length(qe)
