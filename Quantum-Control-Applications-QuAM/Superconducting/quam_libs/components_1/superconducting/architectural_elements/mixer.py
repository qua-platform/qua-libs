from quam.components import Mixer

from quam import quam_dataclass


@quam_dataclass
class StandaloneMixer(Mixer):
    @property
    def name(self):
        for name, frequency_converter in self.parent.parent.items():
            if self.parent == frequency_converter:
                return name
        raise KeyError()