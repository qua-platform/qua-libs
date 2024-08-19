from ..instruments import Instruments
from ..connectivity import Connectivity
from ..connectivity.wiring_spec import WiringSpec
from ..connectivity.wiring_spec_enums import (
    WiringFrequency,
    WiringIOType,
    WiringLineType,
)
from .wire_to_spec import allocate_dc_channels, allocate_rf_channels


def allocate_wiring(connectivity: Connectivity, instruments: Instruments):
    line_type_fill_order = [
        WiringLineType.RESONATOR,
        WiringLineType.DRIVE,
        WiringLineType.FLUX,
        WiringLineType.COUPLER,
    ]

    specs = connectivity.specs
    for line_type in line_type_fill_order:
        for spec in specs:
            if spec.line_type == line_type:
                _allocate_channels(spec, instruments)


def _allocate_channels(spec: WiringSpec, instruments: Instruments):
    if spec.frequency == WiringFrequency.DC:
        allocate_dc_channels(spec, instruments)

    elif spec.frequency == WiringFrequency.RF:
        allocate_rf_channels(spec, instruments)

    else:
        raise NotImplementedError()
