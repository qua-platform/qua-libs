from qm.qua import *
from quam.core import quam_dataclass, QuamRoot
from quam.components import Qubit
from typing import (
    Dict,
    Optional,
    Sequence,
    Union,
)
from dataclasses import field
from quam.utils.qua_types import (
    ScalarFloat,
    QuaVariableFloat,
)
from quam.components.pulses import (
    BaseReadoutPulse,
)
from quam.components.channels import (
    Channel,
    SingleChannel,
    InOutSingleChannel,
    MWChannel,
)
from quam.utils.pulse import add_amplitude_scale_to_pulse_name


#############################################################
## Adding integration to the measurement
#############################################################


def measure_integrated(
    self,
    pulse_name: str,
    amplitude_scale: Optional[Union[ScalarFloat, Sequence[ScalarFloat]]] = None,
    qua_var: QuaVariableFloat = None,
    stream=None,
) -> QuaVariableFloat:
    pulse: BaseReadoutPulse = self.operations[pulse_name]

    if qua_var is None:
        qua_var = declare(fixed)

    pulse_name_with_amp_scale = add_amplitude_scale_to_pulse_name(
        pulse_name, amplitude_scale
    )

    integration_weight_labels = list(pulse.integration_weights_mapping)
    measure(
        pulse_name_with_amp_scale,
        self.name,
        integration.full(integration_weight_labels, qua_var),
        adc_stream=stream,
    )
    return qua_var


InOutSingleChannel.measure_integrated = measure_integrated

#############################################################
## Qubit abstraction
#############################################################


@quam_dataclass
class HyperfineQubit(Qubit):
    shelving: SingleChannel = None
    readout: InOutSingleChannel = None


@quam_dataclass
class GlobalOperations(Qubit):
    global_mw: MWChannel = None
    ion_displacement: Channel = None


@quam_dataclass
class Quam(QuamRoot):
    qubits: Dict[str, HyperfineQubit] = field(default_factory=dict)
    global_op: GlobalOperations = None
