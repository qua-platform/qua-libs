from ..connectivity.wiring_spec import WiringSpec


class WirerException(Exception):
    def __init__(self, wiring_spec: WiringSpec):
        message = (
            f"Couldn't find available {wiring_spec.frequency.value} channels "
            f"satisfying the following specfication {wiring_spec.io_spec} for "
            f"the {wiring_spec.line_type.value} line."
        )
        super(WirerException, self).__init__(message)
