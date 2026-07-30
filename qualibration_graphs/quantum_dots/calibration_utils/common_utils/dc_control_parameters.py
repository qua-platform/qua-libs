from qualibrate.core.parameters import RunnableParameters


class DCControlParameters(RunnableParameters):
    dc_control: bool = False
    """Apply sweep center via external DC (VirtualDCSet) instead of OPX offset."""
