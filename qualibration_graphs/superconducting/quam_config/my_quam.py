from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam, FluxTunableCrossDriveQuam


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
class Quam(FluxTunableCrossDriveQuam):
    pass
