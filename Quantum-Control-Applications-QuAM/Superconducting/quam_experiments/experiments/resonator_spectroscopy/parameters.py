from quam_experiments.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class Parameters(CommonNodeParameters, QubitsExperimentNodeParameters):
    num_averages: int = 100
    frequency_span_in_mhz: float = 30.0
    frequency_step_in_mhz: float = 0.1
