import matplotlib.pyplot as plt
import numpy as np
import qcodes as qc
from qcodes import (
    Measurement,
    experiments,
    initialise_database,
    initialise_or_create_database_at,
    load_by_guid,
    load_by_run_spec,
    load_experiment,
    load_last_experiment,
    load_or_create_experiment,
    new_experiment,
)
from qcodes.dataset.plotting import plot_dataset
from qcodes.logger.logger import start_all_logging
# from qcodes.tests.instrument_mocks import DummyInstrument, DummyInstrumentWithMeasurement

from OPX_driver import *

pulse_len = 1000
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2)},
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 5e6,
            "operations": {
                "playOp": "constPulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": pulse_len,  # in ns
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
        "readoutPulse": {
            "operation": "measure",
            "length": pulse_len,
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
            "digital_marker": "ON",
            "integration_weights": {"x": "xWeights", "y": "yWeights"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "xWeights": {
            "cosine": [1.0] * (pulse_len // 4),
            "sine": [0.0] * (pulse_len // 4),
        },
        "yWeights": {
            "cosine": [0.0] * (pulse_len // 4),
            "sine": [1.0] * (pulse_len // 4),
        },
    },
}


def get_prog(N_max=3,f_start=-100e6,f_stop=100e6,f_pts=100,voltage_pts=10):
    df=(f_stop-f_start)/f_pts
    with program() as prog:
        vn=declare(int)
        N = declare(int)
        f = declare(int)
        I = declare(fixed)
        result_str = declare_stream()
        # with for_(vn,0,vn<voltage_pts,vn+1):
        with for_(f, f_start, f < f_stop, f + df):
            update_frequency('qe1', f)
            with for_(N, 0, N < N_max, N + 1):
                measure('readoutPulse', 'qe1', None, demod.full('x', I))
                save(I, result_str)

        with stream_processing():
            result_str.buffer(N_max).map(FUNCTIONS.average()).save_all('result')

    return prog

f_pts=100
voltage_range = np.linspace(0,10,10)
opx = OPX(config)
station = qc.Station()
station.add_component(opx)
exp = load_or_create_experiment(experiment_name='my experiment',
                                sample_name="this sample")

meas = Measurement(exp=exp, station=station)

v1 = Parameter(name='voltage', set_cmd=lambda x: x)
spectrum = Parameter(name='spectrum', get_cmd=None, vals=Arrays(shape=(f_pts,)))

meas.register_parameter(v1)  # register the first independent parameter
meas.register_parameter(spectrum, setpoints=(v1,))  # now register the dependent oone


with meas.run() as datasaver:
    for v in voltage_range:
        opx.simulate_prog(get_prog(f_pts=f_pts),duration=100000)
        print(opx.get_res())
        v1.set(v)
        datasaver.add_result((spectrum, opx.get_res()), (v1, v))
    dataset = datasaver.dataset

plot_dataset(dataset)
