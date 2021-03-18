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


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 5e6,
            "operations": {
                "playOp": "constPulse",
            },
        },
    },
    "pulses": {
        "constPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
}

def get_prog():
    with program() as prog:
        r = Random()
        a= declare(fixed)
        assign(a,r.rand_fixed())
        result_str = declare_stream()
        play("playOp", "qe1")
        save(a,result_str)
        with stream_processing():
            result_str.save_all('result')

    return prog

opx = OPX(config)
station = qc.Station()
station.add_component(opx)
exp = load_or_create_experiment(experiment_name='my experiment',
                                sample_name="this sample")

meas = Measurement(exp=exp, station=station)
idp=Parameter(name='idp',set_cmd=lambda x:x )
dp=Parameter(name='dp',get_cmd=None)
meas.register_parameter(idp)  # register the first independent parameter
meas.register_parameter(dp,setpoints=(idp,))  # now register the dependent oone
with meas.run() as datasaver:
    for n in range(8):
        opx.simulate_prog(get_prog())
        print(opx.get_res())
        idp.set(n)
        datasaver.add_result((dp,opx.get_res()),(idp,n))
    dataset = datasaver.dataset

plot_dataset(dataset)