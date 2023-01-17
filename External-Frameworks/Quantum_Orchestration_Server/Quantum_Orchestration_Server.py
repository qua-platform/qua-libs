from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from fastapi import FastAPI, APIRouter, Depends, BackgroundTasks
import uvicorn
from pydantic import BaseModel, IPvAnyAddress, Field
from typing import Optional, Union
from configuration import *
from qm.simulate.credentials import create_credentials
from typing import List
import time

# host = '172.16.2.103'
# port = 81
host = "theo-4c195fa0.dev.quantum-machines.co"
port = 443
class Quantum_Orchestration_Server:
    def __init__(self):
        self.router = APIRouter()
        self.experiments = {}
        self.exp_gettables = {}
        self.exp_settables = {}
        self.exp_parameters = {}
        self.exp_qua = {}
        self.qmm = None
        self.job = None
        self.fetcher = None
        self.results = {}
        self.fetch_processing = False
        self.simulation_config = SimulationConfig()
        self.contruct_native_routers()

    ### This part can be signafacantly improved with input/output response models
    ### This is an MVP so I will implement just the basics
    def contruct_native_routers(self):
        self.router.add_api_route("/create_qmm", self.create_qmm, methods=["GET"],
                                  description="Create Quantum Machine Manager", tags=["Native Function"])
        self.router.add_api_route("/set_sim_config", self.create_simulation_config, methods=["GET"],
                                  description="Set the simulation configuration", tags=["Native Function"])
        self.router.add_api_route("/simulate", self.simulate, methods=["GET"],
                                  description="Simulate the qua code", tags=["Native Function"])
        self.router.add_api_route("/run", self.execute, methods=["PUT"],
                                  description="Run the qua code", tags=["Native Function"])
        self.router.add_api_route("/get_data", self.get_results, methods=["GET"],
                                  description="Get the latest data", tags=["Native Function"])
        self.router.add_api_route("/job_running", self.job_running, methods=["GET"],
                                  description="Query if the job is still running", tags=["Native Function"])
        self.router.add_api_route("/kill_job", self.halt_job, methods=["GET"],
                                  description="Kill the job", tags=["Native Function"])

    def create_qmm(self, host: str = host, port: int = port):
        if not self.qmm:
            self.qmm = QuantumMachinesManager(host=host, port=port, credentials=create_credentials())
            return "QMM Create"
        else:
            return "QMM already exists"

    def create_simulation_config(self, clk_cycles: int = 0):
        self.simulation_config = SimulationConfig(duration=clk_cycles)
        return "The simulation will run for {} clock cycles".format(clk_cycles)

    def simulate(self, exp: str, qua_name: str):
        exp_class = self.experiments[exp]['type']
        if qua_name in self.exp_qua[exp_class]:
            exp_instance = eval("self.experiments['{}']['inst']".format(exp, qua_name))
            qua_program = self.context_manager_executer(exp, qua_name)
            self.job = self.qmm.simulate(exp_instance.config, qua_program, self.simulation_config)
            # self.job.result_handles.wait_for_all_values()
            sim_res = self.job.get_simulated_samples()
            results_dict = {}
            for con in sim_res.__dict__.keys():
                results_dict[con] = {"analog": {}, "digital": {}}
                for chan in sim_res.__dict__[con].analog:
                    results_dict[con]["analog"][chan] = sim_res.__dict__[con].analog[chan].tolist()
                for chan in sim_res.__dict__[con].digital:
                    results_dict[con]["digital"][chan] = sim_res.__dict__[con].digital[chan].tolist()
            return results_dict
        else:
            return "Error: QUA code doesn't exist"

    def job_running(self):
        return self.fetch_processing

    def halt_job(self):
        if self.job is not None:
            if self.fetcher is not None:
                self.job.halt()
        return True

    def get_results(self):
        return self.results

    def execute(self, exp: str, qua_name: str, background_tasks: BackgroundTasks):
        exp_class = self.experiments[exp]['type']
        if qua_name in self.exp_qua[exp_class]:
            exp_instance = eval("self.experiments['{}']['inst']".format(exp, qua_name))
            qua_program = self.context_manager_executer(exp, qua_name)
            qm = self.qmm.open_qm(exp_instance.config)
            self.job = qm.execute(qua_program)
            result_handles = list(qos.job.result_handles._all_results.keys())
            self.fetcher = fetching_tool(self.job, data_list=result_handles, mode="live")
            self.fetch_processing = True
            background_tasks.add_task(self.fetch_results, result_handles)
        else:
            return "Error: QUA code doesn't exist"

    def fetch_results(self, data_list):
        while self.fetch_processing:
            self.fetch_processing = self.fetcher.is_processing()
            data = self.fetcher.fetch_all()
            for i, label in enumerate(data_list):
                self.results[label] = data[i].tolist()
            time.sleep(0.1)

    def add_experiment(self, experiment_class, name: str, configuration: dict):
        exp_type = experiment_class.__name__
        instance = experiment_class(configuration)
        self.experiments[name] = {"type": exp_type, "inst": instance}
        self.build_experiment_api(name)

    def build_experiment_api(self, name):
        inst = self.experiments[name]['inst']
        type = self.experiments[name]['type']
        if type in self.exp_gettables.keys():
            for gettable in self.exp_gettables[type]:
                func = getattr(inst, gettable)
                self.router.add_api_route("/{}/{}".format(name, gettable), func, methods=["GET"], tags=[type])
        if type in self.exp_settables.keys():
            for settable in self.exp_settables[type]:
                func = getattr(inst, settable)
                self.router.add_api_route("/{}/{}".format(name, settable), func, methods=["PUT"], tags=[type])
        if type in self.exp_parameters.keys():
            for param_i in self.exp_parameters[type]:
                Param, param = param_i
                # self.model = getattr(inst, Param)
                # self.set_model_to_none(inst, Param)
                func = self.param_sttr(inst, Param, param)
                self.router.add_api_route("/{}/{}".format(name, param), func, methods=["PATCH"], tags=[type])

    def set_model_to_none(self, instance, Parameters):
        model = getattr(instance, Parameters)
        for variable in model.__dict__["__fields__"].keys():
            model.__dict__["__fields__"][variable].default = None


    def param_sttr(self, instance, Parameters, param):
        if param not in instance.__dict__.keys():
            setattr(instance, param, getattr(instance, Parameters)())

        model = getattr(instance, Parameters)
        # self.model = model
        # for variable in model.__dict__["__fields__"].keys():
        #     model.__dict__["__fields__"][variable].default = None
        # self.model = model
        def parameter_setter(item: model = Depends()):
            stored_params = getattr(instance, param).dict()
            # stored_param_model = getattr(instance, Parameters)(**stored_params)
            stored_param_model = model(**stored_params)
            update_data = item.dict(exclude_defaults=True)
            # update_data = item.dict(exclude_unset=True)
            # self.item = item
            updated_item = stored_param_model.copy(update=update_data)
            # print(stored_params, update_data)
            setattr(instance, param, updated_item)
            return getattr(instance, param)

        return parameter_setter

    def parameter(self, instance_parameter):
        def decorator(func):
            def wrapper(*args, **kwargs):
                experiment, Parameter = func.__qualname__.split(".")
                if experiment not in self.exp_parameters.keys():
                    self.exp_parameters[experiment] = []
                self.exp_parameters[experiment].append([Parameter, instance_parameter])
                return func

            return wrapper()

        return decorator

    def get(self, func):
        def wrapper(*args, **kwargs):
            experiment, gettable = func.__qualname__.split(".")
            if experiment not in self.exp_gettables.keys():
                self.exp_gettables[experiment] = []
            self.exp_gettables[experiment].append(gettable)
            return func

        return wrapper()

    def set(self, func):
        def wrapper(*args, **kwargs):
            experiment, settable = func.__qualname__.split(".")
            if experiment not in self.exp_settables.keys():
                self.exp_settables[experiment] = []
            self.exp_settables[experiment].append(settable)
            return func

        return wrapper()

    def qua_code(self, func):
        def wrapper(*args, **kwargs):
            experiment, qua_code = func.__qualname__.split(".")
            if experiment not in self.exp_qua.keys():
                self.exp_qua[experiment] = []
            self.exp_qua[experiment].append(qua_code)
            return func

        return wrapper()

    def context_manager_executer(self, exp, qua_code):
        with program() as qua_program:
            getattr(self.experiments[exp]["inst"], qua_code)()
        return qua_program


qos = Quantum_Orchestration_Server()

#
# class ODMR:
#     @qos.parameter("params")
#     class Parameters(BaseModel):
#         # start_frequency: Optional[int] = Field(default=None, gt=-400e6, st=400e6,
#         #                                        title="Starting frequency of ODMR experiment")
#         # stop_frequency: Optional[int] = Field(default=None, gt=-400e6, st=400e6,
#         #                                       title="Stopping frequency of ODMR experiment")
#         # step_frequency: Optional[int] = Field(default=None, gt=0e6, st=400e6,
#         #                                       title="Stepping frequency of ODMR experiment")
#         # n_avg: Optional[int] = Field(default=None, gt=0, title="Number of times to repeat the ODMR experiment")
#         start_frequency: Optional[int] = Field(default=-250e6, gt=-400e6, st=400e6,
#                                                title="Starting frequency of ODMR experiment")
#         stop_frequency: Optional[int] = Field(default=250e6, gt=-400e6, st=400e6,
#                                               title="Stopping frequency of ODMR experiment")
#         step_frequency: Optional[int] = Field(default=10e6, gt=0e6, st=400e6,
#                                               title="Stepping frequency of ODMR experiment")
#         n_avg: Optional[int] = Field(default=10000, gt=0, title="Number of times to repeat the ODMR experiment")
#         # # start_frequency: Optional[int] = int(-250e6)
#         # stop_frequency: Optional[int] = int(250e6)
#         # step_frequency: Optional[int] = int(10e6)
#         # n_avg: Optional[int] = int(1e6)
#
#     def __init__(self, configuration):
#         self.config = configuration
#         self.params = self.Parameters(**{"start_frequency": -250000000, "stop_frequency": 250000000, "step_frequency": 10000000, "n_avg": 10000})
#
#     @qos.qua_code
#     def odmr(self):
#         f_vec = np.arange(self.params.start_frequency, self.params.step_frequency + 0.1, self.params.step_frequency)
#         times = declare(int, size=100)
#         counts = declare(int)  # variable for number of counts
#         counts_st = declare_stream()  # stream for counts
#         f = declare(int)  # frequencies
#         n = declare(int)  # number of iterations
#         n_st = declare_stream()  # stream for number of iterations
#
#         with for_(n, 0, n < self.params.n_avg, n + 1):
#             with for_(f, self.params.start_frequency, f <= self.params.stop_frequency, f + self.params.step_frequency):
#                 update_frequency("NV", f)  # update frequency
#                 align()  # align all elements
#                 play("cw", "NV", duration=int(1000 // 4))  # play microwave pulse
#                 play("laser_ON", "AOM", duration=int(1000 // 4))
#                 measure("long_readout", "SPCM", None, time_tagging.analog(times, 1000, counts))
#
#                 save(counts, counts_st)  # save counts on stream
#                 save(n, n_st)  # save number of iteration inside for_loop
#
#         with stream_processing():
#             counts_st.buffer(len(f_vec)).average().save("counts")
#             n_st.save("iteration")
#
#     @qos.set
#     def set(self, val):
#         self.a = val
#
#     @qos.get
#     def get(self):
#         return self.a
#
#
# class Time_Resolved:
#     @qos.parameter("params")
#     class Parameters(BaseModel):
#         start_time: Optional[int] = int(16)
#         stop_time: Optional[int] = int(516)
#         step_time: Optional[int] = int(10)
#         n_avg: Optional[int] = int(1e6)
#
#     def __init__(self, configuration):
#         self.config = configuration
#         self.params = self.Parameters()
#
#     def hahn_echo_pulse(self, t):
#         times = declare(int, size=100)
#         counts = declare(int)  # variable for number of counts
#         play("pi_half", "NV")
#         wait(t)
#         play("pi", "NV")
#         wait(t)
#         play("pi_half", "NV")
#         align()
#         play("laser_ON", "AOM")
#         measure("long_readout", "SPCM", None, time_tagging.analog(times, 1000, counts))
#         return counts
#
#     @qos.qua_code
#     def t2(self):
#         t_vec = np.arange(self.params.start_time, self.params.stop_time + 0.1, self.params.step_time)
#
#         counts_st = declare_stream()  # stream for counts
#         t = declare(int)  # frequencies
#         n = declare(int)  # number of iterations
#         n_st = declare_stream()  # stream for number of iterations
#
#         play("laser_ON", "AOM")
#         with for_(n, 0, n < self.params.n_avg, n + 1):
#             with for_(t, self.params.start_time, t <= self.params.stop_time, t + self.params.step_time):
#                 counts = self.hahn_echo_pulse(t)
#                 save(counts, counts_st)  # save counts on stream
#                 save(n, n_st)  # save number of iteration inside for_loop
#
#         with stream_processing():
#             counts_st.buffer(len(t_vec)).average().save("counts")
#             n_st.save("iteration")
#
#
# qos.add_experiment(ODMR, "odmr", config)
# qos.add_experiment(ODMR, "odmr2", config)
# qos.add_experiment(Time_Resolved, "tr", config)
#
# app = FastAPI()
# app.include_router(qos.router)
# uvicorn.run(app, host="127.0.0.1", port=8000)

# <in a new file
# r = requests.get(com('create_qmm'))
# r = requests.patch(com('odmr/params'), params={"start_frequency": int(-100e6), "n_avg": 10000})
# r = requests.put(com('run'), params={"exp": 'odmr', "qua_name": "odmr"})
#
# while requests.get(com("job_running")).json():
#     r = requests.get(com('get_data'))
#     print(r.json()["iteration"])
#     time.sleep(0.1)
