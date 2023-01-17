from qm.qua import *
from configuration import config
from pydantic import BaseModel
from Quantum_Orchestration_Server import Quantum_Orchestration_Server
from fastapi import FastAPI
import uvicorn

ip = "127.0.0.1"
port = 8000
qos = Quantum_Orchestration_Server()


class Spectroscopy:
    @qos.parameter("params")
    class Parameters(BaseModel):
        f_min: Optional[int] = int(55e6)
        f_max: Optional[int] = int(65e6)
        df: Optional[int] = int(50e3)
        n_avg: Optional[int] = int(3e3)  # Number of averaging loops
        cooldown_time: Optional[int] = int(2e3) // 4  # Resonator cooldown time in clock cycles (4ns)
        flux_settle_time: Optional[int] = int(4e3) // 4  # Resonator cooldown time in clock cycles (4ns)
        a_min: Optional[float] = -1
        a_max: Optional[float] = 0
        da: Optional[float] = 0.01

    def __init__(self, configuration):
        # If this line is omitted, the server will
        # create an instance for you, but your editor won't autocomplete.
        # For good practice, it is advised that the user independently adds
        # this line to the __init__ function.
        self.config = configuration
        self.params = self.Parameters()

    @qos.qua_code
    def resonator_spec_1D(self):
        n = declare(int)  # Averaging index
        f = declare(int)  # Resonator frequency
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n_st = declare_stream()

        with for_(n, 0, n < self.params.n_avg, n + 1):
            with for_(f, self.params.f_min, f <= self.params.f_max, f + self.params.df):
                update_frequency("resonator", f)  # Update the resonator frequency
                # Measure the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to cooldown
                wait(self.params.cooldown_time, "resonator", "flux_line")
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)
            save(n, n_st)

        with stream_processing():
            I_st.buffer(int(self.params.f_max - self.params.f_min) // self.params.df + 1).average().save("I")
            Q_st.buffer(int(self.params.f_max - self.params.f_min) // self.params.df + 1).average().save("Q")
            n_st.save("iteration")

    @qos.qua_code
    def resonator_spec_2D(self):
        n = declare(int)  # Averaging index
        f = declare(int)  # Resonator frequency
        a = declare(fixed)  # Flux amplitude pre-factor
        I = declare(fixed)
        Q = declare(fixed)
        I_st = declare_stream()
        Q_st = declare_stream()
        n_st = declare_stream()

        with for_(n, 0, n < self.params.n_avg, n + 1):
            with for_(a, self.params.a_min, a < self.params.a_max + self.params.da / 2,
                      a + self.params.da):  # Notice it's < a_max + da/2 to include a_max
                with for_(f, self.params.f_min, f <= self.params.f_max, f + self.params.df):
                    # Update the resonator frequency
                    update_frequency("resonator", f)
                    # Adjust the flux line
                    play("const" * amp(a), "flux_line")
                    wait(self.params.flux_settle_time, "resonator", "qubit")
                    # Measure the resonator
                    measure(
                        "readout",
                        "resonator",
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                    )
                    # Wait for the resonator to cooldown
                    wait(self.params.cooldown_time, "resonator")
                    # Save data to the stream processing
                    save(I, I_st)
                    save(Q, Q_st)
            save(n, n_st)

        with stream_processing():
            I_st.buffer((self.params.f_max - self.params.f_min) // self.params.df + 1).buffer(
                int((self.params.a_max - self.params.a_min) / self.params.da + 1)).average().save("I")
            Q_st.buffer((self.params.f_max - self.params.f_min) // self.params.df + 1).buffer(
                int((self.params.a_max - self.params.a_min) / self.params.da + 1)).average().save("Q")
            n_st.save("iteration")


qos.add_experiment(Spectroscopy, "spec", config)

# Create the server and add qos to it
app = FastAPI()
app.include_router(qos.router)
# Initialize the server
uvicorn.run(app, host=ip, port=port)