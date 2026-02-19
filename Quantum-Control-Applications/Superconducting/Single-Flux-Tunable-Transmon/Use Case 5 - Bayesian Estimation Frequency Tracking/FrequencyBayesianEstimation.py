# %% Imports
from typing import Optional
import numpy as np
from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from configuration import *  # provides: config, host_ip, cluster_name, threshold, confusion_matrix, freq_vs_flux_01_quad_term, ...


u = unit(coerce_to_integer=True)

# =============================================================================
# Experiment parameters
# =============================================================================

class Params:
    # Experiment
    num_repetitions: int = 100
    detuning: int = 2 * u.MHz
    physical_detuning: int = 0 * u.MHz

    # Ramsey tau sweep (ns)
    min_wait_time_in_ns: int = 36
    max_wait_time_in_ns: int = 6000
    wait_time_step_in_ns: int = 40

    # Bayesian frequency grid (MHz)
    f_min: float = 1.0
    f_max: float = 3.0
    df: float = 0.01

    # Data collection
    keep_shot_data: bool = True

    # Execution
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


# =============================================================================
# Build QUA program
# =============================================================================

def build_qua_program(params: Params):
    """
    Build the QUA program for Bayesian frequency estimation (single qubit: Q1).

    Returns
    -------
    BayesFreq: QUA program
    meta: dict
        Useful arrays needed later (vf array, idle_times ticks, etc.)
    """
    # Frequency grid (MHz)
    v_f = np.arange(params.f_min, params.f_max + 0.5 * params.df, params.df)

    # tau sweep in QUA clock ticks (4 ns)
    idle_times = np.arange(
        params.min_wait_time_in_ns // 4,
        params.max_wait_time_in_ns // 4,
        params.wait_time_step_in_ns // 4,
    )

    # Effective detuning used in phase accumulation (MHz)
    detuning_eff = params.detuning - params.physical_detuning

    # Flux shift amplitude conversion (device-specific)
    # NOTE: this assumes freq_vs_flux_01_quad_term exists and sign makes the sqrt valid
    flux_shift = np.sqrt(-params.physical_detuning / freq_vs_flux_01_quad_term)

    # Config-derived constants
    flux_const_amp = config["waveforms"]["Q1.z.const.wf"]["sample"]
    x180_duration_ns = config["pulses"]["Q1.xy.x180_DragCosine.pulse"]["length"]
    x180_duration_ticks = x180_duration_ns // 4
    Q1_z_offset = config["controllers"]["con1"]["fems"]["5"]["analog_outputs"]["1"]["offset"]

    with program() as BayesFreq:
        # --------------------
        # Declarations
        # --------------------
        n = declare(int)
        n_st = declare_stream()

        t = declare(int)
        phase = declare(fixed)

        I = declare(fixed)
        Q = declare(fixed)

        state = declare(bool)
        state_st = declare_stream()

        # Bayesian variables
        frequencies = declare(fixed, value=v_f.tolist())
        Pf = declare(fixed, value=(np.ones(len(v_f)) / len(v_f)).tolist())

        Pf_st = declare_stream()
        estimated_frequency = declare(fixed)
        estimated_frequency_st = declare_stream()

        norm = declare(fixed)
        t_sample = declare(fixed)
        C = declare(fixed)
        rk = declare(fixed)

        f_idx = declare(int)

        # SPAM parameters from confusion matrix
        alpha = declare(fixed)
        beta = declare(fixed)
        assign(alpha, confusion_matrix[0][1] - confusion_matrix[1][0])
        assign(beta, 1 - confusion_matrix[0][1] - confusion_matrix[1][0])

        # --------------------
        # Program body
        # --------------------
        set_dc_offset("Q1.z", "single", Q1_z_offset)

        with for_(n, 0, n < params.num_repetitions, n + 1):
            save(n, n_st)

            with for_(*from_array(t, idle_times)):
                # phase = detuning_eff [Hz?] * tau [s], but implemented in fixed-point style
                assign(phase, Cast.mul_fixed_by_int(detuning_eff * 1e-9, 4 * t))

                play("x90", "Q1.xy")
                frame_rotation_2pi(phase, "Q1.xy")

                # align Z and XY timing
                wait(x180_duration_ticks, "Q1.z")
                wait(t, "Q1.xy")

                # flux pulse during the wait
                play("const" * amp(flux_shift / flux_const_amp), "Q1.z", duration=t)

                play("x90", "Q1.xy")

                # readout
                measure(
                    "readout",
                    "Q1.resonator",
                    None,
                    dual_demod.full("iw1", "iw2", I),
                    dual_demod.full("iw3", "iw1", Q),
                )
                assign(state, I > threshold)

                if params.keep_shot_data:
                    save(state, state_st)

                align()

                # conditional x180 (active reset)
                play("x180", "Q1.xy", condition=Cast.to_bool(state))

                # Bayesian update
                assign(rk, Cast.to_fixed(state) - 0.5)
                assign(t_sample, Cast.mul_fixed_by_int(1e-3, t * 4))  # tau in us

                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    assign(C, Math.cos2pi(frequencies[f_idx] * t_sample))
                    assign(Pf[f_idx], (0.5 + rk * (alpha + beta * C) * 0.99) * Pf[f_idx])

                # normalize posterior
                assign(norm, Cast.to_fixed(0.01 / Math.sum(Pf)))
                assign(norm, Math.abs(norm))
                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    assign(Pf[f_idx], Cast.mul_fixed_by_int(norm * Pf[f_idx], 100))

                align()
                reset_frame("Q1.xy")

                # posterior mean estimate
                assign(estimated_frequency, Math.dot(frequencies, Pf))

                # timestamp heartbeat
                play("x90" * amp(0), "Q1.xy", duration=4, timestamp_stream="time_stamp")

                # save posterior + reset posterior (matches your original behavior)
                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    save(Pf[f_idx], Pf_st)
                    assign(Pf[f_idx], 1 / len(v_f))

                save(estimated_frequency, estimated_frequency_st)

        # --------------------
        # Stream processing
        # --------------------
        with stream_processing():
            n_st.save("n")
            Pf_st.buffer(params.num_repetitions, len(v_f)).save("Pf")
            if params.keep_shot_data:
                state_st.buffer(params.num_repetitions, len(idle_times)).save("state")
            estimated_frequency_st.buffer(params.num_repetitions).save("estimated_frequency")

    meta = {
        "v_f": v_f,
        "idle_times_ticks": idle_times,
    }
    return BayesFreq, meta


# =============================================================================
# Run + (optional) fetch
# =============================================================================

def run_experiment(program, *, host_ip: str, cluster_name: str):
    """Compile and execute the program."""
    qmm = QuantumMachinesManager(host=host_ip, cluster_name=cluster_name)
    qm = qmm.open_qm(config)
    job = qm.execute(program)
    return job


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    params = Params()
    BayesFreq, meta = build_qua_program(params)

    job = run_experiment(BayesFreq, host_ip=host_ip, cluster_name=cluster_name)
