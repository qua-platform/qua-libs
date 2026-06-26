# CMA-ES Orbit — Timing / Profiling Report

- **Generated:** 2026-06-02 17:59:03
- **Qubit pair:** `q1_q2`
- **Push mode:** batched array input streams (1 RPC/stream/gen)

## Load

| Parameter | Value |
|---|---|
| generations | 5 |
| population_size | 10 |
| num_shots | 500 |
| num_circuits | 20 |
| orbit_depth | 30 |

## Headline

| Metric | Value |
|---|---|
| Total wall-clock runtime | 596.451 s |
| Mean time per generation | 116.633 s |
| input_stream_pushes (total) | 30 |

## Logical categories (partition of total runtime)

| Category | Total (s) | % of total |
|---|---:|---:|
| communication | 7.986 | 1.3% |
| compile_upload | 9.178 | 1.5% |
| opx_execute | 575.138 | 96.4% |
| cmaes_calculation | 0.039 | 0.0% |
| host_score_compute | 0.003 | 0.0% |
| setup_other | 3.269 | 0.5% |
| unaccounted | 0.838 | 0.1% |

## What `opx_execute` Includes

`opx_execute` starts immediately after the host finishes the six `push_to_input_stream` calls for one generation. It ends when both `survival_target` and `survival_control` have published the next stream-processed result item. It excludes host-side push, fetch, score computation, CMA-ES math, compile/upload, and session setup.

### Pair `q1_q2`

| Work item per generation | Count |
|---|---:|
| shot bodies | 400,000 |
| initializes | 400,000 |
| measurements | 400,000 |
| `ramp_to_zero` calls | 400,000 |
| `reset_frame` calls | 400,000 |
| stream `save` calls | 400,000 |
| Clifford loop iterations | 12,000,000 |
| native XY gates inside Clifford sequences | 22,610,000 |
| extra pi-variant `x180` gates | 200,000 |
| total native XY gates | 22,810,000 |
| frequency updates | 40 |
| input-stream advances | 6 |

| Estimated OPX time component | Total/gen (s) | % of `opx_execute` | Basis |
|---|---:|---:|---|
| loop / align / save overhead | 8.784 | 7.6% | 400,000 bodies x 21.96 us |
| initialize | 3.360 | 2.9% | 400,000 bodies x 8.40 us simulated span |
| Clifford XY gate loop | 90.327 | 78.5% | 22,610,000 native gates x 3.995 us |
| pi-prep x180 | 0.200 | 0.2% | 200,000 gates x 1.00 us pulse |
| measure + readout | 4.827 | 4.2% | 400,000 bodies x 12.07 us inferred span |
| residual / production hierarchy | 7.529 | 6.5% | measured `opx_execute` minus estimated components |

| Derived metric | Value |
|---|---:|
| measured mean `opx_execute` per shot body | 287.57 us |
| measured mean `opx_execute` normalized by native XY gate | 5.04 us |
| normal-circuit native gates used | 1,138 |
| pi-circuit native gates used | 1,123 |
| target/control x90 length | 1000 ns / 1000 ns |
| target/control x180 length | 1000 ns / 1000 ns |
| target/control initialize inferred duration | 4416.0 ns / 4416.0 ns |
| target/control initialize estimated span | 8400.0 ns / 8400.0 ns |
| target/control measure inferred duration | 12068.0 ns / 12068.0 ns |

Each shot body is:

`reset_frame -> align -> initialize -> align -> optional x180 (pi variant only) -> depth random Clifford loop (array lookup + switch_ + XY pulse per native gate) -> align -> measure -> align -> voltage_sequence.ramp_to_zero -> align -> Cast.to_int -> save`

The initialize duration above is the macro-reported value. For the current non-heralded balanced initialize state, simulation showed an 8.4 us waveform span, so measured/simulated timing should be used for budget attribution rather than that inferred property.


## Per-iteration phases (summed over all generations)

| Phase | Total (s) | % of total | Mean/gen (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| cmaes_ask | 0.019 | 0.0% | 3.82 | 0.51 | 16.05 |
| cmaes_tell | 0.020 | 0.0% | 4.04 | 2.49 | 6.90 |
| fetch | 3.277 | 0.5% | 655.43 | 646.50 | 667.02 |
| opx_execute | 575.138 | 96.4% | 115027.51 | 110688.42 | 119003.19 |
| push | 4.709 | 0.8% | 941.83 | 936.30 | 947.88 |
| score_compute | 0.003 | 0.0% | 0.58 | 0.29 | 1.23 |

## One-time setup phases

| Phase | Total (s) | % of total |
|---|---:|---:|
| build_program | 0.148 | 0.0% |
| cmaes_init | 0.003 | 0.0% |
| compile_upload | 9.178 | 1.5% |
| connect | 0.988 | 0.2% |
| generate_circuits | 0.000 | 0.0% |
| generate_config | 0.384 | 0.1% |
| session_open | 1.745 | 0.3% |

## Figure

![timing breakdown](cmaes_orbit_timing_latest.png)

## OPX Execute Figure

![opx execute breakdown](cmaes_orbit_opx_execute_latest.png)

## Raw report

```
========================================================================
CMA-ES ORBIT — TIMING / PROFILING REPORT
========================================================================
Total wall-clock run time :    596.451 s
Generations executed      : 5
Mean time per generation  :    116.633 s
Counter[input_stream_pushes] = 30
Counter[candidates_evaluated] = 50

-- One-time setup phases -----------------------------------------------
  build_program               0.148 s     0.0%   
  cmaes_init                  0.003 s     0.0%   
  compile_upload              9.178 s     1.5%   
  connect                     0.988 s     0.2%   
  generate_circuits           0.000 s     0.0%   
  generate_config             0.384 s     0.1%   
  session_open                1.745 s     0.3%   

-- Per-iteration phases (summed over all generations) ------------------
  phase                       total     %tot     mean/gen       min       max
  cmaes_ask                   0.019 s     0.0%        3.82ms     0.51ms    16.05ms
  cmaes_tell                  0.020 s     0.0%        4.04ms     2.49ms     6.90ms
  fetch                       3.277 s     0.5%      655.43ms   646.50ms   667.02ms
  opx_execute               575.138 s    96.4%   115027.51ms 110688.42ms 119003.19ms
  push                        4.709 s     0.8%      941.83ms   936.30ms   947.88ms
  score_compute               0.003 s     0.0%        0.58ms     0.29ms     1.23ms

-- Logical categories --------------------------------------------------
  communication               7.986 s     1.3%   
  compile_upload              9.178 s     1.5%   
  opx_execute               575.138 s    96.4%   
  cmaes_calculation           0.039 s     0.0%   
  host_score_compute          0.003 s     0.0%   
  setup_other                 3.269 s     0.5%   
  unaccounted                 0.838 s     0.1%   

-- What opx_execute includes --------------------------------------------
  Starts after the six candidate input-stream pushes complete; ends when both survival result handles publish the next generation.
  Pair q1_q2:
    shot bodies/gen          : 400,000 (287.57 us per body at measured mean)
    initialize / measure/gen : 400,000 / 400,000
    Clifford steps/gen       : 12,000,000
    native XY gates/gen      : 22,810,000 (5.04 us per gate if all OPX time is normalized by gates)
    pi-prep x180 gates/gen   : 200,000
    reset/ramp/save/gen      : 400,000 / 400,000 / 400,000
    freq updates / stream advances/gen: 40 / 6
========================================================================
```
