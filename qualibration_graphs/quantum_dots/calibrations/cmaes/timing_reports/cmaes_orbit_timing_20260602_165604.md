# CMA-ES Orbit — Timing / Profiling Report

- **Generated:** 2026-06-02 16:56:04
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
| Total wall-clock runtime | 652.025 s |
| Mean time per generation | 127.424 s |
| input_stream_pushes (total) | 30 |

## Logical categories (partition of total runtime)

| Category | Total (s) | % of total |
|---|---:|---:|
| communication | 7.918 | 1.2% |
| compile_upload | 9.418 | 1.4% |
| opx_execute | 629.180 | 96.5% |
| cmaes_calculation | 0.021 | 0.0% |
| host_score_compute | 0.003 | 0.0% |
| setup_other | 4.664 | 0.7% |
| unaccounted | 0.821 | 0.1% |

## What `opx_execute` Includes

`opx_execute` starts after the host finishes the six `push_to_input_stream`
calls for one generation and ends when both `survival_target` and
`survival_control` publish the next stream-processed result item. It excludes
host-side push, fetch, score computation, CMA-ES math, compile/upload, and
session setup.

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
| result-handle items published | 2 |

| Derived metric | Value |
|---|---:|
| measured mean `opx_execute` per shot body | 314.59 us |
| measured mean `opx_execute` normalized by native XY gate | 5.52 us |
| normal/pi circuit native gates used | 1,138 / 1,123 |
| target/control x90 length | 1000 ns / 1000 ns |
| target/control initialize inferred duration | 4416.0 ns / 4416.0 ns |
| target/control measure inferred duration | 12068.0 ns / 12068.0 ns |

Each shot body is:

`reset_frame -> align -> initialize -> align -> optional x180 (pi variant only) -> depth random Clifford loop (array lookup + switch_ + XY pulse per native gate) -> align -> measure -> align -> voltage_sequence.ramp_to_zero -> align -> Cast.to_int -> save`

The initialize duration above is the macro-reported value. For the current
non-heralded balanced initialize state, simulation showed an 8.4 us waveform
span, so measured/simulated timing should be used for budget attribution rather
than that inferred property.

## Per-iteration phases (summed over all generations)

| Phase | Total (s) | % of total | Mean/gen (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| cmaes_ask | 0.008 | 0.0% | 1.66 | 0.32 | 6.38 |
| cmaes_tell | 0.012 | 0.0% | 2.48 | 0.69 | 3.76 |
| fetch | 3.237 | 0.5% | 647.35 | 642.66 | 653.59 |
| opx_execute | 629.180 | 96.5% | 125836.05 | 121325.71 | 129647.14 |
| push | 4.681 | 0.7% | 936.25 | 928.63 | 941.04 |
| score_compute | 0.003 | 0.0% | 0.69 | 0.05 | 2.41 |

## One-time setup phases

| Phase | Total (s) | % of total |
|---|---:|---:|
| build_program | 0.199 | 0.0% |
| cmaes_init | 0.002 | 0.0% |
| compile_upload | 9.418 | 1.4% |
| connect | 0.981 | 0.2% |
| generate_circuits | 0.000 | 0.0% |
| generate_config | 0.398 | 0.1% |
| session_open | 3.084 | 0.5% |

## Figure

![timing breakdown](cmaes_orbit_timing_latest.png)

## Raw report

```
========================================================================
CMA-ES ORBIT — TIMING / PROFILING REPORT
========================================================================
Total wall-clock run time :    652.025 s
Generations executed      : 5
Mean time per generation  :    127.424 s
Counter[input_stream_pushes] = 30
Counter[candidates_evaluated] = 50

-- One-time setup phases -----------------------------------------------
  build_program               0.199 s     0.0%   
  cmaes_init                  0.002 s     0.0%   
  compile_upload              9.418 s     1.4%   
  connect                     0.981 s     0.2%   
  generate_circuits           0.000 s     0.0%   
  generate_config             0.398 s     0.1%   
  session_open                3.084 s     0.5%   

-- Per-iteration phases (summed over all generations) ------------------
  phase                       total     %tot     mean/gen       min       max
  cmaes_ask                   0.008 s     0.0%        1.66ms     0.32ms     6.38ms
  cmaes_tell                  0.012 s     0.0%        2.48ms     0.69ms     3.76ms
  fetch                       3.237 s     0.5%      647.35ms   642.66ms   653.59ms
  opx_execute               629.180 s    96.5%   125836.05ms 121325.71ms 129647.14ms
  push                        4.681 s     0.7%      936.25ms   928.63ms   941.04ms
  score_compute               0.003 s     0.0%        0.69ms     0.05ms     2.41ms

-- Logical categories --------------------------------------------------
  communication               7.918 s     1.2%   
  compile_upload              9.418 s     1.4%   
  opx_execute               629.180 s    96.5%   
  cmaes_calculation           0.021 s     0.0%   
  host_score_compute          0.003 s     0.0%   
  setup_other                 4.664 s     0.7%   
  unaccounted                 0.821 s     0.1%   
========================================================================
```
