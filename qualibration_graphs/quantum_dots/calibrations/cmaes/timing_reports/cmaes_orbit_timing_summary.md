# CMA-ES Orbit — Timing / Profiling Summary

Phase-resolved timing analysis of the host ↔ OPX CMA-ES optimisation loop in
[`03b_cmaes_orbit_timing.py`](../03b_cmaes_orbit_timing.py), produced by the
built-in `TimingProfiler`. Each run also auto-writes
`cmaes_orbit_timing_latest.{md,png}` plus a timestamped copy to this folder.

The loop is timed in these phases:

| Phase | Meaning |
|---|---|
| `cmaes_ask` / `cmaes_tell` | CMA-ES sampling / covariance update (host) |
| `push` | host → OPX: streaming the candidate parameters in |
| `opx_execute` | OPX runs the queries (host waits for results) |
| `fetch` | OPX → host: retrieving the survival probabilities |
| `score_compute` | host-side numpy scoring |
| `compile_upload` | `qm.execute` — compile + upload + job start (one-time) |
| `connect` / `generate_config` / `build_program` / `session_open` | setup (one-time) |

---

## 1. Optimisation: batched input-stream pushes (6× fewer RPCs)

`job.push_to_input_stream()` is a **blocking gRPC round-trip**, and its latency
(~156 ms) is **per-call, independent of payload size**. The original code issued
one push per value — `6 streams × pop_size` RPCs per generation. Declaring the
input streams as `size=pop_size` **arrays** and pushing one list per stream
collapses that to `6` RPCs/generation (advanced once per generation, indexed per
candidate in QUA). Behaviour is unchanged.

A/B at a small/fast config (5 generations, pop 6, 50 shots × 5 circuits × depth 10):

```
┌─────────────────────────────┬─────────────────┬─────────────────┬──────────────┐
│           Metric            │ Single (before) │ Batched (after) │    Change    │
├─────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ Total runtime               │ 49.83 s         │ 27.97 s         │ 1.78× faster │
├─────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ push total                  │ 28.31 s         │ 4.68 s          │ 6.0× less    │
├─────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ push per generation         │ 5661 ms         │ 936 ms          │ 6.0× less    │
├─────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ push % of runtime           │ 56.8%           │ 16.7%           │ —            │
├─────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ RPCs / generation           │ 36              │ 6               │ 6× fewer     │
├─────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ Latency per RPC             │ 157 ms          │ 156 ms          │ unchanged    │
├─────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ input_stream_pushes (total) │ 180             │ 30              │ —            │
└─────────────────────────────┴─────────────────┴─────────────────┴──────────────┘
```

The identical ~156 ms/RPC before and after proves the cost is pure per-RPC
round-trip overhead; collapsing 36 → 6 RPCs yields exactly the 6.0× push
reduction. Further halving is possible by reducing the *stream count* (pack the
2 fixed + 4 int params into 2 array streams → ~2 RPCs/gen).

---

## 2. Realistic load (production measurement size)

**Config:** 5 generations · population 10 · 500 shots · 20 circuits · orbit depth 30 · pair `q1_q2`
(batched pushes). Generated 2026-06-02 17:59.

| Metric | Value |
|---|---|
| Total wall-clock runtime | **596.45 s** (9.9 min) |
| Mean time per generation | **116.63 s** |
| input_stream_pushes (total) | 30 |

### Logical categories (partition of total runtime)

| Category | Total (s) | % of total |
|---|---:|---:|
| **opx_execute** | **575.14** | **96.4%** |
| compile_upload | 9.18 | 1.5% |
| communication (push + fetch) | 7.99 | 1.3% |
| setup_other | 3.27 | 0.5% |
| cmaes_calculation | 0.04 | 0.0% |
| host_score_compute | 0.003 | 0.0% |
| unaccounted | 0.84 | 0.1% |

### What `opx_execute` includes

`opx_execute` starts after the six candidate input-stream pushes complete and
ends when both `survival_target` and `survival_control` publish the next
stream-processed generation result. It excludes host push/fetch, host scoring,
CMA-ES math, compile/upload, and session setup.

For the realistic `q1_q2` run above, one generation contains:

| OPX work item | Count / generation |
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
| stream-processed result items | 2 |

At the measured mean `opx_execute` time of 115.028 s/generation, this is
287.57 us per shot body, or 5.04 us if normalized by total native XY gates.

| Estimated OPX time component | Total/gen (s) | % of `opx_execute` | Basis |
|---|---:|---:|---|
| loop / align / save overhead | 8.784 | 7.6% | 400,000 bodies x 21.96 us |
| initialize | 3.360 | 2.9% | 400,000 bodies x 8.40 us simulated span |
| Clifford XY gate loop | 90.327 | 78.5% | 22,610,000 native gates x 3.995 us |
| pi-prep x180 | 0.200 | 0.2% | 200,000 gates x 1.00 us pulse |
| measure + readout | 4.827 | 4.2% | 400,000 bodies x 12.07 us inferred span |
| residual / production hierarchy | 7.529 | 6.5% | measured `opx_execute` minus estimated components |

Each shot body is:

`reset_frame -> align -> initialize -> align -> optional x180 (pi variant only) -> depth random Clifford loop (array lookup + switch_ + XY pulse per native gate) -> align -> measure -> align -> voltage_sequence.ramp_to_zero -> align -> Cast.to_int -> save`

The current persisted state uses non-heralded `BalancedInitializeMacro`.
Simulation showed that initialize spans 8.4 us even though the macro
`inferred_duration` property reports 4.416 us, so use measured/simulated timing
for budget attribution rather than the inferred property.

### Per-iteration phases (summed over all 5 generations)

| Phase | Total (s) | % of total | Mean/gen (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| opx_execute | 575.138 | 96.4% | 115027.51 | 110688.42 | 119003.19 |
| push | 4.709 | 0.8% | 941.83 | 936.30 | 947.88 |
| fetch | 3.277 | 0.5% | 655.43 | 646.50 | 667.02 |
| cmaes_tell | 0.020 | 0.0% | 4.04 | 2.49 | 6.90 |
| cmaes_ask | 0.019 | 0.0% | 3.82 | 0.51 | 16.05 |
| score_compute | 0.003 | 0.0% | 0.58 | 0.29 | 1.23 |

### One-time setup phases

| Phase | Total (s) | % of total |
|---|---:|---:|
| compile_upload | 9.178 | 1.5% |
| session_open | 1.745 | 0.3% |
| connect | 0.988 | 0.2% |
| generate_config | 0.384 | 0.1% |
| build_program | 0.148 | 0.0% |
| cmaes_init | 0.003 | 0.0% |
| generate_circuits | 0.000 | 0.0% |

![timing breakdown](cmaes_orbit_timing_latest.png)

![opx execute breakdown](cmaes_orbit_opx_execute_latest.png)

---

## 3. Takeaways

- **At production size the OPX execution dominates (96.4%).** Everything host-side
  — push, fetch, CMA-ES math, compile, setup — is collectively ~3.5%. The loop is
  almost entirely quantum-measurement-bound, so host-side micro-optimisation has
  little headroom *at this load*.
- **The batched-push fix still matters for low-shot / fast-eval regimes**, where it
  cut total runtime 1.78× (push 56.8% → 16.7%). It costs nothing at high load, so
  it is a strict improvement.
- **CMA-ES calculation is free** (~20 ms total). The optimiser is never the bottleneck.
- **To speed up the realistic run, target `opx_execute`**: fewer shots/circuits or a
  shallower orbit depth (the score's Fisher-information sweet spot), parallelising
  the two qubits, or trimming per-shot init/measure/ramp overhead. `compile_upload`
  (~9 s) and setup (~4 s) are fixed and amortise over more generations.

> Note: at the small/fast config the proportions invert — `compile_upload` and
> communication become the largest costs because the quantum work is ~200× smaller.
> Always read these percentages relative to the load.
