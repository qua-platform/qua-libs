# Sticky channel corruption when playing zero-amplitude pulses on sibling channels

## Summary

Playing a zero-amplitude pulse (`amplitude_scale=0.0`) on sticky LF-FEM channels
corrupts the sticky hold of a neighboring channel on the same FEM. This is a
**simulation-level** bug observed on the OPX1000 (cluster CS_4, firmware as of 2026-02-18).

The bug manifests in the `quam-builder` `VoltageSequence` layer because
`_common_voltages_change()` plays on **every** channel in the virtual gate set —
including channels with zero voltage delta — rather than skipping no-change channels.

## Impact

Any experiment using `VoltageSequence.step_to_voltages()`, `step_to_point()`, or
the QuAM qubit macros (`qubit.empty()`, `qubit.initialize()`, etc.) where:

1. The virtual gate set contains multiple physical channels, and
2. Only a subset of channels change voltage at each step

...will have its sticky voltage levels corrupted on the channels that *did* change,
because the zero-delta plays on sibling channels reset or interfere with the
active channel's sticky hold.

## Reproduction

**Script:** `tests/simulation/mwe_sticky_zero_delta.py`

**Sequence under test (all stages identical):**

```
empty(-0.1V, 100ns) → align → init(+0.05V, 40ns) → align → 15 XY pulses (~1500ns) → align → measure(-0.05V, 200ns)
```

The init step is deliberately 40ns (short), while the XY drive takes ~1500ns.
If sticky + align work correctly, the plunger must hold at +0.05V for the full
XY window.

### Stage 1: Single channel — PASS

Play only on `plunger_1`. No other sticky channels are touched.

**Result:** Plunger holds at **+0.0500V** during the entire XY window.

```
800-  900: +0.0500    ← init pulse lands
900- 4000: +0.0500    ← sticky hold through XY (correct)
```

### Stage 2: All 6 channels (zero-delta on 5) — FAIL

Identical to Stage 1, but also plays `amplitude_scale=0.0` on the other 5 sticky
channels (`plunger_2`, `plunger_3`, `plunger_4`, `sensor_DC_1`, `sensor_DC_2`).
This mimics what `VoltageSequence._common_voltages_change()` does internally.

**Result:** Plunger briefly reaches +0.013V, then drops to **-0.0500V**, then
settles at **0.0000V** — completely wrong.

```
800-  900: +0.0130    ← init pulse partially applied
900- 1200: -0.0500    ← sticky corrupted
1300- 4000: +0.0000   ← settled at zero
```

### Stage 3: All 6 channels + LF-FEM XY drive — PASS

Same as Stage 2, but the XY drive uses an IQ channel on the **same LF-FEM**
(ports 5/6 at 100 MHz IF) instead of the MW-FEM.

**Result:** Plunger holds at **+0.0500V** — perfect.

```
800- 4000: +0.0500    ← sticky holds correctly
```

## Analysis

| Stage | XY FEM | Zero-delta plays | Plunger during XY | Result |
|-------|--------|-----------------|-------------------|--------|
| 1 | MW-FEM | None | +0.050V | PASS |
| 2 | MW-FEM | 5 sibling LF channels | -0.002V | **FAIL** |
| 3 | LF-FEM | 5 sibling LF channels | +0.050V | PASS |

The bug triggers when:
- Multiple sticky LF-FEM channels play simultaneously
- **AND** the XY drive is on the MW-FEM (cross-FEM align)

When the XY drive is on the same LF-FEM (Stage 3), the same zero-delta plays
do not corrupt the sticky hold. This suggests the corruption is related to
**cross-FEM timing/align interactions** in the simulator, not to the zero-delta
play itself in isolation.

## Plot

![Sticky zero-delta bug](./mwe_sticky_zero_delta.png)

- **Top (Stage 1):** Blue plunger holds flat at +0.05V through the orange XY window. PASS.
- **Middle (Stage 2):** Blue plunger drops to -0.05V then 0.0V during XY. FAIL.
- **Bottom (Stage 3):** Blue plunger holds flat at +0.05V with LF-FEM XY. PASS.

## Recommended fix

**In `quam-builder` (`VoltageSequence._play_step_on_channel`):** Skip the play
when `delta_v == 0` and the value is a Python float (not a QUA variable):

```python
def _play_step_on_channel(self, channel, delta_v, duration):
    ...
    if not is_qua_type(delta_v) and float(str(delta_v)) == 0.0:
        # No voltage change — skip play to avoid corrupting sticky state
        if not is_qua_type(duration):
            channel.wait(int(float(str(duration))) >> 2)
        return
    ...
```

This preserves timing (the channel still waits for the correct duration) but
avoids the zero-amplitude play that corrupts sibling sticky channels.

## Environment

- Cluster: CS_4 (172.16.33.114)
- FEM layout: MW in slots 1,2,3; LF in slots 4,5
- quam-builder: chore/examples-compile-pass branch
- qm-qua: as installed in `qualibration_graphs/quantum_dots/.venv`
