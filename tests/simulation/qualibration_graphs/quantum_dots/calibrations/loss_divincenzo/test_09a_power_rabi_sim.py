"""Simulation test for 09a_power_rabi."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "09a_power_rabi"


@pytest.mark.simulation
def test_power_rabi_simulation(simulation_runner):
    """Run simulation, verify area under curve is near zero for all channels."""
    result = simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "min_amp_factor": 1.0,
            "amp_factor_step": 0.5,
        },
    )
    if result is None:
        pytest.skip("simulation_runner did not return samples")

    samples, artifacts_dir = result
    if samples is None:
        pytest.skip("No simulated samples captured")

    areas = compute_area_under_curve(samples)
    append_area_to_readme(artifacts_dir, areas)

    # Diagnostic: dump waveform segment info for failing channel
    for con_name in sorted(samples.keys()):
        con = samples[con_name]
        for port_name in sorted(con.analog.keys()):
            wf = np.asarray(con.analog[port_name])
            if np.iscomplexobj(wf):
                wf = wf.real
            mean_v = float(np.mean(wf))
            if abs(mean_v) > 0.0005:
                total_area = float(np.sum(wf))
                print(
                    f"\n--- {con_name}/{port_name}: mean={mean_v:.6e}, "
                    f"total_area={total_area:.1f} V*ns, N={len(wf)} ---"
                )
                # Find segments of constant voltage
                diffs = np.diff(wf)
                change_idx = np.nonzero(diffs)[0]
                seg_starts = np.concatenate([[0], change_idx + 1])
                seg_ends = np.concatenate([change_idx + 1, [len(wf)]])
                for s, e in zip(seg_starts, seg_ends):
                    seg_v = wf[s]
                    seg_dur = e - s
                    seg_area = float(np.sum(wf[s:e]))
                    if abs(seg_v) > 1e-6:
                        print(
                            f"  [{s:6d}-{e:6d}] {seg_dur:5d} ns @ {seg_v:+.6f} V  "
                            f"area={seg_area:+.2f} V*ns"
                        )

    assert_balanced_analog_means_if_strict(areas)
