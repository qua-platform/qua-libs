# Adaptive calibration examples

This folder demonstrates [QUAlibrate's advanced calibration graph capabilities](https://qualibrate-docs.quantum-machines.co/advanced_calibration_graphs/) using superconducting-qubit calibration nodes: adaptive loops, failure branches, result-driven parameter updates, and nested subgraphs.

- [99a_demo_graph.py](99a_demo_graph.py) combines adaptive spectroscopy with power Rabi, IQ blobs, a readout optimisation subgraph, T1, and randomized benchmarking.
- [99b_demo_spectroscopy.py](99b_demo_spectroscopy.py) provides a smaller example of adaptive spectroscopy and refinement before power Rabi.

## Customisable optimisers

The added [calibration_utils/optimisers](../../calibration_utils/optimisers/) folder holds reusable callbacks that define calibration decisions. Loop conditions passed through `on=` decide whether a target needs another iteration; callbacks passed through `resolve_params=` return parameter overrides for the next iteration or connected node.

For example, [spectroscopy_optimiser.py](../../calibration_utils/optimisers/spectroscopy_optimiser.py) adjusts spectroscopy sweep ranges, drive amplitudes, or pulse lengths, and checks fit success and linewidths.

These callbacks are modular and customisable: reuse them across graphs, replace their thresholds, or combine additional fit metrics and more complex optimisation algorithms. Keeping this logic separate lets the same measurement nodes support different calibration strategies, while graphs define the sequence, retry limits, and recovery paths.
