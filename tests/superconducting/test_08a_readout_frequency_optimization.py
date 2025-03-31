from qualibrate import QualibrationLibrary


def test_08a_readout_frequency_optimization(library: QualibrationLibrary):
    node_template = library.nodes["08a_readout_frequency_optimization"]

    executed_node, run_summary = node_template.run(load_data_id=1563)
