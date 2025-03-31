from qualibrate import QualibrationLibrary


def test_08c_readout_power_optimization(library: QualibrationLibrary):
    node_template = library.nodes["08c_readout_power_optimization"]

    executed_node, run_summary = node_template.run(load_data_id=1564)
