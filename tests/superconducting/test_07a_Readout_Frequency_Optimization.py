from qualibrate import QualibrationLibrary


def test_07a_Readout_Frequency_Optimization(library: QualibrationLibrary):
    node_template = library.nodes["07a_Readout_Frequency_Optimization"]

    executed_node, run_summary = node_template.run(load_data_id=1563)
