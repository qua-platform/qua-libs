from qualibrate import QualibrationLibrary


def test_02a_resonator_spectroscopy(library: QualibrationLibrary):
    node_template = library.nodes["02a_resonator_spectroscopy"]

    executed_node, run_summary = node_template.run(load_data_id=1581)
