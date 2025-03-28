from qualibrate import QualibrationLibrary


def test_06_Ramsey(library: QualibrationLibrary):
    node_template = library.nodes["06_Ramsey"]

    executed_node, run_summary = node_template.run(load_data_id=1575)
