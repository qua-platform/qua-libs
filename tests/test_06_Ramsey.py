from qualibrate import QualibrationLibrary


def test_06_Ramsey(library: QualibrationLibrary):
    node = library.nodes["06_Ramsey"]

    node.run(load_data_id=1293)
