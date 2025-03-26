from qualibrate import QualibrationLibrary


def test_02a_resonator_spectroscopy(qualibrate_test_config):

    library = QualibrationLibrary.get_active_library()

    node = library.nodes["02a_resonator_spectroscopy"]

    node.run(load_data_id=12)
