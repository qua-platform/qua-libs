from qualibs.graph import *


def test_cal_save_to_db():
    a = CalibrationNode('cal_n', lambda x, y=None, z=None: ({'x': 1}, 'out_spec'))
    a.optimal_params = {'x': 1}
    a.check_data_params = {'y': 2, 'z': 3}
    a.calibrate_params = {'y': 4}
    gdb = GraphDB('cal_db.db')
    b = CalibrationGraph('cal_g', graph_db=gdb)
    b.add_nodes(a)
    b.run(verbose=True)
