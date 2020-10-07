from qualibs.results.ORM import *

def test_save_record():
    dbcon = DBConnector(DB_path=':memory:')
    now = datetime.datetime.now
    res_a = Results(graph_id=1, node_id=1, result_id=1, user_id='Dan', start_time=now() + datetime.timedelta(days=-2),
                    end_time=now(),
                    res_name="this",
                    res_val='that')
    dbcon.save(res_a)
    rdr = DataReader(':memory:')
    res=rdr.fetch(DataReaderQuery(graph_id=1))
    assert res is res_a



