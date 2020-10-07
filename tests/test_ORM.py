from qualibs.results.ORM import *


def test_save_record(db_connector):
    now = datetime.datetime.now
    res_a = Results(graph_id=1, node_id=1, result_id=1, user_id='Dan', start_time=now() + datetime.timedelta(days=-2),
                    end_time=now(),
                    res_name="this",
                    res_val='that')
    db_connector.save(res_a)
    rdr = DataReader(db_connector) # Gal - this should receive the same connector
    res = rdr.fetch(DataReaderQuery(graph_id=1))
    assert res is res_a
