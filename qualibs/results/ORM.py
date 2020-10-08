import datetime
import os
import time
from dataclasses import dataclass
from random import randint
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Sequence, DATETIME
from sqlalchemy.orm import sessionmaker
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.util.compat import contextmanager

Base = declarative_base()


class Results(Base):
    __tablename__ = 'Results'
    graph_id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey('Nodes.node_id', ondelete="CASCADE"), primary_key=True)
    result_id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    start_time = Column(DATETIME, nullable=False)
    end_time = Column(DATETIME, nullable=False)
    res_name = Column(String, nullable=False)
    res_val = Column(String, nullable=False)

    def __repr__(self):
        return "<Result(graph_id='%s', node_id='%s', result_id='%s')>" % (
            self.graph_id, self.node_id, self.result_id)


class Metadata(Base):
    __tablename__ = 'Metadata'
    graph_id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey('Nodes.node_id', ondelete="CASCADE"), primary_key=True)
    data_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    val = Column(String, nullable=False)

    def __repr__(self):
        return "<Metadata(graph_id='%s', node_id='%s', data_id='%s')>" % (
            self.graph_id, self.node_id, self.data_id)


class Nodes(Base):
    __tablename__ = 'Nodes'
    graph_id = Column(Integer, ForeignKey('Graphs.graph_id', ondelete="CASCADE"), primary_key=True)
    node_id = Column(Integer, primary_key=True)
    results = relationship("Results", cascade="all, delete-orphan")
    metadat = relationship("Metadata", cascade="all, delete-orphan")

    def __repr__(self):
        return "<Node(graph_id='%s', node_id='%s')>" % (
            self.graph_id, self.node_id)


class Graphs(Base):
    __tablename__ = 'Graphs'
    graph_id = Column(Integer, primary_key=True)
    graph_script = Column(String)
    # results = relationship("Results", cascade="all, delete-orphan")
    # metadata = relationship("Metadata", cascade="all, delete-orphan")
    nodes = relationship("Nodes", cascade="all, delete-orphan")

    def __repr__(self):
        return "<Graph(graph_id='%s')>" % (
            self.graph_id)


@dataclass
class DataReaderQuery:
    # object to collect arguments to the query
    graph_id: Optional[int] = None
    node_id: Optional[int] = None
    result_id: Optional[int] = None
    id_range: Optional[Tuple[int, int]] = None
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    user_id: Optional[str] = None


# dr.graph(graph_id).node(node_id).results()
# fluent interface --> method chaining

class DataReader:
    """ A class to run queries against the persistence backend """

    def __init__(self, DB_path, engine=''):
        self.conn = DBConnector(DB_path, engine)

    def fetch(self, query_obj: DataReaderQuery):
        """
                Fetch records from the persistence backend using a DataReaderQuery object
                :param query_obj: object containing query parameters
                :type: query_obj: DataReaderQuery
                :return:
         """
        if not isinstance(query_obj, DataReaderQuery):
            raise TypeError('query_obj must be a DataReaderQuery object')
        return self.conn.fetch(query_obj)
        # if query_obj.id:
        #     self.conn.fetch(id=query_obj.id)
        # elif query_obj.start_time:
        #     self.conn.fetch(start_time=query_obj.start_time)


class ResultItem:
    def __init__(self, result):
        self.graph_id = result.graph_id
        self.node_id = result.node_id
        self.result_id = result.result_id
        self.start_time = result.start_time
        self.end_time = result.end_time
        self.user_id = result.user_id
        self.res_name = result.res_name
        self.res_val = result.res_val

    def __eq__(self):
        pass

    def __repr__(self):
        return "<ResultItem(graph_id='%s', node_id='%s', result_id='%s',res_name='%s',res_val='%s')>" % (
            self.graph_id, self.node_id, self.result_id, self.res_name, self.res_val)


class NodeDataWriter:
    def __init__(self, node_id, graph_id, dbsaver: 'DBConnector'):
        self._node_id = node_id
        self._graph_id = graph_id
        self._dbsaver = dbsaver

    def save(self):
        self._dbsaver.save(DataReaderQuery(graph_id=self._graph_id, node_id=self._node_id))


class DBConnector:
    def __init__(self, DB_path='my_db.db', engine='', backend='sqlite'):
        if engine:
            self._engine = engine
        else:
            self._engine = create_engine(f'sqlite:///{DB_path}', echo=True)

        Base.metadata.create_all(self._engine)

        self._Session = sessionmaker(bind=self._engine)
        # session = Session()

    @contextmanager
    def _session_maker(self):
        """Provide a transactional scope around a series of operations."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def save(self, items):
        with self._session_maker() as sess:
            if type(items) is not list:
                sess.add(items)
            else:
                sess.add_all(items)

    def fetch(self, query_obj: DataReaderQuery):
        with self._session_maker() as sess:
            query = sess.query(Results).order_by(Results.graph_id)
            if query_obj.graph_id:
                query = query.filter(Results.graph_id == query_obj.graph_id)

            if query_obj.start_time:
                query = query.filter(Results.start_time >= query_obj.start_time)

            if query_obj.end_time:
                query = query.filter(Results.start_time <= query_obj.end_time)

            if query_obj.user_id:
                query = query.filter(Results.user_id == query_obj.user_id)

            return [ResultItem(row) for row in query.all()]


if __name__ == '__main__':
    # engine = create_engine('sqlite:///:memory:', echo=True)
    try:
        os.remove('my_db.db')
    except FileNotFoundError:
        pass

    now = datetime.datetime.now

    # saver = DBSaver(DB_path=':memory:')
    dbcon = DBConnector()

    res_a = Results(graph_id=1, node_id=1, result_id=1, user_id='Dan', start_time=now() + datetime.timedelta(days=-2),
                    end_time=now(),
                    res_name="this",
                    res_val='that')
    res_b = Results(graph_id=1, node_id=2, result_id=2, user_id='Gal', start_time=now() + datetime.timedelta(days=-2),
                    end_time=now(),
                    res_name="this",
                    res_val='that')
    graph = Graphs(graph_id=randint(1000, 2000), graph_script='a')

    dbcon.save(res_a)

    dbcon.save(
        [Results(graph_id=randint(1000, 2000), node_id=1, result_id=1, user_id='Gal',
                 start_time=now() + datetime.timedelta(days=-2), end_time=now(),
                 res_name="this",
                 res_val=str(randint(-1e6, 1e6))) for x in range(10)])

    dbcon.save([res_b, graph])

    # time.sleep(5)
    rdr = DataReader('my_db.db')

    # a = rdr.fetch(DataReaderQuery(start_time=datetime.datetime.today() + datetime.timedelta(days=-3),
    #                               end_time=datetime.datetime.today() + datetime.timedelta(days=-1),
    #                               user_id='Gal'))

    rdr.fetch(DataReaderQuery(graph_id=1))
