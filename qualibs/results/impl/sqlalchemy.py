from typing import Optional, List, Union, Iterable

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Sequence, DATETIME
from sqlalchemy.orm import sessionmaker
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.util.compat import contextmanager

from qualibs.results.api import BaseResultsConnector, DataReaderQuery, Result

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

    def to_model(self):
        return Result(
            graph_id=self.graph_id,
            node_id=self.node_id,
            result_id=self.result_id,
            user_id=self.user_id,
            start_time=self.start_time,
            end_time=self.end_time,
            res_name=self.res_name,
            res_val=self.res_val,
        )


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


class SqlAlchemyResultsConnector(BaseResultsConnector):
    def __init__(self, backend='sqlite:///:memory:', echo=False):  # GAL - by default use memory database
        super(SqlAlchemyResultsConnector, self).__init__()
        self._engine = create_engine(backend, echo=echo)
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

    def save(self, items: Union[Result, Iterable[Result]]):
        with self._session_maker() as sess:
            iter_items = [items] if type(items) is Result else list(items)
            rows = [Results(**it.__dict__) for it in iter_items]
            if len(rows) == 1:
                sess.add(rows[0])
            else:
                sess.add_all(rows)

    def _query(self, query_obj: DataReaderQuery):
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

            return query

    def fetch(self, query_obj: DataReaderQuery) -> List[Result]:
        return list(row.to_model() for row in self._query(query_obj))

    def fetch_first(self, query_obj: DataReaderQuery) -> Optional[Result]:
        return self._query(query_obj).first().to_model()

# if __name__ == '__main__':
#     # engine = create_engine('sqlite:///:memory:', echo=True)
#     try:
#         os.remove('my_db.db')
#     except FileNotFoundError:
#         pass
#
#     now = datetime.datetime.now
#
#     # saver = DBSaver(DB_path=':memory:')
#     dbcon = DBConnector()
#
#     res_a = Results(graph_id=1, node_id=1, result_id=1, user_id='Dan', start_time=now() + datetime.timedelta(days=-2),
#                     end_time=now(),
#                     res_name="this",
#                     res_val='that')
#     res_b = Results(graph_id=1, node_id=2, result_id=2, user_id='Gal', start_time=now() + datetime.timedelta(days=-2),
#                     end_time=now(),
#                     res_name="this",
#                     res_val='that')
#     graph = Graphs(graph_id=randint(1000, 2000), graph_script='a')
#
#     dbcon.save(res_a)
#
#     dbcon.save(
#         [Results(graph_id=randint(1000, 2000), node_id=1, result_id=1, user_id='Gal',
#                  start_time=now() + datetime.timedelta(days=-2), end_time=now(),
#                  res_name="this",
#                  res_val=str(randint(-1e6, 1e6))) for x in range(10)])
#
#     dbcon.save([res_b, graph])
#
#     # time.sleep(5)
#     rdr = DataReader('my_db.db')
#
#     # a = rdr.fetch(DataReaderQuery(start_time=datetime.datetime.today() + datetime.timedelta(days=-3),
#     #                               end_time=datetime.datetime.today() + datetime.timedelta(days=-1),
#     #                               user_id='Gal'))
#
#     rdr.fetch(DataReaderQuery(graph_id=1))
