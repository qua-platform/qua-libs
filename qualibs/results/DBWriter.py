from abc import ABC, abstractmethod
import sqlite3 as sl
from typing import Tuple


class GraphStore(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def write_res(self):
        raise NotImplementedError()

    @abstractmethod
    def read_res(self):
        raise NotImplementedError()

    def write_nodes(self):
        raise NotImplementedError()

    @abstractmethod
    def read_nodes(self):
        raise NotImplementedError()

    @abstractmethod
    def write_metadata(self):
        raise NotImplementedError()

    @abstractmethod
    def read_metadata(self):
        raise NotImplementedError()


class GraphStoreSqlite(GraphStore):

    def __init__(self, db_path):
        super.__init__()
        self.con = sl.connect(db_path)

    def init_db(self):
        with self.con:
            self.con.execute(""" 
               CREATE TABLE  IF NOT EXISTS Results 
                ( 
                    graph_id   INT NOT NULL, 
                    node_id    INTEGER NOT NULL, 
                    result_id  INTEGER NOT NULL, 
                    user_id    TEXT, 
                    start_time DATETIME, 
                    end_time   DATETIME, 
                    res_name   TEXT, 
                    res_val    TEXT, 
                    PRIMARY KEY (graph_id, node_id, result_id), 
                    CONSTRAINT node_results_FK FOREIGN KEY (node_id) REFERENCES Nodes (node_id) ON DELETE CASCADE ON UPDATE CASCADE 
                ); 

                        """)
            self.con.execute(""" 
                        CREATE TABLE  IF NOT EXISTS Nodes 
                        ( 
                            graph_id INTEGER NOT NULL, 
                            node_id  INTEGER NOT NULL, 
                            PRIMARY KEY (graph_id, node_id), 
                            CONSTRAINT graph_nodes_FK FOREIGN KEY (graph_id) REFERENCES Graphs (graph_id) ON DELETE CASCADE ON UPDATE CASCADE 
                        ); 
                        """)
            self.con.execute(""" 
                        CREATE TABLE  IF NOT EXISTS Graphs 
                        ( 
                            graph_id     INTEGER NOT NULL PRIMARY KEY, 
                            graph_script TEXT    NOT NULL 
                        ); 

                        """)

            self.con.execute(""" 
                        CREATE TABLE  IF NOT EXISTS Metadata 
                        ( 
                            graph_id INTEGER NOT NULL, 
                            node_id  INTEGER NOT NULL, 
                            data_id  INTEGER NOT NULL, 
                            name     TEXT    NOT NULL, 
                            val      TEXT    NOT NULL, 
                            PRIMARY KEY (graph_id, node_id, data_id), 
                            CONSTRAINT node_metadata_FK FOREIGN KEY (node_id) REFERENCES Nodes(node_id) ON DELETE CASCADE ON UPDATE CASCADE 
                        ); 
                        """)

    def write_res(self, result: Tuple[int, int, int, str, int, int, str, str]):
        with self.con as con:
            con.execute(""" 
                           INSERT INTO Results (graph_id,node_id,result_id,user_id,start_time,end_time,res_name,res_val) 
                           VALUES (?,?,?,?,?,?,?,?); 
                           """, result)

    def read_res(self):
        pass

    def write_nodes(self):
        pass

    def read_nodes(self):
        pass

    def write_metadata(self):
        pass

    def read_metadata(self):
        pass
