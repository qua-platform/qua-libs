import pytest

from qualibs.results.ORM import DBConnector


@pytest.fixture()
def db_connector():
    return DBConnector(DB_path=':memory:')
