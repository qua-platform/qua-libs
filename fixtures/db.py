import pytest

from qualibs.results.impl.sqlalchemy import SqlAlchemyResultsConnector

target = ':memory:'


@pytest.fixture()
def results_connector():
    return SqlAlchemyResultsConnector(backend=f'sqlite:///{target}', echo=False)  # this is memory sqlite
