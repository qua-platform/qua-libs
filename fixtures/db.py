import pytest

from qualibs.results.impl.sqlalchemy import SqlAlchemyResultsConnector


@pytest.fixture()
def results_connector():
    return SqlAlchemyResultsConnector(backend='sqlite:///:memory:', echo=True)  # this is memory sqlite
