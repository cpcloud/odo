from __future__ import absolute_import, division, print_function

from getpass import getuser
import os
import itertools

import pytest
sa = pytest.importorskip('sqlalchemy')

from odo.backends.csv import CSV
from odo import odo, resource, drop, discover
from odo.utils import assert_allclose, tmpfile


names = ('tbl%d' % i for i in itertools.count())


@pytest.yield_fixture
def pathname():
    with tmpfile('.sqlite') as fn:
        yield fn


@pytest.fixture
def tablename():
    return next(names)


@pytest.fixture(params=[
    'mysql+pymysql://%s@localhost:3306/test::{tablename}' % getuser(),
    'postgresql+psycopg2://postgres@localhost/test::{tablename}',
    'sqlite:///{pathname}::{tablename}',
])
def url(request, pathname, tablename):
    return request.param.format(pathname=pathname, tablename=tablename)


@pytest.yield_fixture
def sql(url):
    try:
        t = resource(url, dshape='var * {a: int64, b: int64}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield t
        finally:
            drop(t)

data = [(1, 2), (10, 20), (100, 200)]


@pytest.yield_fixture(scope='module')
def csv():
    with tmpfile('.csv') as fn:
        with open(fn, 'w') as f:
            f.write('\n'.join(','.join(map(str, row)) for row in data))
        yield CSV(fn)


@pytest.fixture
def complex_csv():
    this_dir = os.path.dirname(__file__)
    return CSV(os.path.join(this_dir, 'dummydata.csv'), has_header=True)


@pytest.yield_fixture
def complex_sql(url):
    dshape = """var * {
        Name: string, RegistrationDate: date, ZipCode: int32, Consts: float64
    }"""
    try:
        t = resource(url, dshape=dshape)
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield t
        finally:
            drop(t)


def test_simple_into(csv, sql):
    odo(csv, sql, dshape=discover(sql))
    assert odo(sql, list) == data


def test_append(csv, sql):
    odo(csv, sql)
    assert odo(sql, list) == data

    odo(csv, sql)
    assert odo(sql, list) == data + data


def test_tryexcept_into(csv, sql):
    with pytest.raises(sa.exc.NotSupportedError):
        odo(csv, sql, quotechar="alpha")  # uses multi-byte character


def test_no_header_no_columns(csv, sql):
    odo(csv, sql, dshape=discover(sql))
    assert odo(sql, list) == data


def test_complex_odo(complex_csv, complex_sql):
    # data from: http://dummydata.me/generate
    odo(complex_csv, complex_sql, dshape=discover(complex_sql))
    assert_allclose(odo(complex_sql, list), odo(complex_csv, list))


def test_sql_to_csv(sql, csv):
    sql = odo(csv, sql)
    with tmpfile('.csv') as fn:
        csv = odo(sql, fn)
        assert odo(csv, list) == data
        assert discover(csv).measure.names == discover(sql).measure.names


def test_sql_select_to_csv(sql, csv):
    sql = odo(csv, sql)
    query = sa.select([sql.c.a])
    with tmpfile('.csv') as fn:
        csv = odo(query, fn)
        assert odo(csv, list) == [(x,) for x, _ in data]


def test_invalid_escapechar(sql, csv):
    with pytest.raises(ValueError):
        odo(csv, sql, escapechar='12')

    with pytest.raises(ValueError):
        odo(csv, sql, escapechar='')
