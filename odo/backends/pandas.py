from __future__ import absolute_import, division, print_function

from datetime import datetime
from functools import partial

from datashape import discover
from datashape import string, object_, datetime_, Option
import datashape

import pandas as pd
import numpy as np

from ..convert import convert


possibly_missing = frozenset({string, datetime_})


def dshape_from_pandas(dtype):
    dshape = datashape.CType.from_numpy_dtype(dtype)
    dshape = string if dshape == object_ else dshape
    return Option(dshape) if dshape in possibly_missing else dshape


@discover.register(pd.DataFrame)
def discover_dataframe(df):
    return len(df) * datashape.Record(
        zip(df.columns, map(dshape_from_pandas, df.dtypes)),
    )


@discover.register(pd.Series)
def discover_series(s):
    return len(s) * dshape_from_pandas(s.dtype)


def coerce_datetimes(df):
    """ Make object columns into datetimes if possible

    Warning: this operates inplace.

    Example
    -------

    >>> df = pd.DataFrame({'dt': ['2014-01-01'], 'name': ['Alice']})
    >>> df.dtypes  # note that these are strings/object
    dt      object
    name    object
    dtype: object

    >>> df2 = coerce_datetimes(df)
    >>> df2
              dt   name
    0 2014-01-01  Alice

    >>> df2.dtypes  # note that only the datetime-looking-one was transformed
    dt      datetime64[ns]
    name            object
    dtype: object
    """
    objects = df.select_dtypes(include=['object'])
    # NOTE: In pandas < 0.17, pd.to_datetime(' ') == datetime(...), which is
    # not what we want.  So we have to remove columns with empty or
    # whitespace-only strings to prevent erroneous datetime coercion.
    columns = [
        c for c in objects.columns
        if not np.any(objects[c].str.isspace() | objects[c].str.isalpha())
    ]
    df2 = objects[columns].apply(partial(pd.to_datetime, errors='ignore'))

    for c in df2.columns:
        df[c] = df2[c]
    return df


@convert.register(pd.Timestamp, datetime)
def convert_datetime_to_timestamp(dt, **kwargs):
    return pd.Timestamp(dt)


@convert.register(pd.Timestamp, float)
def nan_to_nat(fl, **kwargs):
    try:
        if np.isnan(fl):
            # Only nan->nat edge
            return pd.NaT
    except TypeError:
        pass
    raise NotImplementedError()


@convert.register(pd.Timestamp, (pd.tslib.NaTType, type(None)))
def convert_null_or_nat_to_nat(n, **kwargs):
    return pd.NaT
