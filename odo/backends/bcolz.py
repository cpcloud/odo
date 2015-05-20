from __future__ import absolute_import, division, print_function

import os
import shutil

from bcolz import ctable, carray

import numpy as np
import pandas as pd

from toolz import keyfilter

import datashape
from datashape.predicates import isrecord
from datashape import discover, from_numpy, var, Fixed

from ..numpy_dtype import dshape_to_numpy
from ..append import append
from ..convert import convert, ooc_types
from ..resource import resource
from ..drop import drop
from ..chunks import chunks

keywords = ['cparams', 'dflt', 'expectedlen', 'chunklen', 'rootdir']


@discover.register((ctable, carray))
def discover_bcolz(c, **kwargs):
    return from_numpy(c.shape, c.dtype)


@append.register((ctable, carray), np.ndarray)
@append.register(ctable, ctable)
@append.register(carray, carray)
def numpy_append_to_bcolz(a, b, **kwargs):
    a.append(b)
    a.flush()
    return a


@append.register((ctable, carray), object)
def numpy_append_to_bcolz(a, b, **kwargs):
    return append(a, convert(chunks(np.ndarray), b, **kwargs), **kwargs)


@append.register(ctable, pd.DataFrame)
def append_frame_to_bcolz(bc, df, **kwargs):
    expectedlen = kwargs.pop('expectedlen', len(df))
    return append(bc, ctable.fromdataframe(df, expectedlen=expectedlen),
                  expectedlen=expectedlen, **kwargs)


@convert.register(ctable, np.ndarray, cost=2.0)
def convert_numpy_to_bcolz_ctable(x, **kwargs):
    return ctable(x, **keyfilter(keywords.__contains__, kwargs))


@convert.register(carray, np.ndarray, cost=2.0)
def convert_numpy_to_bcolz_carray(x, **kwargs):
    return carray(x, **keyfilter(keywords.__contains__, kwargs))


@convert.register(np.ndarray, (carray, ctable), cost=1.0)
def convert_bcolz_to_numpy(x, **kwargs):
    return x[:]


@append.register((carray, ctable), chunks(np.ndarray))
def append_carray_with_chunks(a, c, **kwargs):
    for chunk in c:
        append(a, chunk)
    a.flush()
    return a


@convert.register(chunks(np.ndarray), (ctable, carray), cost=1.2)
def bcolz_to_numpy_chunks(x, chunksize=2**20, **kwargs):
    def load():
        first_n = min(1000, chunksize)
        first = x[:first_n]
        yield first
        for i in range(first_n, x.shape[0], chunksize):
            yield x[i: i + chunksize]
    return chunks(np.ndarray)(load)


@resource.register('.*\.bcolz/?')
def resource_bcolz(uri, dshape=None, expected_dshape=None, **kwargs):
    if os.path.exists(uri):
        try:
            return ctable(rootdir=uri)
        # __rootdirs__ doesn't exist because we aren't a ctable
        except IOError:
            return carray(rootdir=uri)
    else:
        if not dshape:
            raise ValueError("Must specify either existing bcolz directory or"
                             " valid datashape")
        dshape = datashape.dshape(dshape)

        if expected_dshape is not None:
            expected_dshape = datashape.dshape(expected_dshape)

        dt = dshape_to_numpy(dshape)
        shape_tail = tuple(map(int, dshape.shape[1:]))  # tail of shape
        if dshape.shape[0] == var or isrecord(dshape.measure):
            shape = (0,) + shape_tail
        else:
            shape = (int(dshape.shape[0]),) + shape_tail

        x = np.empty(shape=shape, dtype=dt)

        kwargs = keyfilter(keywords.__contains__, kwargs)
        expectedlen = kwargs.pop('expectedlen',
                                 int(expected_dshape[0])
                                 if expected_dshape is not None and
                                 isinstance(expected_dshape[0], Fixed)
                                 else None)

        constructor = ctable if isrecord(dshape.measure) else carray
        return constructor(x, rootdir=uri, expectedlen=expectedlen, **kwargs)


@drop.register((carray, ctable))
def drop_bcolz(b, **kwargs):
    b.flush()
    shutil.rmtree(b.rootdir)


ooc_types |= set((carray, ctable))
