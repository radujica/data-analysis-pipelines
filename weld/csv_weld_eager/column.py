import numpy as np
from grizzly.encoders import NumPyEncoder, NumPyDecoder

from lazy_data import LazyData
from lazy_result import LazyResult


class Column(LazyData):
    encoder = NumPyEncoder()
    decoder = NumPyDecoder()

    def __init__(self, name, table, data_id, dtype):
        self.name = name
        self.table = table
        self.data_id = data_id
        self.dtype = dtype

    def eager_read(self):
        # make use of cache by retrieving
        df = LazyResult.retrieve_file(self.table.file_id)

        slice_ = slice(self.table.slice_start, self.table.nrows, 1)

        data = df[self.name][slice_].values

        # treat any object dtype as str
        if self.dtype.char == 'O':
            data = data.astype(np.str)

        return data

    def eager_head(self, n=10):
        # skip the cache and re-use read_file method with param from Table
        # which will now only read first n rows
        df = self.table.read_file(n)

        data = df[self.name][:n].values

        # treat any object dtype as str
        if self.dtype.char == 'O':
            data = data.astype(np.str)

        return data

    def lazy_skip_columns(self, columns):
        # pandas allows skipping some columns efficiently through the usecols parameter
        for column in columns:
            self.table.usecols.remove(column)

    def lazy_slice_rows(self, slice_):
        # the parser needs to read until stop anyway, and filter later through eager_read
        self.table.slice_start = slice_.start
        self.table.nrows = slice_.stop
