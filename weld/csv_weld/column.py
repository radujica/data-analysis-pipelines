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

    def eager_read(self, slice_=None):
        df = LazyResult.retrieve_file(self.table.file_id)

        if slice_ is None:
            slice_ = slice(0, self.table.nrows, 1)

        return df[self.name][slice_].values

    def eager_head(self, n=10):
        return self.eager_read(slice(0, n, 1))

    def lazy_skip_columns(self, columns):
        for column in columns:
            self.table.usecols.remove(column)

    def lazy_slice_rows(self, slice_):
        self.table.nrows = slice_.stop