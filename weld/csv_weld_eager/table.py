import pandas as pd

from column import Column
from lazy_file import LazyFile
from lazy_result import LazyResult


class Table(LazyFile):
    _FILE_FORMAT = 'csv'

    def __init__(self, path):
        self.file_id = LazyResult.generate_file_id(Table._FILE_FORMAT)
        LazyResult.register_lazy_file(self.file_id, self)

        self.path = path
        header_df = self.read_metadata()
        # the params used to lazy_slice_rows and lazy_skip_columns
        self.slice_start = None
        self.nrows = None
        self.usecols = list(header_df)

        self.columns = self._create_columns(header_df)

    def _create_columns(self, header_df):
        from weld.weldobject import WeldObject

        columns = {}
        for column_name in header_df:
            data_id = LazyResult.generate_data_id(column_name)
            column = Column(column_name, self, data_id, header_df[column_name].dtype)

            weld_input_name = WeldObject.generate_input_name(data_id)
            LazyResult.register_lazy_data(weld_input_name, column)

            # force read it eagerly
            LazyResult.input_mapping[str(weld_input_name)] = column.eager_read()

            columns[column_name] = column

        return columns

    def read_metadata(self):
        # pandas already does type inference which is neat
        return pd.read_csv(self.path, sep=',', engine='c', nrows=1)

    # nrows as param used by Column.eager_head to bypass cache and not read everything
    def read_file(self, nrows=None):
        return pd.read_csv(self.path,
                           sep=',',
                           engine='c',
                           nrows=self.nrows if not nrows else nrows,
                           usecols=self.usecols)
