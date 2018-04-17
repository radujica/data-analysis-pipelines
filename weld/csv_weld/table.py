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
        self.columns = self._create_columns(header_df)

        self.nrows = None
        self.usecols = self.columns.keys()

    def _create_columns(self, header_df):
        columns = {}
        for column_name in header_df:
            data_id = LazyResult.generate_data_id(column_name)
            column = Column(column_name, self, data_id, header_df[column_name].dtype)
            columns[column_name] = column
            LazyResult.register_lazy_data(data_id, column)

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
