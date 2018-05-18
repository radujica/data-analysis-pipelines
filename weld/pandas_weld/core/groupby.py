from collections import OrderedDict

from grizzly.encoders import numpy_to_weld_type

from pandas_weld import DataFrame, Index, Series, MultiIndex
from pandas_weld.weld import weld_groupby_aggregate, weld_get_column


# TODO: grouping on all columns should result in empty df
class DataFrameGroupBy(object):
    """ Intermediary DataFrame-like with grouped data

    Only an aggregation can be evaluated.

    Parameters
    ----------
    expression : WeldObject
        created through weld_groupby
    by : list of str
        the column names on which the data was grouped
    by_types : list of np.dtype
        the types of the by columns
    columns : list of str
        the remaining column names which shall be aggregated
    column_types : list of np.dtype
        the types of the columns

    """
    def __init__(self, expression, by, by_types, columns, column_types):
        self.expr = expression

        self.by = by
        self.by_types = by_types
        self.columns = columns
        self.columns_types = column_types

    def _aggregate(self, operation):
        aggregated_data = weld_groupby_aggregate(self.expr,
                                                 [str(numpy_to_weld_type(k)) for k in self.by_types],
                                                 [str(numpy_to_weld_type(k)) for k in self.columns_types],
                                                 operation)

        if len(self.by) == 1:
            new_index = Index(weld_get_column(aggregated_data, 0, True),
                              self.by_types[0],
                              self.by[0])
        else:
            arrays = [weld_get_column(aggregated_data, index, True) for index in xrange(len(self.by))]
            new_index = MultiIndex.from_arrays(arrays, self.by)

        new_data = OrderedDict()
        for i in xrange(len(self.columns)):
            column_name = self.columns[i]
            new_data[column_name] = Series(weld_get_column(aggregated_data, i, False),
                                           self.columns_types[i],
                                           new_index,
                                           column_name)

        return DataFrame(new_data, new_index)

    def sum(self):
        return self._aggregate('+')

    def prod(self):
        return self._aggregate('*')

    def min(self):
        return self._aggregate('min')

    def max(self):
        return self._aggregate('max')

    def mean(self):
        raise NotImplementedError
