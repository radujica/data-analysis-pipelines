from grizzly.encoders import numpy_to_weld_type
from weld.types import WeldBit
from lazy_data import LazyData
from indexes import Index
from indexes import MultiIndex
from pandas_weld.weld import weld_filter, weld_element_wise_op, weld_aggregate, weld_merge_single_index, \
    weld_merge_triple_index, weld_index_to_values
from series import Series
from utils import replace_slice_defaults, weld_to_numpy_type
import numpy as np


class DataFrame(object):
    """ Weld-ed pandas DataFrame

    Parameters
    ----------
    data : dict
        column names -> data array or LazyData
    index : Index, MultiIndex, or RangeIndex

    See also
    --------
    pandas.DataFrame

    """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __repr__(self):
        string_representation = """column names:\n\t%(columns)s\nindex names:\n\t%(indexes)s"""

        return string_representation % {'columns': self.data.keys(),
                                        'indexes': self.index.names}

    # TODO: prettify
    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        """ Evaluates by creating a str representation of the DataFrame

        Parameters
        ----------
        see LazyData

        Returns
        -------
        str

        """
        materialized_columns = {}
        # TODO: fix bug; what if not lazydata?
        for column in self.data.items():
            materialized_columns[column[0]] = column[1].evaluate(verbose=verbose)

        string_representation = """%(repr)s\nindex:\n\n%(index)s\ncolumns:\n\t%(columns)s"""

        return string_representation % {'repr': self.__repr__(),
                                        'index': self.index.evaluate(verbose, decode, passes,
                                                                     num_threads, apply_experimental_transforms),
                                        'columns': materialized_columns}

    def __iter__(self):
        for column_name in self.data:
            yield column_name

    def __getitem__(self, item):
        """ Retrieve a portion of the DataFrame

        Has consequences! When slicing, any previous and/or following operations on
        the data within will be done only on this subset of the data

        Parameters
        ----------
        item : str, slice, or list of str, Series of bool
            if str, returns a column as a Series;
            if slice, returns a sliced DataFrame;
            if list, returns a DataFrame with only the columns from the list;
            if Series, returns a filtered DataFrame only with the rows corresponding to
            True in the Series

        Returns
        -------
        Series or DataFrame

        """
        if isinstance(item, str):
            element = self.data[item]
            if isinstance(element, LazyData):
                return Series(element.expr,
                              weld_to_numpy_type(element.weld_type),
                              self.index,
                              item,
                              element.data_id)
            elif isinstance(element, np.ndarray):
                return Series(element,
                              element.dtype,
                              self.index,
                              item)
            else:
                raise TypeError('column is neither LazyData nor np.ndarray')
        elif isinstance(item, slice):
            item = replace_slice_defaults(item)

            new_data = {}
            for column_name in self:
                # making series because Series has the proper method to slice something; re-use the code above
                series = self[str(column_name)]
                # the actual slice handled by Series getitem
                new_data[column_name] = series[item]

            # index slice handled by index
            new_index = self.index[item]

            return DataFrame(new_data, new_index)
        elif isinstance(item, list):
            new_data = {}

            for column_name in item:
                if not isinstance(column_name, str):
                    raise TypeError('expected a list of column names as strings')

                new_data[column_name] = self.data[column_name]

            return DataFrame(new_data, self.index)
        elif isinstance(item, LazyData):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected series of bool to filter DataFrame rows')

            new_data = {}
            for column_name in self:
                data = self.data[column_name]

                if isinstance(data, LazyData):
                    weld_type = data.weld_type
                    data_id = data.data_id
                    dtype = weld_to_numpy_type(weld_type)
                    data = data.expr
                elif isinstance(data, np.ndarray):
                    weld_type = numpy_to_weld_type(data.dtype)
                    dtype = data.dtype
                    data_id = None
                else:
                    raise TypeError('expected data in column to be of type LazyData or np.ndarray')

                new_data[column_name] = Series(weld_filter(data,
                                                           item.expr,
                                                           weld_type),
                                               dtype,
                                               self.index,
                                               column_name,
                                               data_id)
            # slice the index
            new_index = self.index[item]

            return DataFrame(new_data, new_index)
        else:
            raise TypeError('expected a str, slice, list, or Series in DataFrame.__getitem__')

    def head(self, n=10):
        """ Eagerly evaluates the DataFrame

        This operation has no consequences, unlike getitem.

        Parameters
        ----------
        n : int
            how many rows to return

        Returns
        -------
        str
            the output of evaluate on the sliced DataFrame

        """
        slice_ = replace_slice_defaults(slice(n))

        new_data = {}
        for column_name in self:
            # making series because Series has the proper method to slice something; re-use the code above
            series = self[str(column_name)]

            # by not passing a data_id, the data is not linked to the input_mapping read
            new_data[column_name] = Series(series.head(n),
                                           series.dtype,
                                           series.index,
                                           series.name,
                                           series.data_id)

        new_index = self.index[slice_]

        return DataFrame(new_data, new_index).evaluate(verbose=False)

    def __setitem__(self, key, value):
        """ Add/update DataFrame column

        Parameters
        ----------
        key : str
            column name
        value : np.ndarray / LazyData
            note it does NOT check for the same length as the other columns

        """
        if not isinstance(key, str):
            raise TypeError('expected key as a str')
        elif not isinstance(value, LazyData) and not isinstance(value, np.ndarray):
            raise TypeError('expected value as LazyData or np.ndarray')

        self.data[key] = value

    def rename(self, columns):
        """ Rename columns

        Currently a simplified version of pandas' rename

        Parameters
        ----------
        columns : dict
            of old name -> new name

        Returns
        -------
        DataFrame
            with the given columns renamed

        """
        new_data = {}
        for column_name in self:
            if column_name in columns.keys():
                new_data[columns[column_name]] = self.data[column_name]
            else:
                new_data[column_name] = self.data[column_name]

        return DataFrame(new_data, self.index)

    def drop(self, columns):
        """ Drop 1 or more columns

        Unlike pandas drop, this is currently restricted to dropping columns

        Parameters
        ----------
        columns : str or list of str
            column name or list of column names to drop

        Returns
        -------
        DataFrame
            returns a new DataFrame without these columns

        """
        if isinstance(columns, str):
            new_data = {}
            for column_name in self:
                if column_name != columns:
                    new_data[column_name] = self.data[column_name]

            return DataFrame(new_data, self.index)
        elif isinstance(columns, list):
            new_data = {}
            for column_name in self:
                if column_name not in columns:
                    new_data[column_name] = self.data[column_name]

            return DataFrame(new_data, self.index)
        else:
            raise TypeError('expected columns as a str or a list of str')

    # TODO: add type conversion(?); pandas works when e.g. column_of_ints - 2.0 => float result
    def _element_wise_operation(self, other, operation):
        if not isinstance(other, (str, unicode, int, long, float, bool)):
            raise TypeError('can only compare with scalars')

        assert isinstance(operation, (str, unicode))

        new_data = {}
        for column_name in self:
            # get as series
            series = self[str(column_name)]
            # apply the operation
            new_data[column_name] = Series(weld_element_wise_op(series.expr,
                                                                other,
                                                                operation,
                                                                series.weld_type),
                                           series.dtype,
                                           self.index,
                                           series.name)

        return DataFrame(new_data, self.index)

    def __add__(self, other):
        return self._element_wise_operation(other, '+')

    def __sub__(self, other):
        return self._element_wise_operation(other, '-')

    def __mul__(self, other):
        return self._element_wise_operation(other, '*')

    def __div__(self, other):
        return self._element_wise_operation(other, '/')

    # TODO: currently converts everything to float64; should act according to the input types
    # TODO: if there are strings it will fail, while in pandas for sum they are concatenated and prod are ignored
    def _aggregate(self, operation):
        assert isinstance(operation, (str, unicode))

        index = []
        data = []
        for column_name in self:
            index.append(column_name)
            # get as series
            series = self[str(column_name)]
            # apply the operation
            data.append(LazyData(weld_aggregate(series.expr,
                                                operation,
                                                series.weld_type),
                                 series.weld_type,
                                 0).evaluate(verbose=False))

        return Series(np.array(data).astype(np.float64), np.dtype(np.float64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    def sum(self):
        """ Eager operation to sum all elements in their respective columns

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('+')

    def prod(self):
        """ Eager operation to multiply all elements in their respective columns

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('*')

    def min(self):
        """ Eager operation to find the min value in each column

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('min')

    def max(self):
        """ Eager operation to find the max value in each column

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('max')

    def count(self):
        """ Eager operation to count the number of values in each column

        Returns
        -------
        Series

        """
        index = []
        data = []
        for column_name in self:
            index.append(column_name)
            # get as series
            series = self[str(column_name)]
            # apply the operation
            data.append(series.count().evaluate(verbose=False))

        return Series(np.array(data), np.dtype(np.int64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    # see TODO at Series.mean()
    def mean(self):
        """ Eager operation to compute the mean of the values in each column

        Returns
        -------
        Series

        """
        index = []
        data = []
        for column_name in self:
            index.append(column_name)
            # get as series
            series = self[str(column_name)]
            # apply the operation
            data.append(series.mean().evaluate(verbose=False))

        return Series(np.array(data), np.dtype(np.float64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    # see TODO at Series.std()
    def std(self):
        """ Eager operation to compute the standard deviation of the values in each column

        Returns
        -------
        Series

        """
        index = []
        data = []
        for column_name in self:
            index.append(column_name)
            # get as series
            series = self[str(column_name)]
            # apply the operation
            data.append(series.std().evaluate(verbose=False))

        return Series(np.array(data), np.dtype(np.float64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    def _merge_single(self, index1, index2):
        data = []
        data_ids = []
        # TODO: fix this duplicate code
        if isinstance(index1, LazyData):
            data_ids.append(index1.data_id)
            data.append(index1.expr)
        elif isinstance(index1, np.ndarray):
            data_ids.append(None)
            data.append(index1)
        else:
            raise TypeError('expected data in index to be of type LazyData or np.ndarray')

        if isinstance(index2, LazyData):
            data_ids.append(index2.data_id)
            data.append(index2.expr)
        elif isinstance(index2, np.ndarray):
            data_ids.append(None)
            data.append(index2)
        else:
            raise TypeError('expected data in index to be of type LazyData or np.ndarray')

        data = weld_merge_single_index(data)

        return [LazyData(data[i], WeldBit(), 1, data_id=data_ids[i]) for i in xrange(2)]

    def _index_to_values(self, levels, labels):
        if isinstance(levels, LazyData):
            weld_type = levels.weld_type
            levels = levels.expr
        elif isinstance(levels, np.ndarray):
            weld_type = numpy_to_weld_type(levels.dtype)
        else:
            raise TypeError('expected levels to be of type LazyData or np.ndarray')

        if isinstance(labels, LazyData):
            labels = labels.expr

        return LazyData(weld_index_to_values(levels, labels), weld_type, 1)

    def _merge_multi(self, index1, index2):
        assert len(index1.levels) == len(index2.levels) == 3

        index1 = [self._index_to_values(index1.levels[i], index1.labels[i]) for i in xrange(3)]
        index2 = [self._index_to_values(index2.levels[i], index2.labels[i]) for i in xrange(3)]

        data = []
        data_ids = []
        # TODO: fix this duplicate code
        for i in xrange(3):
            if isinstance(index1[i], LazyData):
                data_ids.append(index1[i].data_id)
                data.append(index1[i].expr)
            elif isinstance(index1[i], np.ndarray):
                data_ids.append(None)
                data.append(index1[i])
            else:
                raise TypeError('expected data in index to be of type LazyData or np.ndarray')

        for i in xrange(3):
            if isinstance(index2[i], LazyData):
                data_ids.append(index2[i].data_id)
                data.append(index2[i].expr)
            elif isinstance(index2[i], np.ndarray):
                data_ids.append(None)
                data.append(index2[i])
            else:
                raise TypeError('expected data in index to be of type LazyData or np.ndarray')

        data = weld_merge_triple_index([data[:3], data[3:6]])

        return [LazyData(data[i], WeldBit(), 1, data_id=data_ids[i]) for i in xrange(2)]

    # TODO: check for same column_names in both DataFrames!
    def merge(self, right):
        """ Join this DataFrame with another

        Currently only inner join on 1-d or 3-d index is supported

        Parameters
        ----------
        right : DataFrame
            to join with

        Returns
        -------
        DataFrame

        """
        if isinstance(self.index, MultiIndex):
            if not isinstance(right.index, MultiIndex):
                raise TypeError('both indexes must be MultiIndex')
            if len(self.index.levels) != 3 or len(right.index.levels) != 3:
                raise ValueError('MultiIndexes must be of the same length + only length 3 is currently supported')
            bool_indexes = self._merge_multi(self.index, right.index)
        else:
            bool_indexes = self._merge_single(self.index, right.index)
        # can filter any of the two dataframes for the new index
        new_index = self.index[bool_indexes[0]]

        new_data = {}
        for column_name in self:
            new_data[column_name] = self[str(column_name)][bool_indexes[0]]

        for column_name in right:
            new_data[column_name] = right[str(column_name)][bool_indexes[1]]

        return DataFrame(new_data, new_index)
