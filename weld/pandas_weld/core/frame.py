from collections import OrderedDict

import numpy as np
import os
from grizzly.encoders import numpy_to_weld_type
from tabulate import tabulate
from weld.types import WeldBit, WeldLong

from indexes import Index, MultiIndex, RangeIndex
from lazy_result import LazyResult
from pandas_weld.weld import weld_filter, weld_element_wise_op, weld_aggregate, weld_merge_single_index, \
    weld_merge_triple_index, weld_index_to_values, weld_groupby, weld_count
from series import Series
from utils import replace_slice_defaults, get_expression_or_raw, evaluate_or_raw, get_weld_type, \
    get_dtype, get_weld_info


class DataFrame(object):
    """ Weld-ed pandas DataFrame

    Parameters
    ----------
    data : dict
        column names -> data array or LazyResult
    index : Index or MultiIndex or RangeIndex

    See also
    --------
    pandas.DataFrame

    """
    _cache_flag = False if os.getenv("LAZY_WELD_CACHE") == 'False' else True

    @staticmethod
    def _gather_dtypes(data):
        dtypes = {}
        for k, v in data.iteritems():
            dtypes[k] = get_dtype(v)

        return dtypes

    def __init__(self, data, index):
        self.data = data
        self.index = index
        self._dtypes = self._gather_dtypes(data)

    @property
    def dtypes(self):
        return Series(np.array(self._dtypes.values(), dtype=np.object),
                      np.dtype(np.object),
                      Index(np.array(self._dtypes.keys()), dtype=np.dtype(np.str)))

    def __repr__(self):
        return "{}(index={}, columns={})".format(self.__class__.__name__,
                                                 repr(self.index),
                                                 repr(self.data.keys()))

    # TODO: perhaps slice the arrays to avoid all the data being printed
    def __str__(self):
        str_data = OrderedDict()

        # NOTE: an evaluate step for MultiIndex to convert from labels & levels to actual values!
        if isinstance(self.index, MultiIndex):
            expanded = self.index.expand()
            names = self.index.names
            for i in range(len(expanded)):
                str_data[names[i]] = expanded[i].evaluate()
        else:
            # there is no guarantee at this point that it has been evaluated
            # TODO: perhaps make more user friendly check to avoid tabulate throwing an exception if not evaluated
            str_data[self.index.name] = self.index.data
        for column in self.data:
            str_data[column] = evaluate_or_raw(self.data[column])

        return tabulate(str_data, headers='keys')

    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        """ Evaluates by creating a str representation of the DataFrame

        Parameters
        ----------
        see LazyResult

        Returns
        -------
        str

        """
        evaluated_index = self.index.evaluate(verbose, decode, passes,
                                              num_threads, apply_experimental_transforms)
        evaluated_data = OrderedDict()
        for k, v in self.data.iteritems():
            evaluated_data[k] = evaluate_or_raw(v, verbose, decode, passes,
                                                num_threads, apply_experimental_transforms)

        return DataFrame(evaluated_data, evaluated_index)

    def __iter__(self):
        for column_name in self.data:
            yield str(column_name)

    def __getitem__(self, item):
        """ Retrieve a portion of the DataFrame

        Has consequences! When slicing, any previous and/or following operations on
        the data within will be done only on this subset of the data

        Parameters
        ----------
        item : str or slice or list of str or LazyResult
            if str, returns a column as a Series;
            if slice, returns a sliced DataFrame;
            if list, returns a DataFrame with only the columns from the list;
            if LazyResult, returns a filtered DataFrame only with the rows corresponding to
            True in the LazyResult

        Returns
        -------
        Series or DataFrame

        """
        if isinstance(item, str):
            element = self.data[item]

            data, dtype = get_weld_info(element, expression=True, dtype=True)

            return Series(data,
                          dtype,
                          self.index,
                          item)
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
        elif isinstance(item, LazyResult):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected series of bool to filter DataFrame rows')

            new_data = {}
            for column_name in self:
                data = self.data[column_name]

                data, weld_type, dtype = get_weld_info(data, expression=True, weld_type=True, dtype=True)

                new_data[column_name] = Series(weld_filter(data,
                                                           item.expr),
                                               dtype,
                                               self.index,
                                               column_name)
            # slice the index
            new_index = self.index[item]

            return DataFrame(new_data, new_index)
        else:
            raise TypeError('expected a str, slice, list, or Series in DataFrame.__getitem__')

    def head(self, n=10, verbose=True, decode=True, passes=None,
             num_threads=1, apply_experimental_transforms=False):
        """ Eagerly evaluates the DataFrame

        This operation has no consequences, unlike getitem.

        Parameters
        ----------
        n : int, optional
            how many rows to return
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

        Returns
        -------
        DataFrame
            the output of evaluate on the sliced DataFrame

        """
        new_data = {}
        for column_name in self:
            # making series because Series has the proper method to slice something; re-use the code above
            series = self[str(column_name)]

            new_data[column_name] = Series(series.head(n),
                                           series.dtype,
                                           series.index,
                                           series.name)

        slice_ = replace_slice_defaults(slice(n))
        new_index = self.index[slice_]

        return DataFrame(new_data, new_index).evaluate(verbose, decode, passes,
                                                       num_threads, apply_experimental_transforms)

    def __setitem__(self, key, value):
        """ Add/update DataFrame column

        Parameters
        ----------
        key : str
            column name
        value : np.ndarray or LazyResult
            note it does NOT check for the same length as the other columns

        """
        if not isinstance(key, str):
            raise TypeError('expected key as a str')
        elif not isinstance(value, LazyResult) and not isinstance(value, np.ndarray):
            raise TypeError('expected value as LazyResult or np.ndarray')

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
        # can't know at this point if the call is needed: if from csv, it is; from netcdf4, it is not.
        # a dataframe might contain data from both sources anyway
        [k.update_columns(columns) for k in self if isinstance(k, LazyResult)]

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
    def _aggregate(self, operation, verbose=True, decode=True, passes=None,
                   num_threads=1, apply_experimental_transforms=False):
        assert isinstance(operation, (str, unicode))

        index = []
        data = []
        for column_name in self:
            index.append(column_name)
            # get as series
            series = self[str(column_name)]
            # apply the operation
            data.append(LazyResult(weld_aggregate(series.expr,
                                                  operation,
                                                  series.weld_type),
                                   series.weld_type,
                                   0).evaluate(verbose, decode, passes,
                                               num_threads, apply_experimental_transforms))

        return Series(np.array(data).astype(np.float64),
                      np.dtype(np.float64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    def sum(self, verbose=True, decode=True, passes=None,
            num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to sum all elements in their respective columns

        Parameters
        ----------
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('+', verbose, decode, passes,
                               num_threads, apply_experimental_transforms)

    def prod(self, verbose=True, decode=True, passes=None,
             num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to multiply all elements in their respective columns

        Parameters
        ----------
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('*', verbose, decode, passes,
                               num_threads, apply_experimental_transforms)

    def min(self, verbose=True, decode=True, passes=None,
            num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to find the min value in each column

        Parameters
        ----------
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('min', verbose, decode, passes,
                               num_threads, apply_experimental_transforms)

    def max(self, verbose=True, decode=True, passes=None,
            num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to find the max value in each column

        Parameters
        ----------
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

        Returns
        -------
        Series
            results are currently converted to float64

        """
        return self._aggregate('max', verbose, decode, passes,
                               num_threads, apply_experimental_transforms)

    def count(self, verbose=True, decode=True, passes=None,
              num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to count the number of values in each column

        Parameters
        ----------
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

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
            data.append(series.count().evaluate(verbose, decode, passes,
                                                num_threads, apply_experimental_transforms))

        return Series(np.array(data),
                      np.dtype(np.int64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    # see TODO at Series.mean()
    def mean(self, verbose=True, decode=True, passes=None,
             num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to compute the mean of the values in each column

        Parameters
        ----------
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

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
            data.append(series.mean().evaluate(verbose, decode, passes,
                                               num_threads, apply_experimental_transforms))

        return Series(np.array(data),
                      np.dtype(np.float64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    # see TODO at Series.std()
    def std(self, verbose=True, decode=True, passes=None,
            num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to compute the standard deviation of the values in each column

        Parameters
        ----------
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

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
            data.append(series.std().evaluate(verbose, decode, passes,
                                              num_threads, apply_experimental_transforms))

        return Series(np.array(data),
                      np.dtype(np.float64),
                      Index(np.array(index).astype(np.str), np.dtype(np.str)))

    # noinspection PyMethodMayBeStatic
    def _merge_single(self, index1, index2):
        data = [get_expression_or_raw(index1), get_expression_or_raw(index2)]

        data = weld_merge_single_index(data, DataFrame._cache_flag)

        return [LazyResult(data[i], WeldBit(), 1) for i in xrange(2)]

    # noinspection PyMethodMayBeStatic
    def _merge_multi(self, index1, index2):
        assert len(index1.levels) == len(index2.levels) == 3

        index1 = index1.expand()
        index2 = index2.expand()

        data = weld_merge_triple_index([[get_expression_or_raw(index1[i]) for i in xrange(3)],
                                        [get_expression_or_raw(index2[i]) for i in xrange(3)]], DataFrame._cache_flag)

        return [LazyResult(data[i], WeldBit(), 1) for i in xrange(2)]

    # TODO: check for same column_names in both DataFrames!
    # TODO: currently a merge join implementation which works well if there are no duplicates, sorted data, and
    # the lengths of the dataframes are similar; should provide a hashjoin implementation (Weld dictmerger) otherwise
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

    def agg(self, aggregations, verbose=True, decode=True, passes=None,
            num_threads=1, apply_experimental_transforms=False):
        """ Eagerly aggregate the columns on multiple queries

        Parameters
        ----------
        aggregations : list of str
            list of desired aggregations; currently supported are
            {'sum', 'prod', 'min', 'max', 'count', 'mean', 'std'}
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyResult

        Returns
        -------
        DataFrame

        """
        if len(aggregations) < 1:
            raise TypeError('expected at least 1 aggregation')

        new_data = {column_name: self[str(column_name)].agg(aggregations, verbose, decode, passes, num_threads,
                                                            apply_experimental_transforms) for column_name in self}
        # get any column's index, since they're (should be) the same
        new_index = new_data[new_data.keys()[0]].index

        return DataFrame(new_data, new_index)

    def describe(self, aggregations):
        """ Lazily aggregate the columns in a single evaluate

        Parameters
        ----------
        aggregations : list of str
            list of desired aggregations; currently supported are
            {'sum', 'prod', 'min', 'max', 'mean', 'std'}

        Returns
        -------
        DataFrame

        """
        if len(aggregations) < 1:
            raise TypeError('expected at least 1 aggregation')

        new_data = {column_name: self[str(column_name)].describe(aggregations) for column_name in self}
        # get any column's index, since they're (should be) the same
        new_index = new_data[new_data.keys()[0]].index

        return DataFrame(new_data, new_index)

    # noinspection SpellCheckingInspection
    def groupby(self, by):
        """ Group by one or more columns

        Parameters
        ----------
        by : str or list of str
            to group by

        Returns
        -------
        DataFrameGroupBy

        """
        # to avoid circular dependency
        from pandas_weld.core.groupby import DataFrameGroupBy

        if isinstance(by, str):
            by_series = self[by]
            by_weld_types = [by_series.weld_type]
            by_data = [get_expression_or_raw(by_series)]
            by_types = [by_series.dtype]

            column_series = [self[column] for column in self if column is not by]
            # need by as a list
            by = [by]
        elif isinstance(by, list):
            if len(by) < 1:
                raise ValueError('expected a list with at least a value')

            by_series = [self[column] for column in by]
            by_weld_types = [k.weld_type for k in by_series]
            by_data = [get_expression_or_raw(column) for column in by_series]
            by_types = [column.dtype for column in by_series]

            column_series = [self[column] for column in self if column not in by]

        else:
            raise TypeError('expected one or more column names')

        column_names = [column.name for column in column_series]
        column_types = [column.dtype for column in column_series]
        column_weld_types = [column.weld_type for column in column_series]

        columns_to_group = [get_expression_or_raw(self[column]) for column in column_names]

        return DataFrameGroupBy(weld_groupby(by_data,
                                             by_weld_types,
                                             columns_to_group,
                                             column_weld_types),
                                by,
                                by_types,
                                column_names,
                                column_types)

    def reset_index(self):
        """ Returns a new DataFrame with previous Index as columns

        Returns
        -------
        DataFrame

        """
        new_columns = OrderedDict()

        # the index
        if isinstance(self.index, (Index, RangeIndex)):
            new_columns[self.index.name] = self.index
        else:
            # is MultiIndex
            for i in xrange(len(self.index.levels)):
                new_columns[self.index.names[i]] = \
                    LazyResult(weld_index_to_values(get_expression_or_raw(self.index.levels[i]),
                                                    get_expression_or_raw(self.index.labels[i])),
                               get_weld_type(self.index.levels[i]),
                               1)

        # the data/columns
        new_columns.update(self.data)

        # assumes at least 1 column
        a_column = get_expression_or_raw(new_columns.values()[-1])
        new_index = RangeIndex(0, LazyResult(weld_count(a_column), WeldLong(), 0).evaluate(), 1)

        return DataFrame(new_columns, new_index)

    def to_csv(self, path_or_buf, sep=',', header=True, index=True):
        """ Write DataFrame to file

        Note: forces evaluation!

        Parameters
        ----------
        path_or_buf : str
            path to file to write to
        sep : str, Optional
            delimiter to use
        header : bool, Optional
            whether to first write the header with the keys
        index : bool, Optional
            whether to write the index

        """
        import csv

        with open(path_or_buf, 'wb') as out_file:
            writer = csv.writer(out_file, delimiter=sep)

            if index:
                df = self.reset_index()
            else:
                df = self

            data = list(df.evaluate().data.itervalues())

            if header:
                writer.writerow(df.data.keys())

            for i in range(len(data[-1])):
                row = []
                for j in range(len(data)):
                    row.append(data[j][i])

                writer.writerow(row)
