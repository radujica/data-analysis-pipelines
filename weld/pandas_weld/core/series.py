from collections import OrderedDict
from grizzly.encoders import numpy_to_weld_type
from weld.types import WeldLong, WeldDouble
from weld.weldobject import WeldObject
from lazy_data import LazyData
from indexes import Index
from pandas_weld.weld import weld_aggregate, weld_compare, weld_filter, weld_element_wise_op, weld_count, weld_mean, \
    weld_standard_deviation, weld_udf
from utils import subset, replace_slice_defaults
import numpy as np


class Series(LazyData):
    """ Weld-ed pandas Series

    Parameters
    ----------
    data : np.ndarray / WeldObject
        raw data or weld expression
    dtype : np.dtype
        of the elements
    index : Index, RangeIndex, or MultiIndex
        index linked to the data; it is assumed to be of the same length
    name : str, optional
        name of the series

    See also
    --------
    pandas.Series

    """

    # TODO: accept index as None and encode as RangeIndex(stop=len(self.expr))
    def __init__(self, data, dtype, index, name=None):
        if not isinstance(data, (np.ndarray, WeldObject)):
            raise TypeError('expected np.ndarray or WeldObject in Series.__init__')

        super(Series, self).__init__(data, numpy_to_weld_type(dtype), 1)

        self.dtype = dtype
        self.index = index
        self.name = name

    @property
    def data(self):
        return self.expr

    def __repr__(self):
        return "{}(name={}, dtype={}, index={})".format(self.__class__.__name__,
                                                        self.name,
                                                        self.dtype,
                                                        repr(self.index))

    def __str__(self):
        return str(self.data)

    def __getitem__(self, item):
        """ Lazy operation to select a subset of the series

        Has consequences! When slicing, any previous and/or following operations on
        the data within will be done only on this subset of the data

        Parameters
        ----------
        item : slice or Series
            if slice, a slice of the data for the number of desired rows; currently
            must contain a stop value and will not work as expected for
            start != 0 and stride != 1;
            if Series, returns a filtered Series only with the elements corresponding to
            True in the item Series

        Returns
        -------
        Series

        """
        if isinstance(item, slice):
            item = replace_slice_defaults(item)

            # update func_args so that less data is read from file
            if isinstance(self, LazyData) and self.data_id is not None:
                index = self.input_mapping.data_ids.index(self.data_id)
                old_args = self.input_mapping.input_function_args[index]
                slice_as_tuple = (slice(item.start, item.stop, item.step),)
                new_args = old_args + (slice_as_tuple,)
                self.input_mapping.update_input_function_args(index, new_args)

            new_index = self.index[item]

            return Series(subset(self, item).expr,
                          self.dtype,
                          new_index,
                          self.name)
        elif isinstance(item, LazyData):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected series of bool to filter DataFrame rows')

            new_index = self.index[item]

            return Series(weld_filter(self.expr,
                                      item.expr,
                                      self.weld_type),
                          self.dtype,
                          new_index,
                          self.name)
        else:
            raise TypeError('expected a slice or a Series of bool in Series.__getitem__')

    def head(self, n=10, verbose=False, decode=True, passes=None,
             num_threads=1, apply_experimental_transforms=False):
        """ Eager operation to read first n rows

        This operation has no consequences, unlike getitem

        Parameters
        ----------
        n : int
            how many rows
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyData

        Returns
        -------
        np.array
            or other raw data

        """
        slice_ = replace_slice_defaults(slice(n))
        data = self.expr

        if self.data_id is not None:
            index = self.input_mapping.data_ids.index(self.data_id)
            old_args = self.input_mapping.input_function_args[index]
            slice_as_tuple = (slice_,)
            new_args = old_args + (slice_as_tuple,)
            data = self.input_mapping.input_functions[index](*new_args)

        if isinstance(data, WeldObject):
            data = self.evaluate(verbose, decode, passes, num_threads, apply_experimental_transforms)
        elif isinstance(data, np.ndarray):
            data = data[:n]
        else:
            raise TypeError('underlying data is neither LazyData nor np.ndarray')

        return data

    # comparisons are limited to scalars
    # TODO: perhaps storing boolean masks is more efficient? ~ bitwise-and vs merged for-loop map
    def _comparison(self, other, comparison):
        if not isinstance(other, (str, unicode, int, long, float, bool)):
            raise TypeError('can only compare with scalars')

        assert isinstance(comparison, (str, unicode))

        return Series(weld_compare(self.expr,
                                   other,
                                   comparison,
                                   self.weld_type),
                      np.dtype(np.bool),
                      self.index,
                      self.name)

    def __lt__(self, other):
        return self._comparison(other, '<')

    def __le__(self, other):
        return self._comparison(other, '<=')

    def __eq__(self, other):
        return self._comparison(other, '==')

    def __ne__(self, other):
        return self._comparison(other, '!=')

    def __ge__(self, other):
        return self._comparison(other, '>=')

    def __gt__(self, other):
        return self._comparison(other, '>')

    # TODO: add type conversion(?); pandas works when e.g. column_of_ints - 2.0 => float result
    def _element_wise_operation(self, other, operation):
        if not isinstance(other, (str, unicode, int, long, float, bool)):
            raise TypeError('can only compare with scalars')

        assert isinstance(operation, (str, unicode))

        return Series(weld_element_wise_op(self.expr,
                                           other,
                                           operation,
                                           self.weld_type),
                      self.dtype,
                      self.index,
                      self.name)

    def __add__(self, other):
        return self._element_wise_operation(other, '+')

    def __sub__(self, other):
        return self._element_wise_operation(other, '-')

    def __mul__(self, other):
        return self._element_wise_operation(other, '*')

    def __div__(self, other):
        return self._element_wise_operation(other, '/')

    def _aggregate(self, operation):
        assert isinstance(operation, (str, unicode))

        return LazyData(weld_aggregate(self.expr,
                                       operation,
                                       self.weld_type),
                        self.weld_type,
                        0)

    def sum(self):
        return self._aggregate('+')

    def prod(self):
        return self._aggregate('*')

    def min(self):
        return self._aggregate('min')

    def max(self):
        return self._aggregate('max')

    def count(self):
        return LazyData(weld_count(self.expr),
                        WeldLong(),
                        0)

    # TODO: not safe against overflows, i.e. the sum in sum/length
    def mean(self):
        return LazyData(weld_mean(self.expr,
                                  self.weld_type),
                        WeldDouble(),
                        0)

    # TODO: same as mean
    def std(self):
        return LazyData(weld_standard_deviation(self.expr,
                                                self.weld_type),
                        WeldDouble(),
                        0)

    def agg(self, aggregations, verbose=False, decode=True, passes=None,
            num_threads=1, apply_experimental_transforms=False):
        """ Eagerly aggregate on multiple queries

        Parameters
        ----------
        aggregations : list of str
            supported aggregations are = {'sum', 'prod', 'min', 'max', 'count', 'mean', 'std'}
        verbose, decode, passes, num_threads, apply_experimental_transforms
            see LazyData

        Returns
        -------
        DataFrame

        """
        if len(aggregations) < 1:
            raise TypeError('expected at least 1 aggregation')

        aggregation_results = OrderedDict()
        for aggregation in aggregations:
            # call the same-name function to compute the aggregation
            aggregation_results[str(aggregation)] = getattr(self, aggregation)()

        values = np.array([k.evaluate(verbose, decode, passes, num_threads,
                                      apply_experimental_transforms) for k in aggregation_results.values()])
        index = np.array(aggregation_results.keys(), dtype=np.str)

        return Series(values.astype(np.float64),
                      np.dtype(np.float64),
                      Index(index, np.dtype(np.float64)),
                      self.name)

    # TODO: implement something like reduce for single result? though this is not supported even by pandas
    def map(self, weld_template, mapping):
        """ Apply custom weld code to this series

        Cannot accept lambdas such as pandas' map, though behaves in a similar fashion

        Parameters
        ----------
        weld_template : str
            the code that will be recorded for execution; it is assumed that the
            length of the resulting Series will be the same after execution
        mapping : dict
            maps placeholders from weld_template to either arrays (np.array or WeldObject) or scalars;
            self (this Series) is included by default as 'self'

        Returns
        -------
        Series
            with the same index

        Examples
        --------
        >>> series = Series(np.array([1, 3, 4]), np.dtype(np.int64), RangeIndex(0, 3, 1))
        >>> weld_template = "map(%(self)s, |e| e + %(scalar)s)"
        >>> mapping = {'scalar': '2L'}
        >>> series.map(weld_template, mapping).evaluate()
        [3 5 6]

        """
        if not isinstance(mapping, dict):
            raise TypeError('mapping must be a dict mapping string placeholders to data')

        mapping.update({'self': self.expr})

        return Series(weld_udf(weld_template,
                               mapping),
                      self.dtype,
                      self.index,
                      self.name)
