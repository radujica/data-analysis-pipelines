from collections import OrderedDict
from tabulate import tabulate
from lazy_result import LazyResult, weld_subset
from grizzly.encoders import numpy_to_weld_type
from pandas_weld.core.utils import get_expression_or_raw, get_weld_type, get_weld_info, \
    replace_slice_defaults
from pandas_weld.weld import weld_filter, weld_unique, weld_index_to_values

import numpy_weld as npw
import os


class MultiIndex(object):
    """ Weld-ed pandas MultiIndex

    It is assumed that the parameter lists are in the same order,
    i.e. same indexes match

    Parameters
    ----------
    levels : list of np.ndarray or list of LazyResult
    labels : list of np.ndarray or list of LazyResult
    names : list of str

    Returns
    -------
    MultiIndex

    See also
    --------
    pandas.MultiIndex

    """
    _cache_flag = False if os.getenv("LAZY_WELD_CACHE") == 'False' else True

    # TODO: add some caching if columns were already expanded?
    def __init__(self, levels, labels, names):
        self.levels = levels
        self.labels = labels
        self.names = names

    @classmethod
    def from_product(cls, levels, names):
        """ Create MultiIndex when no labels are available

        Parameters
        ----------
        levels : list of np.ndarray or list of LazyResult
        names : list of str

        Returns
        -------
        MultiIndex

        """
        labels = npw.cartesian_product_indices(levels, MultiIndex._cache_flag)

        return cls(levels, labels, names)

    @classmethod
    def from_arrays(cls, arrays, names):
        weld_types = [get_weld_type(k) for k in arrays]
        arrays = [get_expression_or_raw(k) for k in arrays]
        levels = [LazyResult(weld_unique(arrays[k], weld_types[k]), weld_types[k], 1) for k in xrange(len(arrays))]

        return cls.from_product(levels, names)

    def __repr__(self):
        return "{}(names={})".format(self.__class__.__name__,
                                     self.names)

    def __str__(self):
        return "{}\n>> Levels\n{}\n>> Labels\n{}".format(self.__class__.__name__,
                                                         str(self.levels),
                                                         str(self.labels))

    def __getitem__(self, item):
        """ Retrieve a portion of the MultiIndex

        Parameters
        ----------
        item : slice or LazyResult
            if slice, returns a sliced MultiIndex;
            if LazyResult, returns a filtered MultiIndex only with the labels corresponding to
            True in the LazyResult

        Returns
        -------
        MultiIndex

        """
        # TODO: filter unnecessary levels too, both slice and LazyResult
        if isinstance(item, slice):
            item = replace_slice_defaults(item)

            new_labels = [LazyResult(weld_subset(get_expression_or_raw(label),
                                                 item),
                                     get_weld_type(label),
                                     1)
                          for label in self.labels]

            return MultiIndex(self.levels, new_labels, self.names)
        elif isinstance(item, LazyResult):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected series of bool to filter DataFrame rows')

            new_labels = []
            for label in self.labels:
                label, weld_type = get_weld_info(label, True, True)

                new_labels.append(LazyResult(weld_filter(label,
                                                         item.expr),
                                             weld_type,
                                             1))

            return MultiIndex(self.levels, new_labels, self.names)
        else:
            raise TypeError('expected slice or LazyResult of bool in MultiIndex.__getitem__')

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False, as_dict=False):
        """ Evaluates by creating a str representation of the MultiIndex

        Parameters
        ----------
        see LazyResult

        Returns
        -------
        str

        """
        index_columns = [index_to_values(self.levels[i],
                                         self.labels[i]).evaluate(verbose, decode, passes, num_threads,
                                                                  apply_experimental_transforms)
                         for i in xrange(len(self.levels))]

        if as_dict:
            return OrderedDict(zip(self.names, index_columns))
        else:
            return tabulate(index_columns, headers=self.names)


def index_to_values(levels, labels):
    levels, weld_type = get_weld_info(levels, expression=True, weld_type=True)

    if isinstance(labels, LazyResult):
        labels = labels.expr

    return LazyResult(weld_index_to_values(levels, labels), weld_type, 1)
