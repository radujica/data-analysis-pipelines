import numpy_weld as npw
from lazy_result import LazyResult
from collections import OrderedDict
from grizzly.encoders import numpy_to_weld_type
from pandas_weld.core.utils import evaluate_or_raw, get_expression_or_raw, get_weld_type, get_weld_info
from pandas_weld.weld import weld_filter, weld_unique


class MultiIndex(object):
    """ Weld-ed pandas MultiIndex

    It is assumed that the parameter lists are in the same order,
    i.e. same indexes match

    Parameters
    ----------
    levels : list of np.ndarray or list of WeldObject
    labels : list of np.ndarray or list of WeldObject
    names : list of str

    Returns
    -------
    MultiIndex

    See also
    --------
    pandas.MultiIndex

    """
    def __init__(self, levels, labels, names):
        self.levels = levels
        self.labels = labels
        self.names = names

    @classmethod
    def from_product(cls, levels, names):
        """ Create MultiIndex when no labels are available

        Parameters
        ----------
        levels : list of np.ndarray or list of WeldObject
        names : list of str

        Returns
        -------
        MultiIndex

        """
        labels = npw.cartesian_product_indices(levels)

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
        if isinstance(item, slice):
            # TODO: figure out a way to lazily slice the index
            return MultiIndex(self.levels, self.labels, self.names)
        elif isinstance(item, LazyResult):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected series of bool to filter DataFrame rows')

            # TODO: filter unnecessary levels too
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
                 apply_experimental_transforms=False):
        """ Evaluates by creating a str representation of the MultiIndex

        Parameters
        ----------
        see LazyResult

        Returns
        -------
        str

        """
        materialized_levels = OrderedDict()
        materialized_labels = OrderedDict()

        for i in xrange(len(self.names)):
            materialized_levels[self.names[i]] = evaluate_or_raw(self.levels[i], verbose, decode, passes,
                                                                 num_threads, apply_experimental_transforms)
            materialized_labels[self.names[i]] = evaluate_or_raw(self.labels[i], verbose, decode, passes,
                                                                 num_threads, apply_experimental_transforms)

        return "{}\n>> Levels\n{}\n>> Labels\n{}".format(self.__class__.__name__,
                                                         materialized_levels,
                                                         materialized_labels)
