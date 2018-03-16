import numpy as np
import numpy_weld as npw
from lazy_data import LazyData
from collections import OrderedDict
from grizzly.encoders import numpy_to_weld_type
from pandas_weld.core.series import Series
from pandas_weld.weld import weld_filter


class MultiIndex(object):
    """ Weld-ed pandas MultiIndex

    It is assumed that the parameter lists are in the same order,
    i.e. same indexes match

    Parameters
    ----------
    levels : list
        np.arrays or LazyData
    labels : list
        np.arrays or LazyData
    names : list
        str

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
        levels : list
            np.arrays or WeldObjects
        names : list
            names of the levels

        Returns
        -------
        MultiIndex

        """
        labels = npw.cartesian_product_indices(levels)

        return cls(levels, labels, names)

    # only names are (already) materialized
    def __repr__(self):
        return "index names:\n\t%s" % str(self.names)

    def __getitem__(self, item):
        """ Retrieve a portion of the MultiIndex

        Parameters
        ----------
        item : slice or Series
            if slice, returns a sliced MultiIndex;
            if Series, returns a filtered MultiIndex only with the labels corresponding to
            True in the Series

        Returns
        -------
        MultiIndex

        """
        if isinstance(item, slice):
            # TODO: figure out a way to slice the index; each data variable might have different dimensions order (?)
            # so it seems more complicated than just adding a parameter to the variable read_data
            return MultiIndex(self.levels, self.labels, self.names)
        elif isinstance(item, Series):
            if not item.weld_type == numpy_to_weld_type('bool'):
                raise ValueError('expected series of bool to filter DataFrame rows')

            new_labels = []
            for label in self.labels:
                if isinstance(label, LazyData):
                    weld_type = label.weld_type
                    data_id = label.data_id
                    label = label.expr
                elif isinstance(label, np.ndarray):
                    weld_type = numpy_to_weld_type(label.dtype)
                    data_id = None
                else:
                    raise TypeError('expected data in column to be of type LazyData or np.ndarray')

                new_labels.append(LazyData(weld_filter(label,
                                                       item.expr,
                                                       weld_type),
                                           weld_type,
                                           1,
                                           data_id))

            return MultiIndex(self.levels, new_labels, self.names)
        else:
            raise TypeError('expected slice or Series of bool in MultiIndex.__getitem__')

    @staticmethod
    def _evaluate_or_raw(array, verbose, decode, passes,
                         num_threads, apply_experimental_transforms):
        if isinstance(array, LazyData):
            return array.evaluate(verbose, decode, passes,
                                  num_threads, apply_experimental_transforms)
        elif isinstance(array, np.ndarray):
            return array
        else:
            raise TypeError('expected LazyData or np.ndarray')

    # TODO: prettify
    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        """ Evaluates by creating a str representation of the MultiIndex

        Parameters
        ----------
        see LazyData

        Returns
        -------
        str

        """
        materialized_levels = OrderedDict()
        materialized_labels = OrderedDict()

        for i in xrange(len(self.names)):
            materialized_levels[self.names[i]] = MultiIndex._evaluate_or_raw(self.levels[i], verbose, decode, passes,
                                                                             num_threads, apply_experimental_transforms)
            materialized_labels[self.names[i]] = MultiIndex._evaluate_or_raw(self.labels[i], verbose, decode, passes,
                                                                             num_threads, apply_experimental_transforms)

        string_representation = """%(repr)s\nlevels:\n\t%(levels)s\nlabels:\n\t%(labels)s"""

        return string_representation % {'repr': self.__repr__(),
                                        'levels': materialized_levels,
                                        'labels': materialized_labels}
