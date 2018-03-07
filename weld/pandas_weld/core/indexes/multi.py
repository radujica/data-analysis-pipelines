import numpy_weld as npw


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

    def evaluate_all(self, verbose=True, decode=True, passes=None, num_threads=1,
                     apply_experimental_transforms=False):
        materialized_levels = {}
        materialized_labels = {}

        for i in xrange(len(self.names)):
            materialized_levels[self.names[i]] = self.levels[i].evaluate(verbose, decode, passes,
                                                                         num_threads, apply_experimental_transforms)
            materialized_labels[self.names[i]] = self.labels[i].evaluate(verbose, decode, passes,
                                                                         num_threads, apply_experimental_transforms)

        string_representation = """%(repr)s\nlevels:\n\t%(levels)s\nlabels:\n\t%(labels)s"""

        return string_representation % {'repr': self.__repr__(),
                                        'levels': materialized_levels,
                                        'labels': materialized_labels}
