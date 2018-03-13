from grizzly.encoders import numpy_to_weld_type_mapping
from lazy_data import LazyData
from series import Series
from utils import replace_slice_defaults
import numpy as np


class DataFrame(object):
    """ Weld-ed pandas DataFrame

    Parameters
    ----------
    data : dict
        column names -> data array or LazyData
    index : pandas_weld.MultiIndex
        index

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

    # this materializes everything right now
    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        materialized_columns = {}
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
        """ Retrieve either column or slice of data

        Has consequences! Any previous and/or following operations on
        the data within will be done only on this subset of the data

        Parameters
        ----------
        item : str or slice
            if str, returns a column as a Series; if slice, returns a sliced DataFrame

        Returns
        -------
        Series or DataFrame

        """
        if isinstance(item, str):
            element = self.data[item]
            if isinstance(element, LazyData):
                return Series(self.data[item].expr,
                              self.data[item].weld_type,
                              self.data[item].data_id)
            elif isinstance(element, np.ndarray):
                return Series(element,
                              numpy_to_weld_type_mapping[str(element.dtype)])
            else:
                raise ValueError('column is neither LazyData nor np.ndarray')
        elif isinstance(item, slice):
            item = replace_slice_defaults(item)

            new_data = {}
            for column_name in self.data:
                # making series because Series has the proper method to slice something; re-use the code above
                series = self[str(column_name)]

                new_data[column_name] = series[item]

            new_index = self.index[item]

            # making a new dataframe here seems kinda pointless atm due to func_args being updated
            return DataFrame(new_data,
                             new_index)
        else:
            raise ValueError('expected a str or slice in DataFrame.__getitem__')

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
        for column_name in self.data:
            # making series because Series has the proper method to slice something; re-use the code above
            series = self[str(column_name)]

            # by not passing a data_id, the data is not linked to the input_mapping read
            new_data[column_name] = Series(series.head(n), series.weld_type)

        new_index = self.index[slice_]

        return DataFrame(new_data, new_index).evaluate(verbose=False)

    def __setitem__(self, key, value):
        self.data[key] = value
