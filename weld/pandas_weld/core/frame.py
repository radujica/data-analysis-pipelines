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

    # this is special in that if there's a slice, it restricts all previous and next operations on the DataFrame
    def __getitem__(self, item):
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

    # TODO: check/test; currently does NOT test for same length
    def __setitem__(self, key, value):
        self.data[key] = value

    # TODO: action like evaluate; does NOT modify input_mapping func_args
    def head(self, n=100):
        raise NotImplementedError
