from collections import OrderedDict
from netCDF4 import num2date
from variable import Variable
import pandas_weld as pdw
import numpy as np
import pandas as pd


class Dataset(object):
    """ Welded wrapper of netCDF4 Dataset

    Parameters
    ----------
    ds : netCDF4 Dataset
        the original dataset
    variables : dict {name : netCDF4_weld Variables}
        used when doing operations on the Dataset to create new result Dataset
    dimensions : dict {name : size}
        used when doing operations on the Dataset to create new result Dataset

    See also
    --------
    netCDF4.Dataset

    """
    # used to assign a unique id to this dataset which is needed for variable tracking
    _dataset_counter = 0

    def __init__(self, ds, variables=None, dimensions=None):
        self.ds = ds
        self._id = self._create_dataset_id()

        if variables is None:
            # create OrderedDict of column_name -> Variable; use the variable name as expression/weld_code
            self.variables = OrderedDict(map(lambda kv: (kv[0],
                                                         Variable(ds,
                                                                  self._id,
                                                                  kv[0],
                                                                  kv[1].dimensions,
                                                                  kv[1].__dict__,
                                                                  kv[0],
                                                                  kv[1].dtype)),
                                             ds.variables.items()))
        else:
            self.variables = variables

        if dimensions is None:
            self.dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1].size), ds.dimensions.items()))
        else:
            self.dimensions = dimensions

        self._columns = [k for k in self.variables if k not in self.dimensions]

    def _create_dataset_id(self):
        ds_id = '_ds' + str(self._dataset_counter)
        self._dataset_counter += 1

        return ds_id

    # this materializes everything right now
    def evaluate_all(self, verbose=True):
        materialized_variables = {}
        for variable in self.variables.items():
            materialized_variables[variable[0]] = variable[1].evaluate(verbose=verbose)

        string_representation = """%(repr)s\ndata: %(data)s"""

        return string_representation % {'repr': self.__repr__(),
                                        'data': materialized_variables}

    # TODO: this shall check what variables are used later and only evaluate those
    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        pass

    # TODO: this could look nicer; look at how pandas does it?
    # currently also returns head data
    def __repr__(self):
        string_representation = """columns:\n\t%(columns)s\ndimensions: %(dimensions)s"""

        column_data = [self.variables[k].head() for k in self._columns]

        return string_representation % {'columns': '\n\t'.join(map(lambda x: str(x),
                                                                   zip(self._columns, column_data))),
                                        'dimensions': self.dimensions.keys()}

    # add number to each variable value; JUST for learning/testing purposes
    def add(self, value):
        return Dataset(self.ds,
                       {k: v.add(value) for k, v in self.variables.iteritems()},
                       self.dimensions)

    # TODO
    def _process_column(self, column_name, ordered_dimensions):
        full_variable = self.variables[column_name]
        variable_dims = full_variable.dimensions
        raw_data = full_variable[0:]  # full_variable.expression?

        shape = tuple(ordered_dimensions[d] for d in variable_dims)
        expanded_data = np.broadcast_to(raw_data.filled(np.nan), shape)
        axes = tuple(variable_dims.index(d) for d in ordered_dimensions)
        transposed_data = np.transpose(expanded_data, axes)
        reshaped_data = transposed_data.reshape(-1)
        return reshaped_data

    # TODO: type the computations in weld IR
    def to_dataframe(self):
        columns = [k for k in self.variables if k not in self.dimensions]
        ordered_dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1].size),
                                             OrderedDict(self.dimensions.items()).items()))

        # the data
        data = []
        for k in columns:
            data.append(self._process_column(k, ordered_dimensions))

        def convert_datetime(variable):
            return pd.to_datetime(num2date(variable[0:], variable.units, calendar=variable.calendar))

        # the dimensions
        indexes = [convert_datetime(self.ds.variables[k]) if hasattr(self.ds.variables[k], 'calendar')
                   else self.ds.variables[k][0:] for k in ordered_dimensions]
        index = pd.MultiIndex.from_product(indexes, names=ordered_dimensions)

        return pdw.DataFrame(dict(zip(columns, data)), index=index)
