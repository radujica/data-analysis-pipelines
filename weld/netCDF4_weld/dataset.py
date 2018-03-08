from collections import OrderedDict
from variable import Variable
import pandas_weld as pdw


class Dataset(object):
    """ Welded wrapper of netCDF4 Dataset

    Parameters
    ----------
    read_file_func : func
        func which when called reads the file; in this case, must return a netCDF4.Dataset which can be used to
        read variables from
    variables : dict {name : netCDF4_weld.Variable}
        used when doing operations on the Dataset to create new result Dataset
    dimensions : dict {name : size}
        used when doing operations on the Dataset to create new result Dataset

    See also
    --------
    netCDF4.Dataset

    """
    # used to assign a unique id to this dataset which is needed for variable tracking
    _dataset_counter = 0

    def __init__(self, read_file_func, variables=None, dimensions=None):
        self.read_file_func = read_file_func
        # TODO: should cache this too
        self.ds = self.read_file_func()
        self._id = self._create_dataset_id()

        if variables is None:
            # create OrderedDict of column_name -> Variable; use the variable name as expression/weld_code
            self.variables = OrderedDict(map(lambda kv: (kv[0],
                                                         Variable(read_file_func,
                                                                  self._id,
                                                                  kv[0],
                                                                  kv[1].dimensions,
                                                                  kv[1].__dict__,
                                                                  kv[0],
                                                                  kv[1].dtype)),
                                             self.ds.variables.items()))
        else:
            self.variables = variables

        if dimensions is None:
            self.dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1].size), self.ds.dimensions.items()))
        else:
            self.dimensions = dimensions

        self._columns = [k for k in self.variables if k not in self.dimensions]

    def _create_dataset_id(self):
        ds_id = '_ds' + str(self._dataset_counter)
        Dataset._dataset_counter += 1

        return ds_id

    # this materializes everything right now
    def evaluate_all(self, verbose=True, decode=True, passes=None, num_threads=1,
                     apply_experimental_transforms=False):
        materialized_variables = {}
        for variable in self.variables.items():
            materialized_variables[variable[0]] = variable[1].evaluate(verbose, decode, passes,
                                                                       num_threads, apply_experimental_transforms)

        string_representation = """%(repr)s\ndata: %(data)s"""

        return string_representation % {'repr': self.__repr__(),
                                        'data': materialized_variables}

    # TODO: this shall check what variables are used later and only evaluate those
    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        pass

    # TODO: this could look nicer; look at how pandas does it?
    def __repr__(self):
        string_representation = """columns:\n\t%(columns)s\ndimensions: %(dimensions)s"""

        return string_representation % {'columns': self._columns,
                                        'dimensions': self.dimensions.keys()}

    # add number to each variable value; JUST for learning/testing purposes
    def add(self, value):
        return Dataset(self.read_file_func,
                       {k: v.add(value) for k, v in self.variables.iteritems()},
                       self.dimensions)

    # broadcast_to does not seem necessary (at least for my datasets)
    # TODO: add 1) transpose, and 2) flatten as weld ops
    def _process_column(self, column_name):
        return self.variables[column_name]

    def _process_dimension(self, name):
        return self.variables[name]

    def to_dataframe(self):
        """ Convert Dataset to pandas_weld DataFrame

        Returns
        -------
        pandas_weld.DataFrame

        """
        columns = [k for k in self.variables if k not in self.dimensions]
        ordered_dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1]),
                                             OrderedDict(sorted(self.dimensions.items())).items()))

        # columns data, either LazyData or raw
        data = [self._process_column(k) for k in columns]
        # the dimensions
        indexes = [self._process_dimension(k) for k in ordered_dimensions]

        index = pdw.MultiIndex.from_product(indexes, list(ordered_dimensions.keys()))

        return pdw.DataFrame(dict(zip(columns, data)), index)
