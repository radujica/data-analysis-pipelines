from collections import OrderedDict

import netCDF4

from lazy_file import LazyFile
from lazy_result import LazyResult
from variable import Variable


class Dataset(LazyFile):
    """ Welded wrapper of netCDF4 Dataset

    Parameters
    ----------
    path : str
        path to netcdf4 file
    variables : dict
        name -> netCDF4_weld.Variable;
        used when doing operations on the Dataset to create new result Dataset
    dimensions : dict
        name -> size;
        used when doing operations on the Dataset to create new result Dataset

    See also
    --------
    netCDF4.Dataset

    """
    _FILE_FORMAT = 'netcdf4'

    def __init__(self, path, variables=None, dimensions=None):
        self.file_id = LazyResult.generate_file_id(Dataset._FILE_FORMAT)
        LazyResult.register_lazy_file(self.file_id, self)

        self.path = path
        self.ds = self.read_metadata()

        if variables is None:
            self.variables = self._create_variables()
        else:
            self.variables = variables

        if dimensions is None:
            self.dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1].size), self.ds.dimensions.items()))
        else:
            self.dimensions = dimensions

        self._columns = [k for k in self.variables if k not in self.dimensions]

    def read_metadata(self):
        return netCDF4.Dataset(self.path)

    def read_file(self):
        # for netcdf4, the method happens to be the same
        return self.read_metadata()

    # create OrderedDict of column_name -> Variable
    def _create_variables(self):
        variables = OrderedDict()

        for kv in self.ds.variables.items():
            # generate a data_id to act as placeholder to the data
            data_id = LazyResult.generate_data_id(kv[0])
            weld_obj, weld_input_id = LazyResult.generate_placeholder_weld_object(data_id,
                                                                                  Variable.encoder,
                                                                                  Variable.decoder)

            variable = Variable(self.file_id, kv[0], kv[1].dimensions, kv[1].shape,
                                kv[1].__dict__, weld_obj, kv[1].dtype)
            LazyResult.register_lazy_data(weld_input_id, variable)

            variables[kv[0]] = variable

        return variables

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

    # TODO: this could look nicer; look at how pandas does it?
    def __repr__(self):
        string_representation = """columns:\n\t%(columns)s\ndimensions: %(dimensions)s"""

        return string_representation % {'columns': self._columns,
                                        'dimensions': self.dimensions.keys()}

    # add number to each variable value; JUST for learning/testing purposes
    def add(self, value):
        return Dataset(self.path,
                       {k: v + value for k, v in self.variables.iteritems()},
                       self.dimensions)
