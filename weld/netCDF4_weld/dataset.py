from collections import OrderedDict
from weld.weldobject import WeldObject
from lazy_data import LazyData
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

    def __init__(self, read_file_func, variables=None, dimensions=None):
        self.read_file_func = read_file_func
        self.ds = self.read_file_func()

        if variables is None:
            self.variables = self._create_variables(read_file_func)
        else:
            self.variables = variables

        if dimensions is None:
            self.dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1].size), self.ds.dimensions.items()))
        else:
            self.dimensions = dimensions

        self._columns = [k for k in self.variables if k not in self.dimensions]

    # create OrderedDict of column_name -> Variable
    def _create_variables(self, read_file_func):
        variables = OrderedDict()

        for kv in self.ds.variables.items():
            # create weld object which will represent this data
            weld_obj = WeldObject(Variable.encoder, Variable.decoder)
            # generate a data_id to act as placeholder to the data
            data_id = LazyData.generate_id(kv[0])
            # update the context of this WeldObject and retrieve the generated _inpX id; WeldObject._registry
            # will hence link this data_id to the _inpX id
            weld_input_id = weld_obj.update(data_id)
            # should always be a new object, else there's a bug somewhere
            assert weld_input_id is not None
            # the code is just the input
            weld_obj.weld_code = '%s' % weld_input_id

            variables[kv[0]] = Variable(read_file_func, kv[0], data_id, kv[1].dimensions,
                                        kv[1].__dict__, weld_obj, kv[1].dtype)

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
        return Dataset(self.read_file_func,
                       {k: v.add(value) for k, v in self.variables.iteritems()},
                       self.dimensions)

    def _process_column(self, column_name):
        return self.variables[column_name]

    def _process_dimension(self, name):
        return self.variables[name]

    # TODO: move to pandas_weld; conceptually, makes more sense there
    def to_dataframe(self):
        """ Convert Dataset to pandas_weld DataFrame

        Returns
        -------
        pandas_weld.DataFrame

        """
        columns = [k for k in self.variables if k not in self.dimensions]
        dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1]),
                                     OrderedDict(self.dimensions.items()).items()))

        # columns data, either LazyData or raw
        data = [self._process_column(k) for k in columns]
        # the dimensions
        indexes = [self._process_dimension(k) for k in dimensions]

        index = pdw.MultiIndex.from_product(indexes, list(dimensions.keys()))

        return pdw.DataFrame(dict(zip(columns, data)), index)
