import numpy as np
import os
from grizzly.encoders import NumPyDecoder, NumPyEncoder
from grizzly.lazy_op import LazyOpResult
from weld.weldobject import WeldObject
from lazy_data import LazyData
from lazy_file import LazyFile
from copy import deepcopy


# TODO: better solutions for recurring problem of having to always check LazyResult or raw;
# maybe make container for all input, i.e. all raw data is wrapped in LazyOpResult?
class LazyResult(LazyOpResult):
    """ Extension of LazyOpResult adding lazy data reading

    The optional parameters are all required when recording data from a lazy parser

    Parameters
    ----------
    expr : np.ndarray / WeldObject
        what shall be evaluated
    weld_type : WeldType
        of the elements
    dim : int
        dimensionality of data

    """
    # file_ and input_ could be merged but kept separate to emphasize they're distinct concepts; intermediate similarly
    # lazy file is only used internally by parsers while lazy_data is used within WeldObject.context
    file_mapping = {}
    data_mapping = {}
    file_cache = {}
    data_cache = {}
    _file_counter = 0
    _data_counter = 0
    # stores weld_ids (e.g. _inpX) -> WeldObject or raw; naive caching similar to Spark's .cache()
    _intermediate_counter = 0
    intermediate_mapping = {}
    # to be able to set it globally (and compare with vs without)
    _cache_flag = False if os.getenv("LAZY_WELD_CACHE") == 'False' else True

    def __init__(self, expr, weld_type, dim):
        super(LazyResult, self).__init__(expr, weld_type, dim)

    @staticmethod
    def _generate_id(file_or_data_str, readable_reference=None):
        id_ = '_%s_id' % file_or_data_str
        if file_or_data_str == 'file':
            id_ = '%s%s' % (id_, LazyResult._file_counter)
            LazyResult._file_counter += 1
        elif file_or_data_str == 'data':
            id_ = '%s%s' % (id_, LazyResult._data_counter)
            LazyResult._data_counter += 1
        else:  # intermediate
            id_ = '%s%s' % (id_, LazyResult._intermediate_counter)
            LazyResult._intermediate_counter += 1

        if readable_reference is not None:
            if not isinstance(readable_reference, (str, unicode)):
                raise TypeError('readable_reference must be a str')
            id_ = '%s_%s' % (id_, readable_reference)

        return id_

    @staticmethod
    def generate_file_id(readable_reference=None):
        """ Generate an id for a lazy file input

        Parameters
        ----------
        readable_reference : str, optional
            included in the id for a more readable id

        Returns
        -------
        str
            id

        """
        return LazyResult._generate_id('file', readable_reference)

    @staticmethod
    def generate_data_id(readable_reference=None):
        """ Generate an id for a lazy data input

        Typically a 1d array, such as a column from a table

        Parameters
        ----------
        readable_reference : str, optional
            included in the id for a more readable id

        Returns
        -------
        str
            id

        """
        return LazyResult._generate_id('data', readable_reference)

    @staticmethod
    def generate_intermediate_id(readable_reference=None):
        """ Generate an id for an intermediate result cache

        Typically a 1d array, such as a column from a table

        Parameters
        ----------
        readable_reference : str, optional
            included in the id for a more readable id

        Returns
        -------
        str
            id

        """
        return LazyResult._generate_id('intermediate', readable_reference)

    @staticmethod
    def _check_add_args(id_, class_, expected_class):
        if not isinstance(id_, (str, unicode)):
            raise TypeError('id should be of type string but received id "{}" of type "{}"'.format(id_, type(id_)))

        if not isinstance(class_, expected_class):
            raise TypeError('expected LazyFile or LazyData but received {}'.format(type(class_)))

    @staticmethod
    def register_lazy_file(file_id, lazy_file):
        """ Only method linking an id to a LazyFile

        Parameters
        ----------
        file_id : str
            expected to be generated by generate_file_id
        lazy_file : LazyFile
            an instance of a LazyFile, typically from a parser

        """
        LazyResult._check_add_args(file_id, lazy_file, LazyFile)
        LazyResult.file_mapping[file_id] = lazy_file

    @staticmethod
    def register_lazy_data(data_id, lazy_data):
        """ Only method linking an id to a LazyData

        Parameters
        ----------
        data_id : str
            expected to be generated by generate_data_id
        lazy_data : LazyData
            an instance of a LazyData, typically from a parser

        """
        LazyResult._check_add_args(data_id, lazy_data, LazyData)
        LazyResult.data_mapping[data_id] = lazy_data

    @staticmethod
    def register_intermediate_result(intermediate_id, lazy_result):
        """ Only method linking an id to a LazyResult

        Parameters
        ----------
        intermediate_id : str
            expected to be generated by generate_data_id
        lazy_result : LazyResult
            an instance of a LazyResult

        """
        LazyResult._check_add_args(intermediate_id, lazy_result, LazyResult)
        LazyResult.intermediate_mapping[intermediate_id] = lazy_result

    def __repr__(self):
        return "{}(weld_type={}, dimension={})".format(self.__class__.__name__,
                                                       self.weld_type,
                                                       self.dim)

    def __str__(self):
        return str(self.expr)

    @staticmethod
    def retrieve_file(file_id):
        """ Returns the LazyFile instance associated with the file_id

        Intended to be used by a parser when reading a fragment of its data, such as a column.
        Using this method ensures the file once read is cached, especially for formats such as csv
        where reading a single column is not possible without passing the entire file (unlike netcdf4 which
        supports reading just a variable).

        Parameters
        ----------
        file_id : str
            expected to be generated by generate_file_id

        """
        if file_id in LazyResult.file_cache:
            return LazyResult.file_cache[file_id]
        else:
            data = LazyResult.file_mapping[file_id].read_file()

            LazyResult.file_cache[file_id] = data

            return data

    @staticmethod
    def retrieve_data(data_id):
        """ Returns the LazyData instance associated with the data_id

        Used within evaluate; enables caching of the data read.

        Parameters
        ----------
        data_id : str
            expected to be generated by generate_data_id

        """
        if data_id in LazyResult.data_cache:
            return LazyResult.data_cache[data_id]
        else:
            data = LazyResult.data_mapping[data_id].eager_read()

            LazyResult.data_cache[data_id] = data

            return data

    @staticmethod
    def fetch_intermediate_result(intermediate_id):
        entry = LazyResult.intermediate_mapping[intermediate_id]

        # so has not yet been evaluated
        if isinstance(entry, LazyResult):
            entry = entry.evaluate()

            # also store the evaluated result
            LazyResult.intermediate_mapping[intermediate_id] = entry
            return entry
        else:
            return entry

    @staticmethod
    def generate_placeholder_weld_object(data_id, encoder, decoder):
        """ Generates a WeldObject which will evaluate to the data
        represented by the placeholder

        Parameters
        ----------
        data_id : str
            expected from generate_data_id, yet not enforced
        encoder : WeldObjectEncoder
        decoder : WeldObjectDecoder

        Returns
        -------
        WeldObject
            with weld_code which would evaluate to the data itself
        """
        # create weld object which will represent this data
        weld_obj = WeldObject(encoder, decoder)
        # update the context of this WeldObject and retrieve the generated _inpX id; WeldObject._registry
        # will hence link this data_id to the _inpX id
        weld_input_id = weld_obj.update(data_id)
        # should always be a new object, else there's a bug somewhere
        assert weld_input_id is not None
        # the code is just the input
        weld_obj.weld_code = '%s' % weld_input_id

        return weld_obj

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        if isinstance(self.expr, WeldObject):
            # replace context values for every lazy recorded file input
            for key, value in self.expr.context.items():
                value = str(value)
                if value in LazyResult.data_mapping:
                    self.expr.context[key] = LazyResult.retrieve_data(value)

                if LazyResult._cache_flag:
                    if value in LazyResult.intermediate_mapping:
                        self.expr.context[key] = LazyResult.fetch_intermediate_result(value)

            return super(LazyResult, self).evaluate(verbose, decode, passes, num_threads, apply_experimental_transforms)
        else:
            return self.expr

    def _update_args(self, func_name, data):
        # this should only be called knowing expr is a WeldObject
        assert isinstance(self.expr, WeldObject)

        # loop through context to identify placeholders
        for value in self.expr.context.values():
            value = str(value)
            # so if value is a data_id, update the args of its read function
            if value in LazyResult.data_mapping:
                lazy_data_class = LazyResult.data_mapping[value]
                getattr(lazy_data_class, func_name)(data)

    def update_columns(self, columns):
        """ Update the columns required from a lazy data source

        Parameters
        ----------
        columns : list of str
            which columns to not read anymore

        """
        if not isinstance(self.expr, np.ndarray):
            self._update_args('lazy_skip_columns', columns)

    def update_rows(self, slice_):
        """ Update the columns required from a lazy data source

        Parameters
        ----------
        slice_ : slice
            which subset of the data/rows to read

        """
        # if np.ndarray, lazily slice the data
        if isinstance(self.expr, np.ndarray):
            self.expr = weld_subset(self.expr, slice_)
        else:
            self._update_args('lazy_slice_rows', slice_)

    def _copy(self):
        copy_expr = WeldObject(self.expr.encoder, self.expr.decoder)
        copy_expr.weld_code = self.expr.weld_code
        copy_expr.context = deepcopy(self.expr.context)
        copy_expr.dependencies = self.expr.dependencies.copy()
        copy_expr.argtypes = deepcopy(self.expr.argtypes)

        return LazyResult(copy_expr, self.weld_type, self.dim)

    def head(self, n=10):
        """ Eagerly read the first n rows of the data.

        Does not have side-effects, such as caching or WeldObject context changes.

        Parameters
        ----------
        n : int, optional
            how many rows to read

        """
        # shortcut for np.ndarray
        if isinstance(self.expr, np.ndarray):
            return self.expr[:n]
        else:
            # this should only be called knowing expr is a WeldObject
            assert isinstance(self.expr, WeldObject)

            # copy self to avoid side-effects
            copy = self._copy()

            # loop through context to identify placeholders AND replace with the take_n values
            for key, value in copy.expr.context.items():
                value = str(value)
                # so if value is a data_id, update the args of its read function
                if value in LazyResult.data_mapping:
                    copy.expr.context[key] = LazyResult.data_mapping[value].eager_head(n)

            return copy.evaluate()[:n]


def weld_subset(array, slice_):
    """ Return a subset of the input array

    Parameters
    ----------
    array : np.array or WeldObject
        1-dimensional array
    slice_ : slice
        subset to return

    Returns
    -------
    WeldObject
        representation of this computation

    """
    weld_obj = WeldObject(NumPyEncoder(), NumPyDecoder())

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    if slice_.step == 1:
        weld_template = """
        slice(
            %(array)s,
            %(slice_start)s,
            %(slice_stop)s
        )"""
    else:
        weld_template = """
        result(
            for(
                iter(%(array)s, %(slice_start)s, %(slice_stop)s, %(slice_step)s),
                appender,
                |b, i, n| 
                    merge(b, n)
            )  
        )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'slice_start': 'i64(%s)' % slice_.start,
                                          'slice_stop': 'i64(%s)' % (slice_.stop - slice_.start),
                                          'slice_step': 'i64(%s)' % slice_.step}

    return weld_obj
