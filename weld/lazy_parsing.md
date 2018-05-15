# Lazy file parsing with Weld

Everything revolves around `LazyResult` which tracks a (lazy) computation graph through `WeldObject`.

Each lazy parser shall generate a unique id (`data_id`) using the `generate_data_id` method which shall
act as placeholder for the data from file. This placeholder needs to be registered with Weld and
put as value in the contexts of the WeldObjects requiring this data. This means we need to
register this lazy data source now, even though it is not used; registering means a WeldObject must be updated
such that an input id (such as `_inp0`) will be generated. This is done through the `generate_placeholder_weld_object`
method which will create this placeholder object with the weld_input_id as weld_code (allowing evaluation right
away).

At this point, a WeldObject could look like: `_inp0 {'_inp0': '_data_id_netcdf4_column1'}`.

When the data is actually required, the placeholder needs to be replaced by the actual data when calling
evaluate. This is done by checking if the `str()` of the values in the WeldObject contexts corresponds to a `data_id`
in this mapping. To actually read the data, we need a mapping from `data_id` to a method which reads it. For this to
work, each parser shall subclass `LazyData` to represent a fragment of the data, with the relevant method being
`eager_read`. This method is then called by `LazyResult` during `evaluate`.

Since not all file formats require the same type of parsing (e.g. the netcdf4 format allows efficient reading of just
1 variable, while csv does not), we shall provide a general handler to the file which shall be used by parsers within
`eager_read`, hence allowing caching of the file. This is particularly useful in the csv example. This is provided
through the `LazyFile` interface.

Since the whole point of lazy parsing is to store if not all data is required, `LazyData` provides the methods
`lazy_slice_rows` and `lazy_skip_columns` which are called by `LazyResult`'s `update_rows` and `update_columns`,
respectively. These methods shall record in some fashion, up to the parser implementation, that only the given subset
of data shall be read, effectively it's as if that's the only data available; it shall apply to the _entire_ pipeline.
Therefore, a pipeline currently (see possible improvements) cannot require different subsets of the data in different
locations. This limitation, however, shall not be a hindrance to typical data analysis pipelines.

Lastly, because it is a very common operation, `LazyResult` features a `head` function which calls `LazyData.eager_head`;
this method is eager in that it is expected to bypass the lazy parameters stored and just returned the first n rows of the data.
It should, therefore, have no side-effects to a pipeline (i.e. the data also won't be cached).

### How to use

To implement your own lazy parser, subclass `LazyFile` and `LazyData`, e.g. `Dataset` and `Variable` for netcdf, and
interact with `LazyResult`'s `generate_*_id` methods and `register_*` to record lazy data existence. That's it.
`netCDF4_weld` and `csv_weld`, of course, stand as examples.

To implement a library such as pandas to work with `LazyResult`, one needs to be subclass `LazyResult` for each evaluable
data.

### Possible improvements
- using str(value) might not be very efficient; perhaps a better way to check/remove placeholders? The whole lazy-
parsing functionality should probably be implemented within WeldObject
- allow different subsets of the data to be read; currently, only one is possible; this shall perhaps be implemented
by generating a new id whenever a new subset is required; this is prerequisite to the next point
- could also be even more intricate such that upon evaluate, the total data required to read would be combined
into one, e.g. rows 3-7, 4-9, 15-20 would be combined into 3-9, 15-20 and read in a single pass
- caching should be smarter, e.g. allow different subsets of the data to be read in one go

# Intermediate Result Caching
With table joins as an example, one would like to store the intermediate result of which rows to keep from each
table. This is (naively) achieved by caching the result in memory after the first triggered computation, inspired
by Spark's .cache(). To implement weld operations with caching, one needs to generate an id for it with
`LazyResult.generate_intermediate_id`, register it with `LazyResult.register_intermediate_result`, and
probably `LazyResult.generate_placeholder_weld_object` which will only contain the result of this intermediate
result.

Turned on by default. Can be disabled by setting the environment variable `LAZY_WELD_CACHE` to `'False'`, i.e.

    export LAZY_WELD_CACHE='False'
