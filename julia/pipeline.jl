using NetCDF
using DataFrames

PATH = "/Workspace/data-analysis-pipelines/weld/pandas_weld/tests/io/"
FILENAME = "sample_ext.nc"

long_raw = ncread(PATH * FILENAME, "longitude")
lat_raw = ncread(PATH * FILENAME, "latitude")
time_raw = map(x -> Base.Dates.DateTime(1950, 1, x), ncread(PATH * FILENAME, "time") + 1)
tg = ncread(PATH * FILENAME, "tg")
tg_ext = ncread(PATH * FILENAME, "tg_ext")

# TODO: generify & make method
# julia DataFrames don't have (multi-)index like in pandas so need to make the cartesian product of the dimensions
longlat = [repmat(long_raw, 1, length(lat_raw))'[:] repmat(lat_raw, length(long_raw), 1)[:]]
dimensions = [repmat(longlat[:, 1], 1, length(time_raw))'[:] repmat(longlat[:, 2], 1, length(time_raw))'[:] repmat(time_raw, length(longlat[:, 1]), 1)[:]]

df = DataFrame(longitude=dimensions[:, 1], latitude=dimensions[:, 2], time=dimensions[:, 3], tg=tg[:], tg_ext=tg_ext[:])
