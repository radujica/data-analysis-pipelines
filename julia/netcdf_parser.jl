function readvar(path::String, name::String)
    raw_data = ncread(path, name)
    scale_factor = ncgetatt(path, name, "scale_factor")
    if scale_factor != Void()
        raw_data *= scale_factor
    end

    raw_data[:]
end

# TODO: generalize(?)
# need to make the cartesian product of the dimensions
function cartesianproduct(long::Array{Float32}, lat::Array{Float32}, time::Array{DateTime})
    longlat = [repeat(long, inner=length(lat)) repeat(lat, outer=length(long))]

    [repeat(longlat[:, 1], inner=length(time)) repeat(longlat[:, 2], inner=length(time)) repeat(time, outer=length(longlat[:, 1]))]
end

# returns a dict of dimension => data
function readdims(path::String)
    long_raw = readvar(path, "longitude")
    lat_raw = readvar(path, "latitude")
    time_raw = map(x -> Base.Dates.DateTime(1950, 1, 1) + Base.Dates.Day(x), readvar(path, "time"))
    dims = cartesianproduct(long_raw, lat_raw, time_raw)
    d = OrderedDict()
    d["longitude"] = Array{Float32}(dims[:, 1])
    d["latitude"] = Array{Float32}(dims[:, 2])
    d["time"] = Array{DateTime}(dims[:, 3])

    d
end

# returns a dict of variable => data, for all variables not dimensions
function readdata(path::String)
    info = ncinfo(path)

    data_names = [k for k in keys(info.vars) if !(k in keys(info.dim))]

    d = Dict()
    for data_name in data_names
        d[data_name] = readvar(path, data_name)
    end

    d
end

# read the entire netcdf file as a DataFrame
function readnetcdf(path::String)
    dict = readdims(path)
    merge!(dict, readdata(path))

    DataFrame(dict)
end
