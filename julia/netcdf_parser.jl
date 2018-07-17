function readvar(path::String, name::String)
    raw_data = ncread(path, name)
    scale_factor = ncgetatt(path, name, "scale_factor")
    if scale_factor != Void()
        raw_data *= scale_factor
    end

    raw_data[:]
end

# # TODO: generalize(?) ===>> this code is slower than it should
# # need to make the cartesian product of the dimensions
# function cartesianproduct(long::Array{Float32}, lat::Array{Float32}, time::Array{DateTime})
#     longlat = [repeat(long, inner=length(lat)) repeat(lat, outer=length(long))]
#
#     [repeat(view(longlat, :, 1), inner=length(time)) repeat(view(longlat, :, 2), inner=length(time)) repeat(time, outer=length(view(longlat, :, 1)))]
# end
#
# # returns a dict of dimension => data
# function readdims(path::String)
#     long_raw = readvar(path, "longitude")
#     lat_raw = readvar(path, "latitude")
#     time_raw = map(x -> Base.Dates.DateTime(1950, 1, 1) + Base.Dates.Day(x), readvar(path, "time"))
#     dims = cartesianproduct(long_raw, lat_raw, time_raw)
#     d = OrderedDict()
#     d["longitude"] = Array{Float32}(view(dims, :, 1))
#     d["latitude"] = Array{Float32}(view(dims, :, 2))
#     d["time"] = Array{DateTime}(view(dims, :, 3))
#
#     d
# end


# looped version, significantly faster for some reason
function cartesianproduct(long::Array{Float32}, lat::Array{Float32}, time::Array{Int32})
    length_long = length(long)
    length_lat = length(lat)
    length_time = length(time)
    total_size = length_long * length_lat * length_time

    d = OrderedDict()

    # inner x2
    new_long = zeros(Float32, total_size)
    times = length_lat * length_time
    index = 1
    for i = 1:length_long
        for j = 1:times
            new_long[index] = long[i]
            index += 1
        end
    end

    d["longitude"] = new_long

    # outer + inner
    new_lat_temp = zeros(Float32, length_long * length_lat)
    index = 1
    for i = 1:length_long
        for j = 1:length_lat
            new_lat_temp[index] = long[j]
            index += 1
        end
    end
    length_new_lat_temp = length(new_lat_temp)
    new_lat = zeros(Float32, total_size)
    index = 1
    for i = 1:length_new_lat_temp
        for j = 1:length_time
            new_lat[index] = new_lat_temp[i]
            index += 1
        end
    end

    d["latitude"] = new_lat

    # outer x2
    new_time = zeros(Int32, total_size)
    times = length_long * length_lat
    index = 1
    for i = 1:times
        for j = 1:length_time
            new_time[index] = time[j]
            index += 1
        end
    end

    d["time"] = new_time

    d
end

# returns a dict of dimension => data
function readdims(path::String)
    long_raw = readvar(path, "longitude")
    lat_raw = readvar(path, "latitude")
    time_raw = readvar(path, "time")
    d = cartesianproduct(long_raw, lat_raw, time_raw)
    d["time"] = map(x -> Base.Dates.DateTime(1950, 1, 1) + Base.Dates.Day(x), d["time"])

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
