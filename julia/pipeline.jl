using NetCDF
using DataFrames
using DataStructures


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




PATH = ENV["HOME2"] * "/datasets/ECAD/original/small_sample/"
df1 = readnetcdf(PATH * "data1.nc")
df2 = readnetcdf(PATH * "data2.nc")

#= PIPELINE =#
# 1. join the 2 dataframes
df = join(df1, df2, on=[:longitude, :latitude, :time], kind=:inner)

# 2. quick preview on the data
println(DataFrames.head(df, 10))

# 3. subset the data
df = df[(df[:latitude] .>= 42.25f0) .& (df[:latitude] .<= 60.25f0), :]
# added +1 because of index from 1 not 0; this does NOT work correctly for some reason
# df = df[709921:1482480, :]

# 4. drop rows with null values
df = df[(df[:tg] .!= -99.99f0) .& (df[:pp] .!= -999.9f0) .& (df[:rr] .!= -999.9f0), :]

# 5. drop columns
delete!(df, [:pp_err, :rr_err])

# 6. UDF 1: compute absolute difference between max and min
function compute_abs_maxmin(column_max, column_min)
    abs.(column_max .- column_min)
end

df[:abs_diff] = compute_abs_maxmin(df[:tx], df[:tn])

# 7. explore the data through aggregations
# TODO: maybe  could use aggregate(df, <no_column>, [min...])?
function aggregations(df::DataFrame, aggregations::Array{String})
    dict = OrderedDict()
    columns_without_time = [k for k in names(df) if !(k == :time)]
    # add the columns to the dataframe
    dict["columns"] = columns_without_time

    for function_name in aggregations
        func = getfield(Main, Symbol(function_name))
        dict[function_name] = colwise(func, df[columns_without_time])
    end

    DataFrame(dict)
end

println(aggregations(df, ["minimum", "maximum", "mean", "std"]))

# 8. compute std per month
# UDF 2: compute custom year+month format
df[:year_month] = map(x -> Dates.year(x) * 100 + Dates.month(x), df[:time])

# groupby cols on on and rename the final columns
on = [:longitude, :latitude, :year_month]
cols = [:tg, :tn, :tx, :pp, :rr]
new_cols = [Symbol(string(k) * "_mean") for k in cols]
means = by(df[[on; cols]], on, df -> DataFrame(OrderedDict(zip(new_cols, colwise(mean, df[cols])))))
# rename!(means, Dict(:tg => :tg_mean, :tn => :tn_mean, :tx => :tx_mean, :pp => :pp_mean, :rr => :rr_mean))
# merge results
df = join(df, means, on=on)
delete!(df, :year_month)

# 9. EVALUATE
println(df)
