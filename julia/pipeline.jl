using NetCDF
using DataFrames
using DataStructures
using ArgParse
using CSV

include("netcdf_parser.jl")

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input", "-i"
            help = "Path to folder containing input files"
            arg_type = String
            required = true
        "--slice", "-s"
            help = "Start and stop of a subset of the data"
            arg_type = String
            required = true
        "--output", "-o"
            help = "Path to output folder"
            arg_type = String
            required = true
    end

    return parse_args(s)
end


print_event(name::String) = println("#" * Dates.format(now(), "HH:MM:SS") * "-" * name)


function main()
    parsed_args = parse_command_line()
    df1 = readnetcdf(parsed_args["input"] * "data1.nc")
    df2 = readnetcdf(parsed_args["input"] * "data2.nc")

    print_event("done_read")

    #= PIPELINE =#
    # 1. join the 2 dataframes
    df = join(df1, df2, on=[:longitude, :latitude, :time], kind=:inner)

    # 2. quick preview on the data
    df_head = DataFrames.head(df, 10)

    print_event("done_head")

    df_head[:time] = Dates.format.(df_head[:time], "yyyy-mm-dd")
    CSV.write(parsed_args["output"] * "head.csv", df_head)

    # 3. subset the data
    slice = split(parsed_args["slice"], ":")
    df = df[parse(Int64, slice[1]) + 1:parse(Int64, slice[2]), :]

    # 4. drop rows with null values
    df = df[(df[:tg] .!= -99.99f0) .& (df[:pp] .!= -999.9f0) .& (df[:rr] .!= -999.9f0), :]

    # 5. drop columns
    delete!(df, [:pp_stderr, :rr_stderr])

    # 6. UDF 1: compute absolute difference between max and min
    function compute_abs_maxmin(column_max, column_min)
        abs.(column_max .- column_min)
    end

    df[:abs_diff] = compute_abs_maxmin(df[:tx], df[:tn])

    # 7. explore the data through aggregations
    # TODO: maybe  could use aggregate(df, <no_column>, [min...])?
    function aggregations(df::DataFrame, aggregations::Array{String})
        dict = OrderedDict()
        exclude = [:longitude, :latitude, :time]
        columns_without_time = [k for k in names(df) if !(k in exclude)]
        # add the columns to the dataframe
        dict["columns"] = columns_without_time

        for function_name in aggregations
            func = getfield(Main, Symbol(function_name))
            dict[function_name] = colwise(func, df[columns_without_time])
        end

        DataFrame(dict)
    end

    df_agg = aggregations(df, ["minimum", "maximum", "mean", "std"])

    print_event("done_agg")

    # need to effectively transpose so it's the same as the other pipelines
    rename!(df_agg, Dict(:columns => :agg, :minimum => :min, :maximum => :max))
    res = OrderedDict()
    aggs = names(df_agg)
    res[aggs[1]] = aggs[2:5]
    for row in eachrow(df_agg)
        res[row[1]] = vec(convert(Array, row[2:5]))
    end
    CSV.write(parsed_args["output"] * "agg.csv", DataFrame(res))

    # 8. compute std per month
    # UDF 2: compute custom year+month format
    df[:year_month] = map(x -> Dates.year(x) * 100 + Dates.month(x), df[:time])

    # groupby cols on on and rename the final columns
    on = [:longitude, :latitude, :year_month]
    cols = [:tg, :tn, :tx, :pp, :rr]
    new_cols = [Symbol(string(k) * "_mean") for k in cols]
    grouped_df = by(df[[on; cols]], on, df -> DataFrame(OrderedDict(zip(new_cols, colwise(mean, df[cols])))))
    delete!(grouped_df, :year_month)

    print_event("done_groupby")

    CSV.write(parsed_args["output"] * "grouped.csv", grouped_df)

end

main()
