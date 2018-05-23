using NetCDF
using DataFrames
using DataStructures
using ArgParse

include("netcdf_parser.jl")

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--path", "-p"
            help = "path to folder containing input files"
            arg_type = String
            required = true
    end

    return parse_args(s)
end


function main()
    parsed_args = parse_command_line()
    df1 = readnetcdf(parsed_args["path"] * "data1.nc")
    df2 = readnetcdf(parsed_args["path"] * "data2.nc")

    #= PIPELINE =#
    # 1. join the 2 dataframes
    df = join(df1, df2, on=[:longitude, :latitude, :time], kind=:inner)

    # 2. quick preview on the data
    println(DataFrames.head(df, 10))

    # 3. subset the data
    df = df[709921:1482480, :]

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

end

main()
