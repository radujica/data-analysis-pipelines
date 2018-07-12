# PIPELINE
p <- argparser::arg_parser("R Pipeline")
p <- argparser::add_argument(p, "--input", help="Path to folder containing input files")
p <- argparser::add_argument(p, "--slice", help="Start and stop of a subset of the data")
p <- argparser::add_argument(p, "--output", help="Path to output folder")

args <- argparser::parse_args(p, argv = commandArgs(trailingOnly = TRUE))


getvar <- function(f, n, scale) {
    v <- ncdf4::ncvar_get(f, n, raw_datavals=T) * scale
    attr(v, "dim") <- NULL
    v
}

f1 <- ncdf4::nc_open(file.path(args$input, "data1.nc"))
tg <- getvar(f1, "tg", 0.01)
tg_stderr <- getvar(f1, "tg_stderr", 0.01)
pp <- getvar(f1, "pp", 0.1)
pp_stderr <- getvar(f1, "pp_stderr", 0.1)
rr <- getvar(f1, "rr", 0.1)
rr_stderr <- getvar(f1, "rr_stderr", 0.1)

longitude <- getvar(f1, "longitude", 1)
latitude <- getvar(f1, "latitude", 1)
time <- getvar(f1, "time", 1)

tunits<-ncdf4::ncatt_get(f1,"time",attname="units")
tustr<-strsplit(tunits$value, " ")
time<-as.Date(time,origin=unlist(tustr)[3])

f1_dim <- tidyr::crossing(longitude, latitude, time)

ncdf4::nc_close(f1)

df1 <- f1_dim
df1$tg <- tg
df1$tg_stderr <- tg_stderr
df1$pp <- pp
df1$pp_stderr <- pp_stderr
df1$rr <- rr
df1$rr_stderr <- rr_stderr


f2 <- ncdf4::nc_open(file.path(args$input, "data2.nc"))

tn <- getvar(f2, "tn", 0.01)
tn_stderr <- getvar(f2, "tn_stderr", 0.01)
tx <- getvar(f2, "tx", 0.01)
tx_stderr <- getvar(f2, "tx_stderr", 0.01)

longitude <- getvar(f2, "longitude", 1)
latitude <- getvar(f2, "latitude", 1)
time <- getvar(f2, "time", 1)

tunits<-ncdf4::ncatt_get(f2,"time",attname="units")
tustr<-strsplit(tunits$value, " ")
time<-as.Date(time,origin=unlist(tustr)[3])

f2_dim <- tidyr::crossing(longitude, latitude, time)

ncdf4::nc_close(f2)

df2 <- f2_dim
df2$tn <- tn
df2$tn_stderr <- tn_stderr
df2$tx <- tx
df2$tx_stderr <- tx_stderr

print_event <- function(name) {
    cat(paste("#", format(Sys.time(), "%H:%M:%S"), "-", name, "\n", sep=""))
}

to_csv <- function(df, name) {
    readr::write_csv(df, paste(args$output, name, ".csv", sep = ""))
}


# PIPELINE
# 1. join the 2 dataframes
df <- dplyr::inner_join(df1, df2, by = c("longitude" = "longitude", "latitude" = "latitude", "time" = "time"))

# 2. quick preview on the data
df_head <- head(df, 10)

print_event('done_head')

to_csv(df_head, "head")

# 3. want a subset of the data, approx. 33%
slice_bounds <- strsplit(args$slice, ":", fixed=T)[[1]]
df <- df[(as.character(strtoi(slice_bounds[1]) + 1):strtoi(slice_bounds[2])),]

# 4. drop rows with null values
df <- dplyr::filter(df, !dplyr::near(tg, -99.99) & !dplyr::near(pp, -999.9) & !dplyr::near(rr, -999.9))

# 5. drop pp_err and rr_err columns
df <- dplyr::select(df, c("longitude", "latitude", "time", "tg", "tg_stderr", "tn", "tn_stderr", "tx", "tx_stderr", "pp", "rr"))

# 6. UDF 1: compute absolute difference between max and min
udf1 <- function(a, b) {
    abs(a-b)
}

df$abs_diff <- udf1(df$tx, df$tn)

# 7. explore the data through aggregations
columns_to_aggregate <- c("tg", "tn", "tx", "pp", "rr", "tg_stderr", "tx_stderr", "tn_stderr", "abs_diff")
df_min <- dplyr::summarize_all(dplyr::select(df, columns_to_aggregate), dplyr::funs(min))
df_min$agg <- "min"
df_max <- dplyr::summarize_all(dplyr::select(df, columns_to_aggregate), dplyr::funs(max))
df_max$agg <- "max"
df_mean <- dplyr::summarize_all(dplyr::select(df, columns_to_aggregate), dplyr::funs(mean))
df_mean$agg <- "mean"
df_sd <- dplyr::summarize_all(dplyr::select(df, columns_to_aggregate), dplyr::funs(sd))
df_sd$agg <- "std"

df_agg <- dplyr::bind_rows(df_min, df_max, df_mean, df_sd)

print_event('done_agg')

to_csv(df_agg, "agg")

# 8. compute mean per month
# UDF 2: compute custom year+month format
udf2 <- function(a) {
    as.integer(format(a, "%Y")) * 100 + as.integer(format(a, "%m"))
}

df$year_month <- udf2(df$time)

df_grouped <- as.data.frame(dplyr::rename(
        dplyr::summarize_all(
            dplyr::group_by(
                dplyr::select(
                    df, c('latitude', 'longitude', 'year_month', 'tg', 'tn', 'tx', 'pp', 'rr')
                ),
                latitude, longitude, year_month
            ),
            mean
        ),
        tg_mean=tg, tn_mean=tn, tx_mean=tx, pp_mean=pp, rr_mean=rr
    ))

df_grouped$year_month <- NULL
df_grouped$longitude <- NULL
df_grouped$latitude <- NULL
df_grouped <- dplyr::summarize_all(df_grouped, dplyr::funs(sum))
df_grouped_res <- data.frame(column=names(df_grouped), grouped_sum=c(df_grouped$tg_mean, df_grouped$tn_mean, df_grouped$tx_mean, df_grouped$pp_mean, df_grouped$rr_mean))

print_event('done_groupby')

to_csv(df_grouped_res, "grouped")
