# install.package("xyz")
#  R -f arg.R --args -i 42

# autoinstaller
# (function(lp) {
# np <- lp[!(lp %in% installed.packages()[,"Package"])]
# if(length(np)) install.packages(np,repos=c("http://cran.rstudio.com/"))
# x <- lapply(lp,function(x){library(x,character.only=TRUE)}) 
# })(c("argparser", "tidyr", "readr", "ncdf4", "dplyr"))

library("argparser")

p <- arg_parser("Yes.")
p <- add_argument(p, "-i", help="i")
p <- add_argument(p, "-s", help="s")
p <- add_argument(p, "-o", help="o")
p <- add_argument(p, "-c", help="c")

args <- parse_args(p, argv = commandArgs(trailingOnly = TRUE))


getvar <- function(f, n, scale=0.01) {
    v <- ncdf4::ncvar_get(f, n, raw_datavals=T) * scale
    attr(v, "dim") <- NULL
   v 
}

f1 <- ncdf4::nc_open(file.path(args$i, "data1.nc"))

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

f1_df <- f1_dim
f1_df$tg <- tg
f1_df$tg_stderr <- tg_stderr
f1_df$pp <- pp
f1_df$pp_stderr <- pp_stderr
f1_df$rr <- rr
f1_df$rr_stderr <- rr_stderr



f2 <- ncdf4::nc_open(file.path(args$i, "data2.nc"))

tn <- getvar(f2, "tn")
tn_stderr <- getvar(f2, "tn_stderr")
tx <- getvar(f2, "tx")
tx_stderr <- getvar(f2, "tx_stderr")


longitude <- getvar(f2, "longitude", 1)
latitude <- getvar(f2, "latitude", 1)
time <- getvar(f2, "time", 1)

tunits<-ncdf4::ncatt_get(f2,"time",attname="units")
tustr<-strsplit(tunits$value, " ")
time<-as.Date(time,origin=unlist(tustr)[3])

f2_dim <- tidyr::crossing(longitude, latitude, time)

ncdf4::nc_close(f2)


f2_df <- f2_dim
f2_df$tn <- tn
f2_df$tn_stderr <- tn_stderr
f2_df$tx <- tx
f2_df$tx_stderr <- tx_stderr


joined <- dplyr::inner_join(f1_df, f2_df)

slice_bounds <- strsplit(args$s, ":", fixed=T)[[1]]
slice <- joined[(slice_bounds[1]:slice_bounds[2]),]
filtered <- dplyr::filter(slice, tg != -99.99, pp != -999.9, rr != -999.9)


dropped <- dplyr::select(filtered, c("longitude", "latitude", "time", "tg", "tg_stderr", "pp", "rr", "tn", "tn_stderr", "tx", "tx_stderr"))

udf <- function(a, b) {
    abs(a-b)
}

dropped$abs_diff <- udf(dropped$tx, dropped$tn)


df_min <- dplyr::summarize_all(dplyr::select(dropped, c( "tg", "tg_stderr", "pp", "rr", "tn", "tn_stderr", "tx", "tx_stderr")), dplyr::funs(min)) 
df_min$op <- "min"
df_max <- dplyr::summarize_all(dplyr::select(dropped, c( "tg", "tg_stderr", "pp", "rr", "tn", "tn_stderr", "tx", "tx_stderr")), dplyr::funs(max)) 
df_max$op <- "max"
df_mean <- dplyr::summarize_all(dplyr::select(dropped, c( "tg", "tg_stderr", "pp", "rr", "tn", "tn_stderr", "tx", "tx_stderr")), dplyr::funs(mean)) 
df_mean$op <- "mean"
df_sd <- dplyr::summarize_all(dplyr::select(dropped, c( "tg", "tg_stderr", "pp", "rr", "tn", "tn_stderr", "tx", "tx_stderr")), dplyr::funs(sd)) 
df_sd$op <- "sd"


aggs <- dplyr::bind_rows(df_min, df_max, df_mean, df_sd)


udf2 <- function(a) {
    as.integer(format(a, "%Y"))*100+as.integer(format(a, "%m"))
}

dropped$year_month <- udf2(dropped$time)

df_grouped <- dplyr::rename(dplyr::summarize_all(dplyr::group_by(dplyr::select(dropped, c('latitude', 'longitude', 'year_month', 'tg', 'tn', 'tx', 'pp', 'rr')), latitude, longitude, year_month), mean), tg_mean=tg, tn_mean=tn, tx_mean=tx, pp_mean=pp, rr_mean=rr)

joined2 <- dplyr::inner_join(dropped, df_grouped)
joined2$year_month <- NULL


print(as.data.frame(head(joined2)))


readr::write_csv(joined2, file.path(args$o, "res1"))


