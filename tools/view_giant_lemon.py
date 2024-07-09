import polars as pl
import sys

basedf = (
    pl.read_csv("../giant.csv")
    .filter(pl.col("solver") == "tulip_approxchol")
    .filter(pl.col("baseline"))
    .with_columns(base_ts=pl.col("time_s"))
    .select(["name", "base_ts"])
)
nsdf = (
    pl.read_csv("../giant.csv")
    .filter(pl.col("solver") == "network_simplex")
    .with_columns(ns_ts=pl.col("time_s"))
    .select(["name", "ns_ts", "probclass"])
)
# nsdf.group_by(["probclass"]).len().sort(by=["probclass"]).write_csv(sys.stdout)
# exit()

t = nsdf.join(basedf, on="name", how="left", coalesce=True)
t = t.with_columns(ts_ratio=pl.col("ns_ts") / pl.col("base_ts"))
t = (
    t.group_by(["probclass"])
    .agg(
        # pl.col("ts_ratio").min().round(10).alias("min ts"),
        pl.col("ts_ratio").median().round(10).alias("med ts"),
        pl.col("ts_ratio").max().round(10).alias("max ts"),
    )
    .sort(by=["probclass"])
)
# print(t)
t.write_csv(sys.stdout)
