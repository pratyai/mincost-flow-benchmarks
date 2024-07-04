import polars as pl
import sys

valcols = [
    "iters",
    "sddm_calls_per_iter",
    "time_s_per_arc_per_iter",
    "fact_s_per_arc_per_iter",
]
solvers = ["tulip_approxchol", "tulip_cmg", "tulip_hypre"]

df = (
    pl.read_csv("../giant.csv")
    .filter(pl.col("solver").is_in(solvers))
    .filter(pl.col("iters") < 200)
    .filter(pl.col("baseline"))
)

reltabs = []
for vc in valcols:
    dfp = df.pivot(index="name", columns="solver", values=vc)
    # dfpbest = dfp.with_columns(best = pl.min_horizontal(*solvers))
    dfpbest = dfp.with_columns(best=pl.col("tulip_approxchol"))
    for s in solvers:
        dfpbest = dfpbest.with_columns((pl.col(s) / pl.col("best")).alias(s))
    dfpbest = dfpbest.drop("best").melt(
        id_vars=["name"], variable_name="solver", value_name=vc
    )
    reltabs.append(dfpbest)

reldf = df[:, ["name", "solver", "probclass"]]
for t in reltabs:
    reldf = reldf.join(t, on=["name", "solver"])

reltabs = []
for vc in valcols:
    t = reldf.group_by(["probclass", "solver"]).agg(
        pl.col(vc).min().round(2).alias("min " + vc[:5]),
        pl.col(vc).max().round(2).alias("max " + vc[:5]),
        # pl.col(vc).median().round(2).alias('median ' + vc[:5]),
        # pl.col(vc).mean().round(2).alias('mean ' + vc[:5]),
    )
    reltabs.append(t)

reldf = reltabs[0]
for t in reltabs[1:]:
    reldf = reldf.join(t, on=["probclass", "solver"])
reldf = reldf.sort("probclass", "solver").filter(pl.col("solver") != "tulip_approxchol")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)

# print(reldf)
reldf.write_csv(sys.stdout)
