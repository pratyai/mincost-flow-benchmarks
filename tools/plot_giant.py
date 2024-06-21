import polars as pl
import hvplot
import hvplot.polars
import numpy as np
from scipy.optimize import curve_fit

valcols = [
    "iters",
    "time_s",
    "sddm_calls_per_iter",
    "time_s_per_arc_per_iter",
    "fact_s_per_arc_per_iter",
]
solvers = ["tulip_approxchol", "tulip_cmg", "tulip_hypre"]

df = (
    pl.read_csv("../giant.csv")
    .filter(pl.col("solver").is_in(solvers))
    .filter(pl.col("iters") < 200)
    .drop_nulls("labels")
    .with_columns(gsize=pl.col("vertices") + pl.col("arcs"))
)
odf = df

PROBCLASS = "goto_sr"
AXIS = "arcs"
METRIC = "time_s"
ARCS_MIN, ARCS_MAX = 1e0, 5e50
REG_ARCS_MIN = 1e3


t = df.filter(pl.col("labels") == PROBCLASS)[:, AXIS]
print(f"min_arcs = {t.min()} , max_arcs = {t.max()}")

df = (
    df.filter(pl.col("labels") == PROBCLASS)
    .filter(pl.col(AXIS) >= ARCS_MIN)
    .filter(pl.col(AXIS) < ARCS_MAX)
)


def curve(x, c, a):
    return c * (x**a)


df_reg = {}
for solver in df[:, "solver"].unique():
    t = (
        odf.filter(pl.col("solver") == solver)
        .filter(pl.col("labels") == PROBCLASS)
        .filter(pl.col(AXIS) >= REG_ARCS_MIN)
        .drop_nulls(METRIC)
    )
    popt, pcov = curve_fit(
        curve,
        t[:, AXIS],
        t[:, METRIC],
        p0=(1e-6, 1.5),
        bounds=([0, 0], [np.inf, np.inf]),
        # nan_policy="omit",
    )
    df_reg[solver] = popt
    print(solver, popt)

all_markers = "^+d"[: len(df["solver"].unique())]
all_markers = dict(zip(sorted(df["solver"].unique()), all_markers))
all_colours = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(df["solver"].unique())]
all_colours = dict(zip(sorted(df["solver"].unique()), all_colours))

df = df.with_columns(
    marker=pl.col("solver").replace(all_markers),
    color=pl.col("solver").replace(all_colours),
)

p = df.hvplot.scatter(
    x=AXIS,
    y=METRIC,
    by="solver",
    marker="marker",
    color="color",
    height=400,
    width=400,
    legend="top_left",
    yticks=10,
)

x = np.linspace(df[:, AXIS].min(), df[:, AXIS].max(), num=100)
for solver in df[:, "solver"].unique():
    y = curve(x, *df_reg[solver])
    xy = pl.DataFrame({AXIS: x, "y": y})
    p = p * xy.hvplot.line(
        x=AXIS,
        y="y",
        line_width=1,
        color=all_colours[solver],
        yticks=10,
    )

hvplot.show(p)
