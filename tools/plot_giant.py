"""
Compare between different solver setups.
"""

import polars as pl
import hvplot
import hvplot.polars
from bokeh.models import NumeralTickFormatter, DatetimeTickFormatter
import numpy as np
import itertools
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

PROBCLASSES = [
    "netgen_8",
    "netgen_sr",
    "netgen_deg",
    "gridgen_8",
    "gridgen_sr",
    "gridgen_deg",
    "grid_long",
    "grid_square",
    "grid_wide",
]
valcols = [
    "iters",
    "time_s",
    "sddm_calls_per_iter",
    "time_s_per_arc_per_iter",
    "fact_s_per_arc_per_iter",
]
solvers = sorted(["tulip_approxchol", "tulip_cmg", "tulip_hypre"])

df = (
    pl.read_csv("../giant.csv")
    .filter(pl.col("solver").is_in(solvers))
    .filter(pl.col("probclass").is_in(PROBCLASSES))
    .filter(pl.col("iters") < 200)
    .drop_nulls("probclass")
    .with_columns(
        gsize=pl.col("vertices") + pl.col("arcs"),
        time_us_per_arc_per_iter=pl.col("time_s_per_arc_per_iter") * 1e6,
    )
)
odf = df

AXIS = "arcs"
METRIC = "time_us_per_arc_per_iter"
ARCS_MIN, ARCS_MAX = 1e0, 5e50
REG_ARCS_MIN, REG_ARCS_MAX = 1e3, 5e50


t = df[:, AXIS]
print(f"min_arcs = {t.min()} , max_arcs = {t.max()}")

df = df.filter(pl.col(AXIS) >= ARCS_MIN).filter(pl.col(AXIS) < ARCS_MAX)


def curve_0(x, c, b):
    return c * (np.log(x) ** b)


def init_0():
    return [1e-6, 3]


def kurve_0(th, x, y):
    return curve_0(x, *th) - y


def serve_0(th, x):
    return curve_0(x, *th)


df_reg = {}
for PROBCLASS, solver in itertools.product(PROBCLASSES, solvers):
    t = (
        odf.filter(pl.col("solver") == solver)
        .filter(pl.col("probclass") == PROBCLASS)
        .filter(pl.col(AXIS) >= REG_ARCS_MIN)
        .filter(pl.col(AXIS) < REG_ARCS_MAX)
        .drop_nulls(METRIC)
        .select([AXIS, METRIC])
        .group_by(AXIS)
        .agg(pl.col(METRIC).median())
        .sort([AXIS])
    )
    lsq = least_squares(
        kurve_0,
        init_0(),
        loss="linear",
        jac="3-point",
        # f_scale=0.1,
        args=(t[:, AXIS], t[:, METRIC]),
        # max_nfev=10000,
        # tr_solver="exact",
    )
    print("c * x^a * log(x)^b:", (PROBCLASS, solver), lsq.x)
    df_reg[(PROBCLASS, solver)] = lsq.x

plotz = []
for PROBCLASS in PROBCLASSES:
    all_markers = "^+d"[: len(solvers)]
    all_markers = dict(zip(solvers, all_markers))
    all_colours = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(solvers)]
    all_colours = dict(zip(solvers, all_colours))

    tdf = (
        df.filter(pl.col("probclass") == PROBCLASS)
        .with_columns(
            marker=pl.col("solver").replace(all_markers),
            color=pl.col("solver").replace(all_colours),
        )
        .with_columns(solver=pl.col("solver").str.strip_prefix("tulip_"))
    )

    ylim = (tdf[:, METRIC].min(), tdf[:, METRIC].max())
    ylim = (
        ylim[0] - (ylim[1] - ylim[0]) * 0.25,
        ylim[1] + (ylim[1] - ylim[0]) * 0.1,
    )
    p = tdf.hvplot.scatter(
        x=AXIS,
        y=METRIC,
        by="solver",
        marker="marker",
        color="color",
        height=400,
        width=400,
        title=PROBCLASS,
        yticks=10,
        xticks=10,
        ylim=ylim,
        rot=45,
        xformatter=NumeralTickFormatter(format="0a"),
        yformatter="%.0fus",
        grid=True,
    ).opts(
        legend_position="bottom_right",
        legend_cols=3,
    )

    x = np.linspace(max(tdf[:, AXIS].min(), REG_ARCS_MIN), tdf[:, AXIS].max(), num=100)
    for solver in solvers:
        if solver != "tulip_approxchol":
            continue
        y = serve_0(df_reg[(PROBCLASS, solver)], x)
        xy = pl.DataFrame({AXIS: x, "y": y})
        p = p * xy.hvplot.line(
            x=AXIS,
            y="y",
            line_width=1,
            color=all_colours[solver],
            line_dash="dotted",
            label=solver.removeprefix("tulip_"),
        )
    plotz.append(p)

p = plotz[0]
for pt in plotz[1:]:
    p = p + pt
p = p.cols(3).opts(shared_axes=False)
hvplot.show(p)
