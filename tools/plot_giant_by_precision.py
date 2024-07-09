"""
Compare between different solver setups.
"""

import polars as pl
import holoviews as hv
import hvplot
import hvplot.polars
from bokeh.models import NumeralTickFormatter, DatetimeTickFormatter
import numpy as np
import itertools
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

hv.extension("bokeh")

PROBCLASSES = [
    "road_path",
    # "netgen_8",
    "grid_long",
    "goto_8",
    # "grid_square",
    "road_flow",
    # "netgen_sr",
    "grid_wide",
    "goto_sr",
    # "spielman",
    # "netgen_deg",
]
valcols = [
    "iters",
    "time_s",
    "sddm_calls_per_iter",
    "time_s_per_arc_per_iter",
    "fact_s_per_arc_per_iter",
]
params = [
    "5e-8,0.0001,1e-8,64",  # typical default
    # "1e-11,0.0001,1e-8,64",
    "1e-11,1e-6,1e-10,64",  # other typical default
    # "1e-16,1e-6,1e-10,128",
    # "5e-8,0.0001,1e-8,128",
    "1e-16,1e-8,1e-12,128",  # ideally the best
]
pick_params = {
    "grid_long": [params[0], params[2]],
    "grid_square": [params[0], params[2]],
    "grid_wide": [params[0], params[2]],
    "goto_8": [params[1], params[2]],
    "goto_sr": [params[1], params[2]],
    "spielman": [params[1], params[2]],
    "netgen_8": [params[0], params[2]],
    "netgen_sr": [params[0], params[2]],
    "netgen_deg": [params[0], params[2]],
    "road_path": [params[1], params[2]],
    "road_flow": [params[0], params[2]],
}
label_params = {
    "5e-8,0.0001,1e-8,64": "baseline #1",
    "1e-11,1e-6,1e-10,64": "baseline #2",
    "1e-16,1e-8,1e-12,128": "high-accuracy",
}

df = (
    pl.read_csv(
        "../giant.csv",
        schema_overrides={
            "epcg": pl.Float64,
            "rhop": pl.Float64,
            "rhod": pl.Float64,
            "floatbits": pl.Int16,
        },
    )
    .filter(pl.col("solver") == "tulip_approxchol")
    .filter(pl.col("probclass").is_in(PROBCLASSES))
    # .filter(pl.col("iters") < 200)
    .drop_nulls("probclass")
    .with_columns(
        gsize=pl.col("vertices") + pl.col("arcs"),
        time_us_per_arc_per_iter=pl.col("time_s_per_arc_per_iter") * 1e6,
        params=pl.concat_str(
            pl.col(["epcg", "rhop", "rhod", "floatbits"]), separator=","
        ),
    )
)
print(df[:, "params"].unique().sort())
df = df.filter(pl.col("params").is_in(params))
odf = df

AXIS = "arcs"
METRIC = "iters"
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
for PROBCLASS, pset in itertools.product(PROBCLASSES, params):
    t = (
        odf.filter(pl.col("probclass") == PROBCLASS)
        .filter(pl.col("params") == pset)
        .filter(pl.col(AXIS) >= REG_ARCS_MIN)
        .filter(pl.col(AXIS) < REG_ARCS_MAX)
        .filter(pl.col("iters") < 200)
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
    print("c * x^a * log(x)^b:", (PROBCLASS, pset), lsq.x)
    df_reg[(PROBCLASS, pset)] = lsq.x

plotz = []
all_markers = ["circle", "diamond", "triangle", "square", "inverted_triangle", "hex"][
    : len(params)
]
all_markers = dict(zip(params, all_markers))
all_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][
    : len(params)
]
all_colours = dict(zip(params, all_colours))
for PROBCLASS in PROBCLASSES:
    p = [hv.HLine(200).opts(color="red", line_width=0.5)]
    tdf = df.filter(pl.col("probclass") == PROBCLASS).drop_nulls(METRIC)
    ylim = (tdf[:, METRIC].min(), tdf[:, METRIC].max())
    ylim = (
        ylim[0] - (ylim[1] - ylim[0]) * 0.4,
        ylim[1] + (ylim[1] - ylim[0]) * 0.1,
    )
    for pset in pick_params[PROBCLASS]:
        ttdf = tdf.filter(pl.col("params") == pset)
        if ttdf.is_empty():
            continue

        p.append(
            ttdf.with_columns(params=pl.col("params").replace(label_params))
            .hvplot.scatter(
                x=AXIS,
                y=METRIC,
                by="params",
                marker=all_markers[pset],
                color=all_colours[pset],
                fill_alpha=0.5,
                height=400,
                width=400,
                title=PROBCLASS,
                ylim=ylim,
                rot=45,
                xformatter=NumeralTickFormatter(format="0a"),
                grid=True,
                logx=True,
            )
            .opts(
                legend_position="bottom_right",
                legend_cols=2,
            )
        )

        x = np.linspace(
            max(ttdf[:, AXIS].min(), REG_ARCS_MIN), ttdf[:, AXIS].max(), num=100
        )
        y = serve_0(df_reg[(PROBCLASS, pset)], x)
        xy = pl.DataFrame({AXIS: x, "y": y})
        p.append(
            xy.hvplot.line(
                x=AXIS,
                y="y",
                line_width=1,
                color=all_colours[pset],
                line_dash="dotted",
                label=label_params[pset],
            )
        )
    plotz.append(hv.Overlay(p).opts(xlabel=AXIS, ylabel=METRIC))

p = plotz[0]
for pt in plotz[1:]:
    p = p + pt
p = p.cols(3).opts(shared_axes=False)
hvplot.show(p)
