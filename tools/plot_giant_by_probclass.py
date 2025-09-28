"""
Compare between different problem classes within the same Tulip+ApproxChol setup.
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

def format_param(p):
    if abs(p) < 0.01:
        return f"{p:.2e}"
    return f"{p:.2f}"

valcols = [
    "iters",
    "time_s",
    "sddm_calls_per_iter",
    "time_s_per_arc_per_iter",
    "fact_s_per_arc_per_iter",
]
solvers = ["tulip_approxchol", "tulip_cmg", "tulip_hypre"]
problems_1 = sorted(["netgen_8", "netgen_sr", "netgen_deg"])
problems_2 = sorted(["gridgen_8", "gridgen_sr", "gridgen_deg"])
problems_3 = sorted(["grid_long", "grid_square", "grid_wide"])
problems_4 = sorted(["goto_8", "goto_sr"])
problems_5 = sorted(["road_flow", "road_path"])
# problem_groups = [problems_1, problems_2]
problem_groups = {"netgen": problems_1, "grid graph": problems_3}
# problem_groups = [problems_1, problems_2 , problems_3, problems_4]
# problem_groups = [problems_5]
problems = sorted(list(itertools.chain(*problem_groups.values())))

df = (
    pl.read_csv("../giant.csv", infer_schema_length=1000)
    .filter(pl.col("solver").is_in(solvers))
    .filter(pl.col("probclass").is_in(problems))
    .filter(pl.col("baseline"))
    # .filter(pl.col("iters") < 200)
    .drop_nulls("probclass")
    .with_columns(
        gsize=pl.col("vertices") + pl.col("arcs"),
        time_us_per_arc_per_iter=pl.col("time_s_per_arc_per_iter") * 1e6,
    )
)
odf = df

SOLVER = "tulip_approxchol"
AXIS = "arcs"
METRICS = ["iters", "time_us_per_arc_per_iter"]
ARCS_MIN, ARCS_MAX = 1e0, 5e50
REG_ARCS_MIN, REG_ARCS_MAX = 1e3, 5e50


t = df.filter(pl.col("solver") == SOLVER)[:, AXIS]
print(f"min_arcs = {t.min()} , max_arcs = {t.max()}")

df = (
    df.filter(pl.col("solver") == SOLVER)
    .filter(pl.col(AXIS) >= ARCS_MIN)
    .filter(pl.col(AXIS) < ARCS_MAX)
)


def curve_0(x, c, b):
    return c * (np.log(x) ** b)


def init_0():
    return [1e-6, 3]


def kurve_0(th, x, y):
    return curve_0(x, *th) - y


def serve_0(th, x):
    return curve_0(x, *th)


def curve_1(x, c, b):
    return c * x * (np.log(x) ** b)


def init_1():
    return [1e-6, 3]


def kurve_1(th, x, y):
    return curve_1(x, *th) - y


def serve_1(th, x):
    return curve_1(x, *th)


def curve(x, c, a, b):
    return c * (x**a) * (np.log(x) ** b)


def init():
    return [1e-6, 0.5, 0]


def kurve(th, x, y):
    return curve(x, *th) - y


def serve(th, x):
    return curve(x, *th)


def curve_power(x, c, a):
    return c * (x**a)


def init_power():
    return [1.0, 0.5]


def kurve_power(th, x, y):
    return curve_power(x, *th) - y


def serve_power(th, x):
    return curve_power(x, *th)



df_reg = {}
metric_kurves = {
    "iters": kurve_0,
    "time_us_per_arc_per_iter": kurve_0,
    "time_s": kurve_1,
}
metric_serves = {
    "iters": serve_0,
    "time_us_per_arc_per_iter": serve_0,
    "time_s": serve_1,
}
metric_inits = {
    "iters": init_0,
    "time_us_per_arc_per_iter": init_0,
    "time_s": init_1,
}

FIT_POWER_LAW_FOR_ITERS = True

for METRIC, pcls in itertools.product(METRICS, problems):
    t = (
        odf.filter(pl.col("solver") == SOLVER)
        .filter(pl.col("probclass") == pcls)
        .filter(pl.col(AXIS) >= REG_ARCS_MIN)
        .filter(pl.col(AXIS) < REG_ARCS_MAX)
        .filter(pl.col("iters") < 200)
        .drop_nulls(METRIC)
        .select([AXIS, METRIC])
        .group_by(AXIS)
        .agg(pl.col(METRIC).median())
        .sort([AXIS])
    )

    if FIT_POWER_LAW_FOR_ITERS and METRIC == "iters":
        kurve_func = kurve_power
        init_func = init_power
    else:
        kurve_func = metric_kurves[METRIC]
        init_func = metric_inits[METRIC]

    lsq = least_squares(
        kurve_func,
        init_func(),
        loss="linear",
        # f_scale=0.1,
        args=(t[:, AXIS], t[:, METRIC]),
        # max_nfev=10000,
        # tr_solver="exact",
    )
    print("c * x^a * log(x)^b:", (METRIC, pcls), lsq.x)
    df_reg[(METRIC, pcls)] = lsq.x


plotz = []
yformatter = {
    "iters": "%d",
    "time_us_per_arc_per_iter": "%.0fus",
    "time_s": "%.0fs",
}
for METRIC in METRICS:
    for key, probs in problem_groups.items():
        all_markers = [
            "circle",
            "diamond",
            "triangle",
            "square",
            "inverted_triangle",
            "hex",
        ][: len(probs)]
        all_markers = dict(zip(probs, all_markers))
        all_colours = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ][: len(probs)]
        all_colours = dict(zip(probs, all_colours))

        tdf = df.filter(pl.col("probclass").is_in(probs)).with_columns(
            marker=pl.col("probclass").replace(all_markers),
            color=pl.col("probclass").replace(all_colours),
        )
        tdf_wo_fails = tdf.filter(pl.col("iters") < 200)
        xlim = (tdf_wo_fails[:, AXIS].min(), tdf_wo_fails[:, AXIS].max())
        xgap, xmid = np.sqrt(xlim[1] / xlim[0]), np.sqrt(xlim[0] * xlim[1])
        xlim = (xmid / (xgap * 2), xmid * (xgap * 4))
        tdf = tdf.filter(pl.col(AXIS) <= xlim[1])

        p_elements = []
        if METRIC == "iters":
            p_elements.append(hv.HLine(200).opts(color="red", line_width=0.5))

        ylim = (tdf[:, METRIC].min(), tdf[:, METRIC].max())
        if METRIC == "iters":
            ylim = (
                ylim[0] * 0.8,
                ylim[1] * 1.2,
            )
        else:
            ylim = (
                ylim[0] - (ylim[1] - ylim[0]) * 0.25,
                ylim[1] + (ylim[1] - ylim[0]) * 0.1,
            )

        scatter_plot = tdf.hvplot.scatter(
            x=AXIS,
            y=METRIC,
            by="probclass",
            marker="marker",
            color="color",
            fill_alpha=0.5,
            height=400,
            width=400,
            title=key,
            yticks=10,
            # xticks=10,
            ylim=ylim,
            xlim=xlim,
            rot=45,
            xformatter=NumeralTickFormatter(format="0a"),
            yformatter=yformatter[METRIC],
            grid=True,
            logx=True,
            logy=(METRIC == "iters"),
        ).opts(
            legend_position="bottom_right",
            legend_cols=3,
        )

        lines = []
        texts = []
        x = np.geomspace(tdf[:, AXIS].min(), tdf[:, AXIS].max(), num=100)
        for i, pcls in enumerate(probs):
            params = df_reg[(METRIC, pcls)]
            if FIT_POWER_LAW_FOR_ITERS and METRIC == "iters":
                serve_func = serve_power
                label_text = f"{format_param(params[0])} * arcs^{params[1]:.2f}"
            else:
                serve_func = metric_serves[METRIC]
                if METRIC == "time_s":
                    label_text = (
                        f"{format_param(params[0])} * arcs * log(arcs)^{params[1]:.2f}"
                    )
                else:
                    label_text = f"{format_param(params[0])} * log(arcs)^{params[1]:.2f}"

            y = serve_func(params, x)
            xy = pl.DataFrame({AXIS: x, "y": y})
            lines.append(
                xy.hvplot.line(
                    x=AXIS,
                    y="y",
                    line_width=1,
                    color=all_colours[pcls],
                    line_dash="dotted",
                    label=pcls,
                )
            )

            text_x = xmid
            if METRIC == "iters":  # log scale
                text_y = ylim[1] / (1.1**i)
            else:  # linear scale
                text_y = ylim[1] - i * (ylim[1] - ylim[0]) * 0.05

            texts.append(
                hv.Text(text_x, text_y, label_text, halign="center", valign="top").opts(
                    color=all_colours[pcls],
                    text_font_size="8pt",
                    bgcolor="white",
                )
            )

        p = scatter_plot * hv.Overlay(lines) * hv.Overlay(texts)
        plotz.append(p.opts(xlabel=AXIS, ylabel=METRIC))

p = plotz[0]
for pt in plotz[1:]:
    p = p + pt
p = p.cols(2).opts(shared_axes=False)
hvplot.show(p)
