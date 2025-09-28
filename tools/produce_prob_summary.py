# Generates a LaTeX table summarizing the benchmark problems from .inspec files.

import polars as pl

def format_num(n):
    if n is None:
        return ""
    if n >= 1e6:
        return f"{round(n/1e6)}M"
    if n >= 1e3:
        return f"{round(n/1e3)}k"
    return f"{int(n)}"

def format_ratio(n):
    if n is None:
        return ""
    return f"{n:.1f}"

# Define the mapping from probclass to a list of .inspec files.
prob_files = {
    "netgen_8": ["netgen_8_40.inspec", "netgen_8_big.inspec"],
    "netgen_sr": ["netgen_sr_30.inspec", "netgen_sr_big.inspec"],
    "netgen_deg": ["netgen_deg_40.inspec", "netgen_deg_big.inspec"],
    "gridgen_8": ["gridgen_8.inspec"],
    "gridgen_sr": ["gridgen_sr.inspec"],
    "gridgen_deg": ["gridgen_deg.inspec"],
    "grid_long": ["gridgraph_long.inspec"],
    "grid_square": ["gridgraph_square.inspec"],
    "grid_wide": ["gridgraph_wide.inspec"],
    "goto_8": ["goto_8_40.inspec", "goto_8_big.inspec"],
    "goto_sr": ["goto_sr_30.inspec", "goto_sr_big.inspec"],
    "road_path": ["road_path.inspec"],
    "road_flow": ["road_flow.inspec"],
    "vision_inv": ["vision_inv.inspec"],
    "vision_prop": ["vision_prop.inspec"],
    "vision_rnd": ["vision_rnd.inspec"],
    "spielman": ["spielman.inspec"],
}

summary_data = []

for probclass, files in prob_files.items():
    df_list = []
    for f in files:
        try:
            df_list.append(pl.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        continue

    df = pl.concat(df_list)

    # Calculate the number of instances
    instances = df.height

    # Calculate summary statistics
    summary = df.select(
        pl.col("vertices").min().alias("min_v"),
        pl.col("vertices").max().alias("max_v"),
        pl.col("arcs").min().alias("min_e"),
        pl.col("arcs").max().alias("max_e"),
        (pl.col("arcs") / pl.col("vertices")).min().alias("min_e_v"),
        (pl.col("arcs") / pl.col("vertices")).max().alias("max_e_v"),
    ).to_dicts()[0]

    summary["probclass"] = probclass
    summary["instances"] = instances
    summary_data.append(summary)

summary_df = pl.DataFrame(summary_data)

# Define the structure of the table
groups = {
    "NETGEN": ["netgen_8", "netgen_sr", "netgen_deg"],
    "GRIDGEN": ["gridgen_8", "gridgen_sr", "gridgen_deg"],
    "GRID": ["grid_long", "grid_square", "grid_wide"],
    "GOTO": ["goto_8", "goto_sr"],
    "ROAD": ["road_path", "road_flow"],
    "VISION": ["vision_inv", "vision_prop", "vision_rnd"],
    "Spielman": ["spielman"],
}

# LaTeX table header
print(
    r"""\begin{tabular}{@{}ll c cc cc cc@{}}
\toprule
\multirow{2}{*}{Class} & & \multirow{2}{*}{\#Instances} & \multicolumn{2}{c}{$\abs{V}$} & \multicolumn{2}{c}{$\abs{E}$} & \multicolumn{2}{c}{$\abs{E} / \abs{V}$} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& & & min & max & min & max & min & max \\
\midrule"""
)

# Generate table rows
for main_class, subclasses in groups.items():
    if main_class == "Spielman":
        row_data = summary_df.filter(pl.col("probclass") == "spielman")
        if not row_data.is_empty():
            row = row_data.row(0, named=True)
            print(
                f"\\multicolumn{{2}}{{c}}{{Spielman}} & {row['instances']} & "
                f"{format_num(row['min_v'])} & {format_num(row['max_v'])} & "
                f"{format_num(row['min_e'])} & {format_num(row['max_e'])} & "
                f"{format_ratio(row['min_e_v'])} & {format_ratio(row['max_e_v'])} \\"
            )
        continue

    print(f"\\multirow{{{len(subclasses)}}}{{*}}{{{main_class}}}", end="")
    for i, subclass_prob in enumerate(subclasses):
        row_data = summary_df.filter(pl.col("probclass") == subclass_prob)
        if not row_data.is_empty():
            row = row_data.row(0, named=True)
            sub_class_name = subclass_prob.split("_", 1)[1].upper().replace("_", " ")
            if i > 0:
                print("&", end="")
            print(
                f" & {sub_class_name} & {row['instances']} & "
                f"{format_num(row['min_v'])} & {format_num(row['max_v'])} & "
                f"{format_num(row['min_e'])} & {format_num(row['max_e'])} & "
                f"{format_ratio(row['min_e_v'])} & {format_ratio(row['max_e_v'])} \\"
            )
    print(r"\\midrule")

# LaTeX table footer
print(r"""\bottomrule
\end{tabular}"""
)