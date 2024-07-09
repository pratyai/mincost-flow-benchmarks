import glob
from pathlib import Path

root = Path("../lemon-runs")
runs = root.glob("*.min.run")

print("name,time_s")
for rf in runs:
    rf_text = rf.read_text()
    if "Feasible flow: found" not in rf_text:
        continue
    rt_line = [l for l in rf_text.split("\n") if "Run NetworkSimplex:" in l][0]
    rt = rt_line.split("real: ")[1].split("s")[0]
    rtime = float(rt)
    rname = str(rf).split("/")[-1].split(".min.run")[0]
    print(f"{rname},{rt}")
