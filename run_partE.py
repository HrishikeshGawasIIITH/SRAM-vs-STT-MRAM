#!/usr/bin/env python3
"""
AMCAS Assignment 1 - Part E (gem5) driver.

Runs the SRAM and STT-MRAM L2 configurations on two GAPBS kernels, parses
m5out/stats.txt, and writes figures/tables/macros into report/.

L2 hit latencies come from Parts B and C, not from the assignment text:
  SRAM  2 MB : CACTI access time 2.90 ns -> 6 cycles @ 2 GHz
  MRAM  8 MB : NVSim  hit latency 3.21 ns -> 7 cycles @ 2 GHz
8 MB is the iso-area capacity: NVSim puts 8 MB of STT-MRAM in 10.32 mm2 against
CACTI's 10.66 mm2 for 2 MB of SRAM.  A third configuration re-runs the MRAM at
14 cycles - the assignment's "2x slower hits" assumption - as a sensitivity
check, because our measured penalty is far smaller than that.

    python3 run_partE.py [--maxinsts 2e8] [--graph 18] [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
GEM5 = ROOT / "gem5" / "build" / "X86" / "gem5.opt"
CFG = ROOT / "gem5_l2.py"
WORK = ROOT / "work" / "e"
IMAGES = ROOT / "report" / "images"
DATA = ROOT / "report" / "data"
PREVIEW = ROOT / "work" / "preview"

plt.rcParams.update({
    "figure.figsize": (5.4, 3.2), "figure.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight",
})
BLUE, RED, GREY, GREEN = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"

CONFIGS = {
    "sram":    dict(l2_size="2MB", lat=6,  label="SRAM 2\\,MB, 6\\,cyc"),
    "mram":    dict(l2_size="8MB", lat=7,  label="MRAM 8\\,MB, 7\\,cyc"),
    "mram14":  dict(l2_size="8MB", lat=14, label="MRAM 8\\,MB, 14\\,cyc"),
}
# Pre-built graph files.  Generating the Kronecker graph inside gem5 burns
# hundreds of millions of instructions in a compute-bound phase with a tiny
# working set, which is NOT what we want to measure; loading a prepared graph
# puts the simulated region in the traversal itself.
KERNEL_OPTS = {
    "bfs":  "-f gapbs/graph16.sg -n 1",
    "sssp": "-f gapbs/graph16.wsg -n 1",
}
KERNELS = list(KERNEL_OPTS)


def env():
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = (str(Path.home() / ".local" / "lib") + ":"
                            + e.get("LD_LIBRARY_PATH", ""))
    return e


def stat(txt: str, name: str):
    m = re.search(rf"^{re.escape(name)}\s+([\d.eE+-]+)", txt, re.M)
    return float(m.group(1)) if m else None


def parse(d: Path) -> dict:
    f = d / "stats.txt"
    if not f.exists():
        return {}
    t = f.read_text()
    hits = stat(t, "system.l2cache.overallHits::total") or 0.0
    misses = stat(t, "system.l2cache.overallMisses::total") or 0.0
    acc = hits + misses
    mr = stat(t, "system.l2cache.overallMissRate::total")
    if mr is None and acc:
        mr = misses / acc
    return {
        "simSeconds": stat(t, "simSeconds"),
        "simInsts": stat(t, "simInsts"),
        "ipc": stat(t, "system.cpu.ipc") or stat(t, "system.cpu.ipc::0"),
        "cpi": stat(t, "system.cpu.cpi") or stat(t, "system.cpu.cpi::0"),
        "l2_hits": hits, "l2_misses": misses,
        "l2_miss_rate": mr,
        "l2_accesses": acc,
        "dram_reads": stat(t, "system.mem_ctrl.dram.bytesRead::total"),
        "cycles": stat(t, "system.cpu.numCycles"),
    }


def run_one(job) -> tuple:
    key, kern, cpu, cfg, maxinsts = job
    tag = f"{kern}_{key}_{cpu}"
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    if (d / "stats.txt").exists() and (d / "stats.txt").stat().st_size > 1000:
        print(f"  [cached] {tag}")
        return tag, parse(d)
    cmd = [str(GEM5), "-d", str(d), str(CFG),
           "--cpu-type", cpu,
           "--l2_size", cfg["l2_size"], "--l2_assoc", "8",
           "--l2-hit-latency", str(cfg["lat"]),
           "--cmd", str(ROOT / "gapbs" / kern),
           "--options", KERNEL_OPTS[kern],
           "--maxinsts", str(int(maxinsts))]
    print(f"  running {tag} ...")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env())
    (d / "run.log").write_text(p.stdout + p.stderr)
    r = parse(d)
    if not r.get("ipc"):
        print(f"  !! {tag} produced no IPC; see {d/'run.log'}")
    return tag, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxinsts", type=float, default=0,
                    help="0 = run each kernel to completion")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    if not GEM5.exists():
        sys.exit(f"gem5 not built at {GEM5}")
    WORK.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    jobs = [(k, kern, "O3", CONFIGS[k], a.maxinsts)
            for kern in KERNELS for k in CONFIGS]
    # Task 3: the winning pair re-run in-order
    jobs += [(k, kern, "Timing", CONFIGS[k], a.maxinsts)
             for kern in KERNELS for k in ("sram", "mram")]

    print(f"gem5: {len(jobs)} runs, {a.jobs} at a time "
          f"(maxinsts={a.maxinsts or 'run to completion'})")
    res = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for tag, r in ex.map(run_one, jobs):
            res[tag] = r

    (DATA / "partE_summary.json").write_text(json.dumps(res, indent=2))

    # ---------------- table ----------------
    rows = []
    for kern in KERNELS:
        for k in CONFIGS:
            r = res.get(f"{kern}_{k}_O3", {})
            if not r.get("ipc"):
                continue
            rows.append(
                f"{kern} & {CONFIGS[k]['label']} & {r['ipc']:.3f} & "
                f"{100*r['l2_miss_rate']:.1f}\\,\\% & "
                f"{1e3*r['simSeconds']:.2f} \\\\")
    (DATA / "table_partE.tex").write_text(
        r"""\begin{tabular}{llrrr}
\toprule
Kernel & L2 configuration & IPC & L2 miss rate & simTime (ms) \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
""")

    # ---------------- figure ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    width = 0.26
    xs = np.arange(len(KERNELS))
    for i, (k, col) in enumerate(zip(CONFIGS, [BLUE, GREEN, RED])):
        ipcs = [res.get(f"{kern}_{k}_O3", {}).get("ipc") or 0 for kern in KERNELS]
        mrs = [100 * (res.get(f"{kern}_{k}_O3", {}).get("l2_miss_rate") or 0)
               for kern in KERNELS]
        lbl = CONFIGS[k]["label"].replace("\\,", " ")
        ax1.bar(xs + (i - 1) * width, ipcs, width, label=lbl, color=col)
        ax2.bar(xs + (i - 1) * width, mrs, width, label=lbl, color=col)
    for ax, ylab in ((ax1, "IPC"), (ax2, "L2 miss rate (%)")):
        ax.set_xticks(xs)
        ax.set_xticklabels(KERNELS)
        ax.set_ylabel(ylab)
    ax1.legend(fontsize=7, loc="upper left")
    fig.suptitle("Part E: iso-area SRAM vs STT-MRAM L2 (O3 core)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "E_ipc_missrate.pdf")
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW / "E_ipc_missrate.png")
    plt.close(fig)

    # ---------------- macros ----------------
    def mac(n, v):
        return r"\newcommand{\%s}{%s}" % (n, v)

    m = []
    for kern in KERNELS:
        K = kern.upper()
        for k in CONFIGS:
            r = res.get(f"{kern}_{k}_O3", {})
            if not r.get("ipc"):
                continue
            tagname = {"sram": "Sram", "mram": "Mram", "mram14": "MramFour"}[k]
            m += [mac(f"E{K}{tagname}Ipc", f"{r['ipc']:.3f}"),
                  mac(f"E{K}{tagname}Miss", f"{100*r['l2_miss_rate']:.1f}"),
                  mac(f"E{K}{tagname}Ms", f"{1e3*r['simSeconds']:.2f}")]
        rs, rm = res.get(f"{kern}_sram_O3", {}), res.get(f"{kern}_mram_O3", {})
        if rs.get("ipc") and rm.get("ipc"):
            m.append(mac(f"E{K}Speedup", f"{rm['ipc']/rs['ipc']:.3f}"))
            m.append(mac(f"E{K}MissDrop",
                         f"{100*(1-(rm['l2_miss_rate'] or 0)/(rs['l2_miss_rate'] or 1)):.0f}"))
        r14 = res.get(f"{kern}_mram14_O3", {})
        if rs.get("ipc") and r14.get("ipc"):
            m.append(mac(f"E{K}SpeedupFourteen", f"{r14['ipc']/rs['ipc']:.3f}"))
        ts, tm = res.get(f"{kern}_sram_Timing", {}), res.get(f"{kern}_mram_Timing", {})
        if ts.get("ipc") and tm.get("ipc"):
            m.append(mac(f"E{K}TimingSpeedup", f"{tm['ipc']/ts['ipc']:.3f}"))
            m.append(mac(f"E{K}TimingSramIpc", f"{ts['ipc']:.3f}"))
            m.append(mac(f"E{K}TimingMramIpc", f"{tm['ipc']:.3f}"))
        # cycles saved per avoided L2 miss: how much of a miss the core hides
        for cpu, nm in (("O3", "Ooo"), ("Timing", "Inorder")):
            a = res.get(f"{kern}_sram_{cpu}", {})
            b = res.get(f"{kern}_mram_{cpu}", {})
            if a.get("cycles") and b.get("cycles") and a.get("l2_misses"):
                dm = a["l2_misses"] - b["l2_misses"]
                dc = a["cycles"] - b["cycles"]
                if dm:
                    m.append(mac(f"E{K}{nm}PerMiss", f"{dc/dm:.0f}"))
                m.append(mac(f"E{K}{nm}MissCut",
                             f"{100*dm/a['l2_misses']:.0f}"))
    m.append(mac("EGraph", "16"))
    (DATA / "macros_partE.tex").write_text("\n".join(m) + "\n")

    print("\nPart E results")
    for kern in KERNELS:
        for k in CONFIGS:
            r = res.get(f"{kern}_{k}_O3", {})
            if r.get("ipc"):
                print(f"  {kern:5} {k:7} O3     IPC {r['ipc']:.3f} | "
                      f"L2 miss {100*r['l2_miss_rate']:5.1f} % | "
                      f"{1e3*r['simSeconds']:.2f} ms")
        for k in ("sram", "mram"):
            r = res.get(f"{kern}_{k}_Timing", {})
            if r.get("ipc"):
                print(f"  {kern:5} {k:7} Timing IPC {r['ipc']:.3f} | "
                      f"L2 miss {100*r['l2_miss_rate']:5.1f} %")


if __name__ == "__main__":
    main()
