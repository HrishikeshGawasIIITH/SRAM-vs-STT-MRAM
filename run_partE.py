#!/usr/bin/env python3
"""
AMCAS Assignment 1 - Part E (gem5) driver.

Runs the SRAM and STT-MRAM L2 configurations on two GAPBS kernels and writes
figures, tables and macros into report/.

Method: gem5's se.py with --fast-forward, so the setup phase and the first
trial are executed on an atomic CPU with the caches live, the detailed core
takes over exactly at the trial boundary, and gem5 resets the statistics at the
switch.  Every number therefore covers a warmed steady-state trial only.

L2 hit latencies come from Parts B and C:
  SRAM  2 MB : CACTI access time 2.90 ns -> 6 cycles @ 2 GHz, 11.47 mm^2
  MRAM  8 MB : NVSim  hit latency 3.21 ns -> 7 cycles @ 2 GHz, 10.32 mm^2
A third configuration re-runs the MRAM at 14 cycles, the "2x slower hit"
assumption, as a sensitivity check on that single input.

    python3 run_partE.py [--jobs 3]
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
SE = ROOT / "gem5" / "configs" / "deprecated" / "example" / "se.py"
WORK = ROOT / "work" / "e18"
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
    "sram":   dict(l2_size="2MB", lat=6,  label=r"SRAM 2\,MB, 6\,cyc"),
    "mram":   dict(l2_size="8MB", lat=7,  label=r"MRAM 8\,MB, 7\,cyc"),
    "mram14": dict(l2_size="8MB", lat=14, label=r"MRAM 8\,MB, 14\,cyc"),
}
# scale-18 Kronecker graph, 262143 nodes / 3.8 M undirected edges
KERNELS = {
    "bfs":  "gapbs/graph18.sg",
    "sssp": "gapbs/graph18.wsg",
}


def env():
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = (str(Path.home() / ".local" / "lib") + ":"
                            + e.get("LD_LIBRARY_PATH", ""))
    return e


def stat(txt: str, *names):
    for n in names:
        m = re.search(rf"^{re.escape(n)}\s+([\d.eE+-]+)", txt, re.M)
        if m:
            return float(m.group(1))
    return None


def warmup_insts(kern: str) -> int:
    """Instructions for setup + trial 1, measured atomically with -n 1."""
    f = WORK / f"n1_{kern}" / "stats.txt"
    if not f.exists():
        sys.exit(f"missing {f}; run the -n 1 atomic measurement first")
    v = stat(f.read_text(), "simInsts")
    if not v:
        sys.exit(f"no simInsts in {f}")
    return int(v)


def parse(d: Path) -> dict:
    f = d / "stats.txt"
    if not f.exists():
        return {}
    t = f.read_text()
    # with --fast-forward the detailed core is switch_cpus
    ipc = stat(t, "system.switch_cpus.ipc", "system.cpu.ipc",
               "system.switch_cpus.ipc::0")
    cyc = stat(t, "system.switch_cpus.numCycles", "system.cpu.numCycles")
    hits = stat(t, "system.l2.overallHits::total",
                "system.l2cache.overallHits::total") or 0.0
    miss = stat(t, "system.l2.overallMisses::total",
                "system.l2cache.overallMisses::total") or 0.0
    mr = stat(t, "system.l2.overallMissRate::total",
              "system.l2cache.overallMissRate::total")
    if mr is None and (hits + miss):
        mr = miss / (hits + miss)
    return {"ipc": ipc, "cycles": cyc,
            "simSeconds": stat(t, "simSeconds"),
            "simInsts": stat(t, "simInsts"),
            "l2_hits": hits, "l2_misses": miss, "l2_miss_rate": mr,
            "l2_accesses": hits + miss}


def run_one(job):
    key, kern, cpu, cfg = job
    tag = f"{kern}_{key}_{cpu}"
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    if (d / "stats.txt").exists() and (d / "stats.txt").stat().st_size > 2000:
        print(f"  [cached] {tag}")
        return tag, parse(d)

    cpu_type = "X86O3CPU" if cpu == "O3" else "X86TimingSimpleCPU"
    cmd = [str(GEM5), "-d", str(d), str(SE),
           f"--cpu-type={cpu_type}", "--caches", "--l2cache",
           "--l1d_size=32kB", "--l1i_size=32kB",
           f"--l2_size={cfg['l2_size']}", "--l2_assoc=8",
           f"--l2-hit-latency={cfg['lat']}",
           "--mem-type=DDR4_2400_8x8", "--mem-size=4GB",
           f"--fast-forward={warmup_insts(kern)}",
           f"--cmd=gapbs/{kern}",
           f"--options=-f {KERNELS[kern]} -n 2"]
    print(f"  running {tag} (warmup {warmup_insts(kern):,} insts) ...")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env())
    (d / "run.log").write_text(p.stdout + p.stderr)
    r = parse(d)
    if not r.get("ipc"):
        print(f"  !! {tag}: no IPC, see {d/'run.log'}")
    return tag, r


def fmt(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if not GEM5.exists():
        sys.exit(f"gem5 not built at {GEM5}")
    for p in (WORK, IMAGES, DATA):
        p.mkdir(parents=True, exist_ok=True)

    jobs = [(k, kern, "O3", CONFIGS[k]) for kern in KERNELS for k in CONFIGS]
    jobs += [(k, kern, "Timing", CONFIGS[k])
             for kern in KERNELS for k in ("sram", "mram")]
    print(f"gem5: {len(jobs)} runs, {a.jobs} at a time, scale-18 graphs, "
          f"atomic warmup then switch at the trial boundary")

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
                f"\\texttt{{{kern}}} & {CONFIGS[k]['label']} & {r['ipc']:.4f} & "
                f"{r['l2_miss_rate']:.4f} & {r['simSeconds']:.6f} \\\\")
    (DATA / "table_partE.tex").write_text(
        r"""\begin{tabular}{llrrr}
\toprule
Kernel & L2 configuration & IPC & L2 miss rate & simSeconds \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
""")

    rows = []
    for kern in KERNELS:
        for k in ("sram", "mram"):
            r = res.get(f"{kern}_{k}_Timing", {})
            if not r.get("ipc"):
                continue
            rows.append(
                f"\\texttt{{{kern}}} & {CONFIGS[k]['label']} & {r['ipc']:.4f} & "
                f"{r['l2_miss_rate']:.4f} & {r['simSeconds']:.6f} \\\\")
    (DATA / "table_partE_inorder.tex").write_text(
        r"""\begin{tabular}{llrrr}
\toprule
Kernel & L2 configuration & IPC & L2 miss rate & simSeconds \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
""")

    # ---------------- figure ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    xs = np.arange(len(KERNELS))
    w = 0.26
    for i, (k, col) in enumerate(zip(CONFIGS, [BLUE, GREEN, RED])):
        ipcs = [res.get(f"{kn}_{k}_O3", {}).get("ipc") or 0 for kn in KERNELS]
        mrs = [100 * (res.get(f"{kn}_{k}_O3", {}).get("l2_miss_rate") or 0)
               for kn in KERNELS]
        lbl = CONFIGS[k]["label"].replace("\\,", " ")
        ax1.bar(xs + (i - 1) * w, ipcs, w, label=lbl, color=col)
        ax2.bar(xs + (i - 1) * w, mrs, w, label=lbl, color=col)
    for ax, yl in ((ax1, "IPC"), (ax2, "L2 miss rate (%)")):
        ax.set_xticks(xs)
        ax.set_xticklabels(list(KERNELS))
        ax.set_ylabel(yl)
    ax1.legend(fontsize=7, loc="upper left")
    fig.suptitle("Part E: iso-area SRAM vs STT-MRAM L2, out-of-order core",
                 fontsize=9.5)
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
        for k, tagname in (("sram", "Sram"), ("mram", "Mram"),
                           ("mram14", "MramFour")):
            r = res.get(f"{kern}_{k}_O3", {})
            if r.get("ipc"):
                m += [mac(f"E{K}{tagname}Ipc", f"{r['ipc']:.4f}"),
                      mac(f"E{K}{tagname}Miss", f"{r['l2_miss_rate']:.4f}"),
                      mac(f"E{K}{tagname}MissPct", f"{100*r['l2_miss_rate']:.1f}"),
                      mac(f"E{K}{tagname}Ms", f"{1e3*r['simSeconds']:.2f}")]
        rs, rm = res.get(f"{kern}_sram_O3", {}), res.get(f"{kern}_mram_O3", {})
        if rs.get("ipc") and rm.get("ipc"):
            m += [mac(f"E{K}Speedup", f"{rm['ipc']/rs['ipc']:.3f}"),
                  mac(f"E{K}SpeedupPct", f"{100*(rm['ipc']/rs['ipc']-1):.1f}"),
                  mac(f"E{K}MissFactor", f"{rs['l2_miss_rate']/rm['l2_miss_rate']:.1f}")]
        r14 = res.get(f"{kern}_mram14_O3", {})
        if rs.get("ipc") and r14.get("ipc"):
            m.append(mac(f"E{K}SpeedupFourteen", f"{r14['ipc']/rs['ipc']:.3f}"))
        ts, tm = res.get(f"{kern}_sram_Timing", {}), res.get(f"{kern}_mram_Timing", {})
        if ts.get("ipc") and tm.get("ipc"):
            m += [mac(f"E{K}TimingSpeedup", f"{tm['ipc']/ts['ipc']:.3f}"),
                  mac(f"E{K}TimingSramIpc", f"{ts['ipc']:.4f}"),
                  mac(f"E{K}TimingMramIpc", f"{tm['ipc']:.4f}")]
            if rs.get("cycles") and rm.get("cycles") and rs.get("l2_misses"):
                for cpu, nm, aa, bb in (("O3", "Ooo", rs, rm),
                                        ("Timing", "Inorder", ts, tm)):
                    dm = aa.get("l2_misses", 0) - bb.get("l2_misses", 0)
                    dc = (aa.get("cycles") or 0) - (bb.get("cycles") or 0)
                    if dm:
                        m.append(mac(f"E{K}{nm}PerMiss", f"{dc/dm:.0f}"))
        m.append(mac(f"E{K}Warmup", f"{warmup_insts(kern)/1e6:.1f}"))
    m.append(mac("EGraph", "18"))
    m.append(mac("EGraphNodes", "262\\,143"))
    m.append(mac("EGraphEdges", "3.8"))
    (DATA / "macros_partE.tex").write_text("\n".join(m) + "\n")

    print("\nPart E results, scale-18, warmed steady-state trial")
    for kern in KERNELS:
        for k in CONFIGS:
            r = res.get(f"{kern}_{k}_O3", {})
            if r.get("ipc"):
                print(f"  {kern:5} {k:7} O3     IPC {r['ipc']:.4f} | "
                      f"L2 miss {r['l2_miss_rate']:.4f} | "
                      f"{1e3*r['simSeconds']:.2f} ms")
        for k in ("sram", "mram"):
            r = res.get(f"{kern}_{k}_Timing", {})
            if r.get("ipc"):
                print(f"  {kern:5} {k:7} Timing IPC {r['ipc']:.4f} | "
                      f"L2 miss {r['l2_miss_rate']:.4f}")
        rs, rm = res.get(f"{kern}_sram_O3", {}), res.get(f"{kern}_mram_O3", {})
        if rs.get("ipc") and rm.get("ipc"):
            print(f"  -> {kern} MRAM/SRAM speedup {rm['ipc']/rs['ipc']:.3f}x")


if __name__ == "__main__":
    main()
