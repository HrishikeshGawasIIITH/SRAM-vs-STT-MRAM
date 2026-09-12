#!/usr/bin/env python3
"""
AMCAS Assignment 1 - Part D (Ramulator 2.0) driver.

Runs the L2 miss trace through DDR4 under four configurations, parses the stats
dump, and writes figures/tables/macros into report/.

    python3 run_partD.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAM = ROOT / "ramulator2.0" / "ramulator2"
PARTD = ROOT / "partD"
WORK = ROOT / "work" / "d"
IMAGES = ROOT / "report" / "images"
DATA = ROOT / "report" / "data"
PREVIEW = ROOT / "work" / "preview"

# DDR4_3200AA, from ramulator2.0/src/dram/impl/DDR4.cpp
TIMING = {"tRCD": 22, "tRP": 22, "tRAS": 52, "tCCD_S": 4, "tCCD_L": 8,
          "tCL": 22, "tCK_ps": 625}

plt.rcParams.update({
    "figure.figsize": (5.4, 3.2), "figure.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight",
})
BLUE, RED, GREY, GREEN, ORANGE = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c", "#ff7f0e"


def save(fig, out: Path):
    fig.savefig(out)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW / (out.stem + ".png"))
    plt.close(fig)


def sum_stat(text: str, base: str) -> float:
    """Sum a per-channel stat (row_hits_0, row_hits_1, ...)."""
    return float(sum(float(v) for v in
                     re.findall(rf"^\s*{base}_\d+:\s*([\d.eE+-]+)", text, re.M)))


def one(text: str, name: str):
    m = re.search(rf"^\s*{name}:\s*([\d.eE+-]+)", text, re.M)
    return float(m.group(1)) if m else None


def read_cmd_counts(path: Path) -> dict[str, float]:
    """CommandCounter writes '<CMD>, <count>' lines to its configured path."""
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text().splitlines():
        if "," in ln:
            k, v = ln.split(",", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out


def run(tag: str, cfg_text: str) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    # give each run its own command-count file
    cmd_path = WORK / f"{tag}_cmds.txt"
    if cmd_path.exists():
        cmd_path.unlink()
    cfg_text = re.sub(r"(path: ).*cmdcount\.txt", rf"\g<1>{cmd_path}", cfg_text)
    cfg = WORK / f"{tag}.yaml"
    cfg.write_text(cfg_text)
    proc = subprocess.run([str(RAM), "--config_file", str(cfg)],
                          cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    (WORK / f"{tag}.out").write_text(out)
    if "memory_system_cycles" not in out:
        print(out[-2000:])
        sys.exit(f"Ramulator run '{tag}' failed")

    reads = one(out, "total_num_read_requests") or 0.0
    writes = one(out, "total_num_write_requests") or 0.0
    hits = sum_stat(out, "row_hits")
    misses = sum_stat(out, "row_misses")
    conflicts = sum_stat(out, "row_conflicts")
    total_rb = hits + misses + conflicts
    rl = sum_stat(out, "read_latency")
    cmds = read_cmd_counts(cmd_path)

    return {
        "tag": tag,
        "cycles": one(out, "memory_system_cycles"),
        "reads": reads, "writes": writes,
        "row_hits": hits, "row_misses": misses, "row_conflicts": conflicts,
        "rb_hit_rate": hits / total_rb if total_rb else 0.0,
        "read_latency_total": rl,
        # NOTE: Ramulator's own avg_read_latency divides by num_read_reqs_N,
        # which send() increments even when the request is REJECTED (queue
        # full) and the frontend retries.  That inflates the denominator and
        # reports averages below tRCD+tCL, which is physically impossible.
        # Divide by the authoritative MemorySystem read count instead.
        "avg_read_cyc": rl / reads if reads else 0.0,
        "acts": cmds.get("ACT", 0.0),
        "pres": cmds.get("PRE", 0.0) + cmds.get("PREA", 0.0),
        "rds": cmds.get("RD", 0.0) + cmds.get("RDA", 0.0),
        "wrs": cmds.get("WR", 0.0) + cmds.get("WRA", 0.0),
    }


def bank_parallelism() -> dict:
    """Replay the trace through both address mappings and count how many of the
    32 banks (2 ranks x 4 bank groups x 4 banks) each one actually reaches."""
    addrs = []
    for ln in (PARTD / "l2miss.trace").read_text().splitlines():
        f = ln.split()
        if len(f) == 2:
            addrs.append(int(f[1], 16) >> 6)      # drop the 64 B burst offset
    nCh, nRa, nBg, nBa, nCo = 1, 2, 4, 4, 1024

    def peel(a, widths):
        out = []
        for w in widths:
            out.append(a % w)
            a //= w
        return out

    out = {}
    for name, widths in (("RoBaRaCoCh", [nCh, nCo, nRa, nBg, nBa]),
                         ("ChRaBaRoCo", [nCo, 65536, nBa, nBg, nRa])):
        ids, per = set(), []
        for a in addrs:
            f = peel(a, widths)
            bid = (f[2], f[3], f[4])
            ids.add(bid)
            per.append(bid)
        win, tot, n = 64, 0, 0
        for i in range(0, max(len(per) - win, 1), win):
            tot += len(set(per[i:i + win]))
            n += 1
        from collections import Counter
        busiest = Counter(per).most_common(1)[0][1] if per else 0
        out[name] = {"banks_used": len(ids), "banks_total": 32,
                     "mean_per_window": tot / max(n, 1),
                     "busiest_share_pct": 100 * busiest / max(len(per), 1)}
    return out


def cfg_for(scheduler="FRFCFS", mapper="RoBaRaCoCh", channels=1) -> str:
    base = (PARTD / "ddr4.yaml").read_text()
    base = re.sub(r"(Scheduler:\n\s+impl: )\w+", rf"\g<1>{scheduler}", base)
    base = re.sub(r"(AddrMapper:\n\s+impl: )\w+", rf"\g<1>{mapper}", base)
    base = re.sub(r"(channel: )\d+", rf"\g<1>{channels}", base)
    return base


def fmt(x, n=2):
    return "--" if x is None else f"{x:.{n}f}"


def main():
    if not RAM.exists():
        sys.exit(f"ramulator2 not built at {RAM}")
    if not (PARTD / "l2miss.trace").exists():
        sys.exit("partD/l2miss.trace missing - run gen_l2trace.py first")
    IMAGES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print("Ramulator: baseline FRFCFS / RoBaRaCoCh / 1 channel ...")
    t1 = run("t1_frfcfs", cfg_for())
    print("Ramulator: FCFS ...")
    t2 = run("t2_fcfs", cfg_for(scheduler="FCFS"))
    print("Ramulator: ChRaBaRoCo (row bits below bank bits) ...")
    t3 = run("t3_chrabaroco", cfg_for(mapper="ChRaBaRoCo"))
    print("Ramulator: 2 channels ...")
    t4 = run("t4_2ch", cfg_for(channels=2))

    # Task 4 extended: where does adding channels stop helping?  Answering
    # "which timing parameter binds" by experiment beats asserting it.
    print("Ramulator: channel sweep ...")
    chan_sweep = []
    for ch in (1, 2, 4, 8):
        r = run(f"t4_ch{ch}", cfg_for(channels=ch))
        r["channels"] = ch
        chan_sweep.append(r)

    ns = TIMING["tCK_ps"] / 1000.0   # ns per memory cycle

    # ---------------- figures ----------------
    runs = [("FR-FCFS\n(baseline)", t1), ("FCFS", t2),
            ("ChRaBaRoCo\nmapping", t3), ("2 channels", t4)]
    labels = [r[0] for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    hr = [100 * r[1]["rb_hit_rate"] for r in runs]
    ax1.bar(labels, hr, color=[BLUE, RED, ORANGE, GREEN], width=0.6)
    for i, v in enumerate(hr):
        ax1.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax1.set_ylabel("row-buffer hit rate (%)")
    ax1.set_ylim(0, max(hr) * 1.3)
    ax1.tick_params(axis="x", labelsize=7.5)

    lat = [r[1]["avg_read_cyc"] for r in runs]
    ax2.bar(labels, lat, color=[BLUE, RED, ORANGE, GREEN], width=0.6)
    for i, v in enumerate(lat):
        ax2.text(i, v + max(lat) * 0.03, f"{v:.0f}", ha="center", fontsize=8)
    ax2.set_ylabel("avg read latency (mem cycles)")
    ax2.set_ylim(0, max(lat) * 1.25)
    ax2.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Part D: scheduler, address mapping and channel count",
                 fontsize=9.5)
    fig.tight_layout()
    save(fig, IMAGES / "D_summary.pdf")

    # row-state breakdown
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    bottom = np.zeros(len(runs))
    for key, lab, col in (("row_hits", "hits", GREEN),
                          ("row_misses", "misses (tRCD)", ORANGE),
                          ("row_conflicts", "conflicts (tRP+tRCD)", RED)):
        vals = np.array([100 * r[1][key] /
                         (r[1]["row_hits"] + r[1]["row_misses"] + r[1]["row_conflicts"])
                         for r in runs])
        ax.barh(labels, vals, left=bottom, color=col, label=lab, height=0.6)
        bottom += vals
    ax.set_xlabel("share of DRAM accesses (%)")
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Where each access lands in the row buffer", fontsize=9)
    ax.grid(False)
    save(fig, IMAGES / "D_rowstate.pdf")

    # channel scaling
    ch = np.array([r["channels"] for r in chan_sweep])
    cyc = np.array([r["cycles"] for r in chan_sweep])
    lat = np.array([r["avg_read_cyc"] for r in chan_sweep])
    fig, ax = plt.subplots()
    ax.plot(ch, cyc[0] / cyc, "o-", ms=4, color=BLUE, label="throughput speedup")
    ax.plot(ch, ch, ls="--", lw=1.0, color=GREY, label="ideal (linear)")
    ax2 = ax.twinx()
    ax2.plot(ch, lat, "s-", ms=4, color=RED, label="avg read latency")
    ax2.set_ylabel("avg read latency (cycles)", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.grid(False)
    ax.set_xscale("log", base=2); ax.set_xticks(ch)
    ax.set_xticklabels([str(c) for c in ch])
    ax.set_xlabel("DDR4 channels")
    ax.set_ylabel("speedup vs 1 channel")
    ax.set_title("Task D4: channel scaling", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, IMAGES / "D_channels.pdf")

    bp = bank_parallelism()

    brows = "\n".join(
        f"\\texttt{{{k}}} & {v['banks_used']} / {v['banks_total']} & "
        f"{v['mean_per_window']:.2f} & {v['busiest_share_pct']:.1f}\\,\\% \\\\"
        for k, v in bp.items())
    (DATA / "table_partD_banks.tex").write_text(
        r"""\begin{tabular}{lccc}
\toprule
Address mapping & banks reached & mean banks per & busiest bank's \\
 & (of 32) & 64-access window & share of traffic \\
\midrule
""" + brows + "\n" + r"""\bottomrule
\end{tabular}
""")

    ns_ = TIMING["tCK_ps"] / 1000.0
    trows = []
    for lab, r, ch in (("Baseline, FR-FCFS", t1, 1), ("FCFS", t2, 1),
                       ("ChRaBaRoCo", t3, 1), ("Two channels", t4, 2)):
        thr = (r["reads"] + r["writes"]) / r["cycles"] / ch
        trows.append(f"{lab} & {thr:.4f} & {100*thr/(1/TIMING['tCCD_L']):.1f}\\,\\% "
                     f"& {100*thr/(1/TIMING['tCCD_S']):.1f}\\,\\% \\\\")
    (DATA / "table_partD_ccd.tex").write_text(
        r"""\begin{tabular}{lccc}
\toprule
Configuration & accesses per cycle & vs $1/t_{CCD\_L}$ & vs $1/t_{CCD\_S}$ \\
 & per channel & (0.125) & (0.250) \\
\midrule
""" + "\n".join(trows) + "\n" + r"""\bottomrule
\end{tabular}
""")

    # ---------------- table ----------------
    rows = []
    for lab, r in [("FR-FCFS, RoBaRaCoCh, 1 ch (baseline)", t1),
                   ("FCFS, RoBaRaCoCh, 1 ch", t2),
                   ("FR-FCFS, ChRaBaRoCo, 1 ch", t3),
                   ("FR-FCFS, RoBaRaCoCh, 2 ch", t4)]:
        rows.append(
            f"{lab} & {r['cycles']:.0f} & {r['avg_read_cyc']:.0f} & "
            f"{r['avg_read_cyc']*ns:.0f} & {100*r['rb_hit_rate']:.1f}\\,\\% & "
            f"{r['acts']:.0f} \\\\")
    (DATA / "table_partD.tex").write_text(
        r"""\begin{tabular}{lrrrrr}
\toprule
Configuration & Memory & Avg read & & Row-buffer & ACT \\
 & cycles & (cycles) & (ns) & hit rate & commands \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
""")

    def mac(n, v):
        return r"\newcommand{\%s}{%s}" % (n, v)

    macros = [
        mac("Dcycles", f"{t1['cycles']:.0f}"),
        mac("Dreads", f"{t1['reads']:.0f}"),
        mac("Dwrites", f"{t1['writes']:.0f}"),
        mac("Dlat", f"{t1['avg_read_cyc']:.0f}"),
        mac("DlatNs", f"{t1['avg_read_cyc']*ns:.0f}"),
        mac("Dhit", f"{100*t1['rb_hit_rate']:.1f}"),
        mac("DfcfsHit", f"{100*t2['rb_hit_rate']:.1f}"),
        mac("DfcfsLat", f"{t2['avg_read_cyc']:.0f}"),
        mac("DfcfsLatNs", f"{t2['avg_read_cyc']*ns:.0f}"),
        mac("DfcfsCost", f"{t2['avg_read_cyc']-t1['avg_read_cyc']:.0f}"),
        mac("DfcfsRatio", fmt(t2['avg_read_cyc']/t1['avg_read_cyc'], 2)),
        mac("DfcfsCycles", f"{t2['cycles']:.0f}"),
        mac("DfcfsSlow", fmt(t2['cycles']/t1['cycles'], 2)),
        mac("DmapHit", f"{100*t3['rb_hit_rate']:.1f}"),
        mac("DmapLat", f"{t3['avg_read_cyc']:.0f}"),
        mac("DmapCycles", f"{t3['cycles']:.0f}"),
        mac("DmapSlow", fmt(t3['cycles']/t1['cycles'], 2)),
        mac("DmapConf", f"{100*t3['row_conflicts']/(t3['row_hits']+t3['row_misses']+t3['row_conflicts']):.1f}"),
        mac("DbaseConf", f"{100*t1['row_conflicts']/(t1['row_hits']+t1['row_misses']+t1['row_conflicts']):.1f}"),
        mac("DtwoLat", f"{t4['avg_read_cyc']:.0f}"),
        mac("DtwoCycles", f"{t4['cycles']:.0f}"),
        mac("DtwoSpeedup", fmt(t1['cycles']/t4['cycles'], 2)),
        mac("DtwoLatGain", fmt(100*(1-t4['avg_read_cyc']/t1['avg_read_cyc']), 0)),
        mac("DtRCD", str(TIMING["tRCD"])),
        mac("DtRP", str(TIMING["tRP"])),
        mac("DtRAS", str(TIMING["tRAS"])),
        mac("DtCCD", str(TIMING["tCCD_L"])),
        mac("DtCL", str(TIMING["tCL"])),
        mac("DtCK", f"{TIMING['tCK_ps']/1000:.3f}"),
        mac("DconflictCost", str(TIMING["tRP"] + TIMING["tRCD"])),
        mac("DbaseActs", f"{t1['acts']:.0f}"),
        mac("DfcfsActs", f"{t2['acts']:.0f}"),
        mac("DmapActs", f"{t3['acts']:.0f}"),
        mac("DchFour", fmt(chan_sweep[0]["cycles"]/chan_sweep[2]["cycles"], 2)),
        mac("DchEight", fmt(chan_sweep[0]["cycles"]/chan_sweep[3]["cycles"], 2)),
        mac("DchFourLat", f"{chan_sweep[2]['avg_read_cyc']:.0f}"),
        mac("DchEightLat", f"{chan_sweep[3]['avg_read_cyc']:.0f}"),
        mac("DbanksBase", str(bp["RoBaRaCoCh"]["banks_used"])),
        mac("DbanksMap", str(bp["ChRaBaRoCo"]["banks_used"])),
        mac("DwinBase", f"{bp['RoBaRaCoCh']['mean_per_window']:.2f}"),
        mac("DwinMap", f"{bp['ChRaBaRoCo']['mean_per_window']:.2f}"),
        mac("DbusyMap", f"{bp['ChRaBaRoCo']['busiest_share_pct']:.0f}"),
        mac("DthrBase", f"{(t1['reads']+t1['writes'])/t1['cycles']:.4f}"),
        mac("DactCeil", f"{32/(TIMING['tRAS']+TIMING['tRP']):.4f}"),
        mac("DactRate", f"{t1['acts']/t1['cycles']:.4f}"),
        mac("DactUtil", f"{100*(t1['acts']/t1['cycles'])/(32/(TIMING['tRAS']+TIMING['tRP'])):.1f}"),
    ]
    (DATA / "macros_partD.tex").write_text("\n".join(macros) + "\n")
    (DATA / "partD_summary.json").write_text(
        json.dumps({"t1": t1, "t2": t2, "t3": t3, "t4": t4,
                    "channel_sweep": chan_sweep, "timing": TIMING}, indent=2))

    print(f"""
Part D  (Ramulator 2.0, DDR4-3200AA, {t1['reads']:.0f} R + {t1['writes']:.0f} W)
  T1 baseline FR-FCFS : {t1['cycles']:.0f} cycles | """
          f"""avg read {t1['avg_read_cyc']:.0f} cyc ({t1['avg_read_cyc']*ns:.0f} ns) | """
          f"""RB hit {100*t1['rb_hit_rate']:.1f} %
  T2 FCFS             : {t2['cycles']:.0f} cycles | """
          f"""avg read {t2['avg_read_cyc']:.0f} cyc | RB hit {100*t2['rb_hit_rate']:.1f} % | """
          f"""ACTs {t1['acts']:.0f} -> {t2['acts']:.0f}
  T3 ChRaBaRoCo       : {t3['cycles']:.0f} cycles | """
          f"""avg read {t3['avg_read_cyc']:.0f} cyc | RB hit {100*t3['rb_hit_rate']:.1f} %
  T4 2 channels       : {t4['cycles']:.0f} cycles | """
          f"""avg read {t4['avg_read_cyc']:.0f} cyc | """
          f"""speedup {t1['cycles']/t4['cycles']:.2f}x
  T4 channel sweep    : """
          + "  ".join(f"{int(r['channels'])}ch={t1['cycles']/r['cycles']:.2f}x"
                      f"/{r['avg_read_cyc']:.0f}cyc" for r in chan_sweep) + "\n")


if __name__ == "__main__":
    main()
