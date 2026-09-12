#!/usr/bin/env python3
"""
AMCAS Assignment 1 - Part B (CACTI) and Part C (NVSim) driver.

Generates every config variant, runs the tools, parses their output, and writes
figures into report/images/ and tables + macros into report/data/.

    python3 run_partBC.py
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
CACTI_DIR = ROOT / "cacti"
NVSIM_DIR = ROOT / "NVSim"
PARTB = ROOT / "partB"
PARTC = ROOT / "partC"
WORK = ROOT / "work" / "bc"
IMAGES = ROOT / "report" / "images"
DATA = ROOT / "report" / "data"
PREVIEW = ROOT / "work" / "preview"

plt.rcParams.update({
    "figure.figsize": (5.4, 3.4), "figure.dpi": 160, "font.size": 9,
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


# ==========================================================================
# CACTI
# ==========================================================================
def cacti_cfg(base: str, overrides: dict[str, str]) -> str:
    """Apply overrides to a CACTI cfg.

    CACTI configs list alternatives as commented-out siblings, so for each key
    we comment out every currently-active line and re-emit one active line in
    the position of the first match.
    """
    lines = base.splitlines()
    for prefix, newline in overrides.items():
        placed = False
        for i, ln in enumerate(lines):
            if ln.startswith(prefix):
                if not placed:
                    lines[i] = newline
                    placed = True
                else:
                    lines[i] = "//" + ln
        if not placed:
            lines.append(newline)
    return "\n".join(lines) + "\n"


CACTI_PATTERNS = {
    "access_ns":     r"Access time \(ns\):\s*([\d.eE+-]+)",
    "cycle_ns":      r"Cycle time \(ns\):\s*([\d.eE+-]+)",
    "read_nJ":       r"Total dynamic read energy per access \(nJ\):\s*([\d.eE+-]+)",
    "write_nJ":      r"Total dynamic write energy per access \(nJ\):\s*([\d.eE+-]+)",
    "leak_mW":       r"Total leakage power of a bank \(mW\):\s*([\d.eE+-]+)",
    "ndwl":          r"Best Ndwl\s*:\s*(\d+)",
    "ndbl":          r"Best Ndbl\s*:\s*(\d+)",
    "nspd":          r"Best Nspd\s*:\s*(\d+)",
    "ntwl":          r"Best Ntwl\s*:\s*(\d+)",
    "ntbl":          r"Best Ntbl\s*:\s*(\d+)",
    "ntspd":         r"Best Ntspd\s*:\s*(\d+)",
    "bitline_ns":    r"Bitline delay \(ns\):\s*([\d.eE+-]+)",
    "decoder_ns":    r"Decoder \+ wordline delay \(ns\):\s*([\d.eE+-]+)",
    "htree_in_ns":   r"H-tree input delay \(ns\):\s*([\d.eE+-]+)",
    "htree_out_ns":  r"H-tree output delay \(ns\):\s*([\d.eE+-]+)",
    "senseamp_ns":   r"Sense Amplifier delay \(ns\):\s*([\d.eE+-]+)",
    "data_area_mm2": r"Data array: Area \(mm2\):\s*([\d.eE+-]+)",
    "tag_area_mm2":  r"Tag array: Area \(mm2\):\s*([\d.eE+-]+)",
    "area_eff_pct":  r"Area efficiency \(Memory cell area/Total area\) -\s*([\d.eE+-]+)",
    "subarray_h_mm": r"Subarray Height \(mm\):\s*([\d.eE+-]+)",
    "subarray_l_mm": r"Subarray Length \(mm\):\s*([\d.eE+-]+)",
    "mat_h_mm":      r"MAT Height \(mm\):\s*([\d.eE+-]+)",
    "nbanks":        r"Number of banks:\s*(\d+)",
    "h_mm":          r"Cache height x width \(mm\):\s*([\d.eE+-]+)",
    "w_mm":          r"Cache height x width \(mm\):\s*[\d.eE+-]+\s*x\s*([\d.eE+-]+)",
}


def parse_cacti(out: str) -> dict:
    """Parse a CACTI run.  Several keys appear once for the data array and again
    for the tag array; the first hit is always the data array."""
    r: dict = {}
    for k, pat in CACTI_PATTERNS.items():
        m = re.search(pat, out)
        r[k] = float(m.group(1)) if m else None
    if r["data_area_mm2"] is not None and r["tag_area_mm2"] is not None:
        r["area_mm2"] = r["data_area_mm2"] + r["tag_area_mm2"]
    r["read_pJ"] = r["read_nJ"] * 1e3 if r["read_nJ"] else None
    r["write_pJ"] = r["write_nJ"] * 1e3 if r["write_nJ"] else None
    # CACTI reports leakage PER BANK; NVSim reports it for the whole cache.
    # Scale so the Part C comparison is like for like.
    if r["leak_mW"] is not None and r["nbanks"]:
        r["leak_total_mW"] = r["leak_mW"] * r["nbanks"]
    # CACTI's own bounding box: the silicon the cache actually occupies,
    # which exceeds the sum of the data and tag array areas.
    if r["h_mm"] and r["w_mm"]:
        r["footprint_mm2"] = r["h_mm"] * r["w_mm"]
    return r


def run_cacti(tag: str, overrides: dict[str, str], base: str) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = WORK / f"cacti_{tag}.cfg"
    cfg.write_text(cacti_cfg(base, overrides))
    proc = subprocess.run([str(CACTI_DIR / "cacti"), "-infile", str(cfg)],
                          cwd=CACTI_DIR, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    (WORK / f"cacti_{tag}.out").write_text(out)
    if "Access time" not in out:
        print(out[-2500:])
        sys.exit(f"CACTI run '{tag}' produced no result")
    res = parse_cacti(out)
    res["tag"] = tag
    return res


# ==========================================================================
# NVSim
# ==========================================================================
NVSIM_PATTERNS = {
    "total_area_mm2": r"Total Area = ([\d.]+)mm\^2",
    "hit_ns":         r"Cache Hit Latency\s*=\s*([\d.]+)ns",
    "miss_ns":        r"Cache Miss Latency\s*=\s*([\d.]+)ns",
    "write_ns":       r"Cache Write Latency\s*=\s*([\d.]+)ns",
    "hit_nJ":         r"Cache Hit Dynamic Energy\s*=\s*([\d.]+)nJ",
    "write_nJ":       r"Cache Write Dynamic Energy\s*=\s*([\d.]+)nJ",
    "leak_mW":        r"Cache Total Leakage Power\s*=\s*([\d.]+)mW",
}


def parse_nvsim(out: str) -> dict:
    r = {}
    for k, pat in NVSIM_PATTERNS.items():
        m = re.search(pat, out)
        r[k] = float(m.group(1)) if m else None
    m = re.search(r"Cell Area \(F\^2\)\s*:\s*([\d.]+)", out)
    r["cell_F2"] = float(m.group(1)) if m else None
    r["read_pJ"] = r["hit_nJ"] * 1e3 if r["hit_nJ"] else None
    r["write_pJ"] = r["write_nJ"] * 1e3 if r["write_nJ"] else None
    return r


def run_nvsim(tag: str, cell_over: dict[str, str]) -> dict:
    """Run NVSim with a variant of partC/sample.cell."""
    WORK.mkdir(parents=True, exist_ok=True)
    cell_txt = (PARTC / "sample.cell").read_text()
    for prefix, newline in cell_over.items():
        cell_txt = re.sub(rf"^{re.escape(prefix)}.*$", newline, cell_txt,
                          flags=re.MULTILINE)
    cell = WORK / f"cell_{tag}.cell"
    cell.write_text(cell_txt)

    cfg_txt = (PARTC / "STT_cache.cfg").read_text()
    cfg_txt = re.sub(r"^-MemoryCellInputFile:.*$",
                     f"-MemoryCellInputFile: {cell}", cfg_txt, flags=re.MULTILINE)
    cfg = WORK / f"nvsim_{tag}.cfg"
    cfg.write_text(cfg_txt)

    proc = subprocess.run([str(NVSIM_DIR / "nvsim"), str(cfg)],
                          cwd=NVSIM_DIR, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    (WORK / f"nvsim_{tag}.out").write_text(out)
    if "CACHE DESIGN -- SUMMARY" not in out:
        print(out[-2500:])
        sys.exit(f"NVSim run '{tag}' produced no result")
    res = parse_nvsim(out)
    res["tag"] = tag
    return res


# ==========================================================================
def fmt(x, n=2):
    return "--" if x is None else f"{x:.{n}f}"


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    base = (PARTB / "cache.cfg").read_text()

    # ---------------- Part B Task 1: baseline -----------------------------
    print("CACTI: baseline 2 MB ...")
    b1 = run_cacti("baseline", {}, base)

    # ---------------- Part B Task 2: capacity sweep -----------------------
    print("CACTI: capacity sweep ...")
    caps = [262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216]
    sweep = []
    for c in caps:
        r = run_cacti(f"cap{c}", {"-size (bytes)": f"-size (bytes) {c}"}, base)
        r["bytes"] = c
        sweep.append(r)

    # ---------------- Part B Task 3: objectives ---------------------------
    print("CACTI: objective sweep ...")
    OBJ = "-design objective (weight delay, dynamic power, leakage power, cycle time, area)"
    OPT = "-Optimize ED or ED^2 (ED, ED^2, NONE):"
    objectives = {
        "delay": {OBJ: f"{OBJ} 100:0:0:0:0", OPT: f'{OPT} "NONE"'},
        "area":  {OBJ: f"{OBJ} 0:0:0:0:100", OPT: f'{OPT} "NONE"'},
        "ed2p":  {OBJ: f"{OBJ} 0:0:0:100:0", OPT: f'{OPT} "ED^2"'},
    }
    objres = {k: run_cacti(f"obj_{k}", v, base) for k, v in objectives.items()}

    # ---------------- Part B Task 4: bitline hand-check -------------------
    # 0.38 r c L^2 for a distributed RC bitline.  Constants stated explicitly so
    # the discrepancy is attributable.
    W_WIRE = 0.045e-6           # minimum-pitch local wire, m (CACTI's own value)
    AR = 1.8                    # Cu aspect ratio at 45 nm
    RHO = 2.2e-8                # effective Cu resistivity incl. scattering, ohm.m
    C_WIRE = 0.20e-15           # local wire capacitance, F per um
    C_CELL = 0.20e-15           # drain junction + contact per cell, F per um of bitline
    L_um = b1["subarray_h_mm"] * 1e3        # bitline length, um
    r_per_um = RHO * 1e-6 / (W_WIRE * AR * W_WIRE)   # ohm per um
    c_per_um = C_WIRE + C_CELL
    t_hand_ns = 0.38 * r_per_um * c_per_um * L_um ** 2 * 1e9
    t_cacti_ns = b1["bitline_ns"]
    ratio = t_cacti_ns / t_hand_ns

    # ---------------- Part C ---------------------------------------------
    print("NVSim: STT-MRAM baseline ...")
    c1 = run_nvsim("baseline", {})
    print("NVSim: TMR 4k/12k ...")
    c3 = run_nvsim("tmr4k12k", {
        "-ResistanceOn (ohm):": "-ResistanceOn (ohm): 4000",
        "-ResistanceOff (ohm):": "-ResistanceOff (ohm): 12000",
    })
    print("NVSim: reset current 100 uA, access transistor resized ...")
    # 54 F^2 at aspect ratio 2.0 is 10.392 F x 5.196 F.  The 6 F access device
    # lies along the long dimension, so 4.392 F of that is contact and spacing.
    # Halving the write current halves the required width, 6 F -> 3 F, giving a
    # long dimension of 7.392 F and a cell of 7.392 x 5.196 = 38.4 F^2.
    c4b = run_nvsim("reset100_resized", {
        "-ResetCurrent (uA):": "-ResetCurrent (uA): 100",
        "-SetCurrent (uA):": "-SetCurrent (uA): 100",
        "-AccessCMOSWidth (F):": "-AccessCMOSWidth (F): 3",
        "-CellArea (F^2):": "-CellArea (F^2): 38.4",
        "-CellAspectRatio:": "-CellAspectRatio: 1.4226",
    })

    print("NVSim: reset current 100 uA ...")
    c4 = run_nvsim("reset100", {
        "-ResetCurrent (uA):": "-ResetCurrent (uA): 100",
        "-SetCurrent (uA):": "-SetCurrent (uA): 100",
    })

    # C3 extended: TMR ratio sweep at fixed R_on
    print("NVSim: TMR sweep ...")
    tmr_sweep = []
    for roff in (4500, 6000, 9000, 12000, 18000):
        r = run_nvsim(f"tmr{roff}", {
            "-ResistanceOff (ohm):": f"-ResistanceOff (ohm): {roff}"})
        r["tmr"] = roff / 3000.0
        tmr_sweep.append(r)

    # C4 extended: write-current sweep, to locate the knee where the access
    # transistor - not the specified AccessCMOSWidth - starts setting cell area
    print("NVSim: write-current sweep ...")
    ireset_sweep = []
    for i in (50, 100, 200, 300, 400, 600, 800):
        r = run_nvsim(f"ireset{i}", {
            "-ResetCurrent (uA):": f"-ResetCurrent (uA): {i}",
            "-SetCurrent (uA):": f"-SetCurrent (uA): {i}"})
        r["ireset"] = i
        ireset_sweep.append(r)

    # ---------------- figures --------------------------------------------
    print("writing figures ...")

    # B2: access time vs log2 capacity
    x = np.log2([s["bytes"] for s in sweep])
    y = np.array([s["access_ns"] for s in sweep])
    m, cst = np.polyfit(x, y, 1)
    incr = np.diff(y) * 1e3          # ps added per capacity doubling
    FO4_PS = 20.0                    # approx FO4 inverter delay at 45 nm

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1.4]})
    ax.plot(x, y, "o-", ms=4, color=BLUE, label="CACTI access time")
    ax.plot(x, m * x + cst, ls="--", lw=1.1, color=GREY,
            label=f"linear fit ({m:.2f} ns/doubling) --- poor")
    ax.set_ylabel("access time (ns)")
    ax.set_title("Task B2: is \"one gate delay per doubling\" visible?", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)

    ax2.bar(x[1:], incr, width=0.55, color=ORANGE)
    ax2.axhline(FO4_PS, color="k", ls="--", lw=1.0)
    ax2.annotate("1 FO4 $\\approx$ 20 ps", xy=(18.3, FO4_PS * 1.9), fontsize=7.5)
    ax2.set_yscale("log")
    for xi, v in zip(x[1:], incr):
        ax2.annotate(f"{v:.0f}", xy=(xi, v * 1.25), ha="center", fontsize=7)
    ax2.set_ylabel("added per\ndoubling (ps)")
    ax2.set_xlabel("$\\log_2$(capacity in bytes)")
    labels = [f"{s['bytes']//1024}k" if s["bytes"] < 1048576
              else f"{s['bytes']//1048576}M" for s in sweep]
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(xi)}\n{lb}" for xi, lb in zip(x, labels)],
                        fontsize=7.5)
    save(fig, IMAGES / "B_task2_access_vs_capacity.pdf")

    # B1: delay breakdown
    listed = (b1["htree_in_ns"] + b1["decoder_ns"] + b1["bitline_ns"]
              + b1["senseamp_ns"] + b1["htree_out_ns"])
    parts = [("H-tree in", b1["htree_in_ns"]), ("Decoder+WL", b1["decoder_ns"]),
             ("Bitline", b1["bitline_ns"]), ("Sense amp", b1["senseamp_ns"]),
             ("H-tree out", b1["htree_out_ns"]),
             ("Output driver", b1["access_ns"] - listed)]
    fig, ax = plt.subplots(figsize=(5.4, 2.2))
    left = 0.0
    colors = [GREY, ORANGE, BLUE, RED, "#9467bd", "#8c564b"]
    for (lab, v), col in zip(parts, colors):
        ax.barh([0], [v], left=left, color=col, edgecolor="white", height=0.5)
        if v / b1["access_ns"] > 0.05:
            ax.text(left + v / 2, 0, f"{lab}\n{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white")
        left += v
    ax.set_yticks([])
    ax.set_xlabel("access time contribution (ns)")
    ax.set_title(f"Task B1: where the {b1['access_ns']:.2f} ns goes", fontsize=9)
    ax.grid(False)
    save(fig, IMAGES / "B_task1_delay_breakdown.pdf")

    # C: SRAM vs STT-MRAM radar-free bar comparison (log ratio)
    metrics = [
        ("Read latency", b1["access_ns"], c1["hit_ns"]),
        ("Write latency", b1["access_ns"], c1["write_ns"]),
        ("Read energy", b1["read_pJ"], c1["read_pJ"]),
        ("Write energy", b1["write_pJ"], c1["write_pJ"]),
        ("Leakage", b1["leak_total_mW"], c1["leak_mW"]),
        ("Area", b1["area_mm2"], c1["total_area_mm2"]),
    ]
    labels = [m[0] for m in metrics]
    ratios = [m[2] / m[1] for m in metrics]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    cols = [RED if r > 1 else GREEN for r in ratios]
    ax.barh(labels, ratios, color=cols, height=0.6)
    ax.axvline(1.0, color="k", lw=1.0)
    ax.set_xscale("log")
    for i, r in enumerate(ratios):
        ax.text(r * (1.12 if r > 1 else 0.88), i, f"{r:.2f}$\\times$",
                va="center", ha="left" if r > 1 else "right", fontsize=8)
    ax.set_xlabel("STT-MRAM / SRAM  (log scale; $<1$ favours MRAM)")
    ax.set_title("Task C1--C2: 2 MB STT-MRAM against the CACTI SRAM baseline",
                 fontsize=9)
    ax.set_xlim(min(ratios) * 0.45, max(ratios) * 2.6)
    save(fig, IMAGES / "C_task1_sram_vs_mram.pdf")

    # C4: area vs write current
    iw = np.array([r["ireset"] for r in ireset_sweep])
    ia = np.array([r["total_area_mm2"] for r in ireset_sweep])
    fig, ax = plt.subplots()
    ax.plot(iw, ia, "o-", ms=4, color=BLUE)
    ax.axvline(200, color=GREY, ls=":", lw=1.0)
    ax.annotate("assignment baseline\n200 $\\mu$A", xy=(206, ia.min() + 0.02),
                fontsize=7.5, color=GREY)
    flat = ia[iw <= 200].max()
    ax.axhline(flat, color=GREEN, ls="--", lw=1.0)
    ax.annotate("floor set by AccessCMOSWidth = 6F,\nnot by the write current",
                xy=(60, flat + 0.03), fontsize=7.5, color=GREEN)
    ax.set_xlabel("write (reset/set) current per cell ($\\mu$A)")
    ax.set_ylabel("total array area (mm$^2$)")
    ax.set_title("Task C4: write current sizes the access transistor,\n"
                 "which sizes the cell", fontsize=9)
    save(fig, IMAGES / "C_task4_area_vs_current.pdf")

    # ---------------- tables ---------------------------------------------
    print("writing tables ...")

    (DATA / "table_partB_baseline.tex").write_text(
        r"""\begin{tabular}{lr}
\toprule
Metric & 2\,MB SRAM L2, 45\,nm \\
\midrule
Access time (ns) & %.2f \\
Cycle time (ns) & %.2f \\
Dynamic read energy (pJ/access) & %.1f \\
Dynamic write energy (pJ/access) & %.1f \\
Leakage, whole cache (mW) & %.0f \\
Total area (mm$^2$) & %.2f \\
Area efficiency (\%%) & %.1f \\
Winning organisation (Ndwl/Ndbl/Nspd) & %d\,/\,%d\,/\,%d \\
Tag organisation (Ntwl/Ntbl/Ntspd) & %d\,/\,%d\,/\,%d \\
\bottomrule
\end{tabular}
""" % (b1["access_ns"], b1["cycle_ns"], b1["read_pJ"], b1["write_pJ"],
       b1["leak_total_mW"], b1["area_mm2"], b1["area_eff_pct"],
       b1["ndwl"], b1["ndbl"], b1["nspd"], b1["ntwl"], b1["ntbl"], b1["ntspd"]))

    def cap_label(nbytes: int) -> str:
        if nbytes < 1048576:
            return f"{nbytes // 1024}" + r"\,kB"
        return f"{nbytes // 1048576}" + r"\,MB"

    rows = "\n".join(
        f"{cap_label(s['bytes'])} & "
        f"{np.log2(s['bytes']):.0f} & {s['access_ns']:.2f} & "
        f"{s['read_pJ']:.0f} & {s['leak_total_mW']:.0f} & {s['area_mm2']:.2f} \\\\"
        for s in sweep)
    (DATA / "table_partB_sweep.tex").write_text(
        r"""\begin{tabular}{lccrrr}
\toprule
Capacity & $\log_2$(bytes) & Access (ns) & Read (pJ) & Leak (mW) & Area (mm$^2$) \\
\midrule
""" + rows + "\n" + r"""\bottomrule
\end{tabular}
""")

    onames = {"delay": "Pure delay (100:0:0:0:0)",
              "area": "Pure area (0:0:0:0:100)",
              "ed2p": r"ED$^2$P"}
    rows = "\n".join(
        f"{onames[k]} & {int(objres[k]['ndwl'])}\\,/\\,{int(objres[k]['ndbl'])}\\,/\\,"
        f"{int(objres[k]['nspd'])} & {objres[k]['access_ns']:.2f} & "
        f"{objres[k]['read_pJ']:.0f} & {objres[k]['area_mm2']:.2f} \\\\"
        for k in ("delay", "area", "ed2p"))
    (DATA / "table_partB_objectives.tex").write_text(
        r"""\begin{tabular}{lcrrr}
\toprule
Objective & Ndwl/Ndbl/Nspd & Access (ns) & Read (pJ) & Area (mm$^2$) \\
\midrule
""" + rows + "\n" + r"""\bottomrule
\end{tabular}
""")

    crows = "\n".join(
        f"{lab} & {sram:.2f} & {mram:.2f} & {mram/sram:.2f}$\\times$ \\\\"
        for lab, sram, mram in metrics)
    (DATA / "table_partC_compare.tex").write_text(
        r"""\begin{tabular}{lrrr}
\toprule
Metric & SRAM (CACTI) & STT-MRAM (NVSim) & Ratio \\
\midrule
""" + crows + "\n" + r"""\bottomrule
\end{tabular}
""")

    var_rows = "\n".join([
        f"Baseline (3\\,/\\,6\\,k$\\Omega$, 200\\,\\si{{\\micro\\ampere}}) & "
        f"{c1['hit_ns']:.2f} & {c1['write_ns']:.2f} & {c1['read_pJ']:.0f} & "
        f"{c1['write_pJ']:.0f} & {c1['leak_mW']:.0f} & {c1['total_area_mm2']:.3f} \\\\",
        f"TMR 3:1 (4\\,/\\,12\\,k$\\Omega$) & "
        f"{c3['hit_ns']:.2f} & {c3['write_ns']:.2f} & {c3['read_pJ']:.0f} & "
        f"{c3['write_pJ']:.0f} & {c3['leak_mW']:.0f} & {c3['total_area_mm2']:.3f} \\\\",
        f"Write current halved (100\\,\\si{{\\micro\\ampere}}) & "
        f"{c4['hit_ns']:.2f} & {c4['write_ns']:.2f} & {c4['read_pJ']:.0f} & "
        f"{c4['write_pJ']:.0f} & {c4['leak_mW']:.0f} & {c4['total_area_mm2']:.3f} \\\\",
        f"100\\,\\si{{\\micro\\ampere}} with cell resized to 38.4\\,F$^2$ & "
        f"{c4b['hit_ns']:.2f} & {c4b['write_ns']:.2f} & {c4b['read_pJ']:.0f} & "
        f"{c4b['write_pJ']:.0f} & {c4b['leak_mW']:.0f} & {c4b['total_area_mm2']:.3f} \\\\",
    ])
    (DATA / "table_partC_variants.tex").write_text(
        r"""\begin{tabular}{lrrrrrr}
\toprule
Cell variant & Read & Write & Read & Write & Leak & Area \\
 & (ns) & (ns) & (pJ) & (pJ) & (mW) & (mm$^2$) \\
\midrule
""" + var_rows + "\n" + r"""\bottomrule
\end{tabular}
""")

    # ---------------- macros ---------------------------------------------
    def mac(n, v):
        return r"\newcommand{\%s}{%s}" % (n, v)

    gate_delay_ps = 1000 * m  # ns per doubling -> ps
    macros = [
        mac("Baccess", fmt(b1["access_ns"])),
        mac("Bcycle", fmt(b1["cycle_ns"])),
        mac("Bread", fmt(b1["read_pJ"], 0)),
        mac("Bwrite", fmt(b1["write_pJ"], 0)),
        mac("Bleak", fmt(b1["leak_total_mW"], 0)),
        mac("Barea", fmt(b1["area_mm2"])),
        mac("Bareaeff", fmt(b1["area_eff_pct"], 1)),
        mac("Bndwl", str(int(b1["ndwl"]))),
        mac("Bndbl", str(int(b1["ndbl"]))),
        mac("Bnspd", str(int(b1["nspd"]))),
        mac("Bhitcycles", str(int(np.ceil(b1["access_ns"] / 0.5)))),
        mac("BperDoubling", fmt(m, 3)),
        mac("BperDoublingPs", fmt(gate_delay_ps, 0)),
        mac("Bbitline", fmt(b1["bitline_ns"], 3)),
        mac("Bhtree", fmt(b1["htree_in_ns"] + b1["htree_out_ns"], 2)),
        mac("BhtreePct", fmt(100 * (b1["htree_in_ns"] + b1["htree_out_ns"])
                             / b1["access_ns"], 0)),
        mac("BbitlinePct", fmt(100 * b1["bitline_ns"] / b1["access_ns"], 0)),
        mac("Bdecoder", fmt(b1["decoder_ns"], 2)),
        mac("Bhandcalc", fmt(t_hand_ns, 3)),
        mac("Bhandratio", fmt(ratio, 1)),
        mac("BblLen", fmt(L_um, 0)),
        mac("Brperum", fmt(r_per_um, 1)),
        mac("Bcperum", fmt(c_per_um * 1e15, 2)),
        mac("BobjDelayAcc", fmt(objres["delay"]["access_ns"])),
        mac("BobjAreaAcc", fmt(objres["area"]["access_ns"])),
        mac("BobjDelayArea", fmt(objres["delay"]["area_mm2"])),
        mac("BobjAreaArea", fmt(objres["area"]["area_mm2"])),
        mac("BobjAreaSaving", fmt(100 * (1 - objres["area"]["area_mm2"]
                                         / objres["delay"]["area_mm2"]), 0)),
        mac("BobjDelayPenalty", fmt(objres["area"]["access_ns"]
                                    / objres["delay"]["access_ns"], 1)),
        # Part C
        mac("Cread", fmt(c1["hit_ns"])),
        mac("Cwrite", fmt(c1["write_ns"])),
        mac("CreadE", fmt(c1["read_pJ"], 0)),
        mac("CwriteE", fmt(c1["write_pJ"], 0)),
        mac("Cleak", fmt(c1["leak_mW"], 0)),
        mac("Carea", fmt(c1["total_area_mm2"], 2)),
        mac("CareaRatio", fmt(b1["area_mm2"] / c1["total_area_mm2"], 1)),
        mac("CleakRatio", fmt(b1["leak_total_mW"] / c1["leak_mW"], 1)),
        mac("CwriteRatio", fmt(c1["write_ns"] / b1["access_ns"], 1)),
        mac("CwriteERatio", fmt(c1["write_pJ"] / b1["write_pJ"], 1)),
        mac("CreadRatio", fmt(c1["hit_ns"] / b1["access_ns"], 2)),
        mac("CdensityMB", fmt(2 * b1["area_mm2"] / c1["total_area_mm2"], 1)),
        mac("CtmrRead", fmt(c3["hit_ns"])),
        mac("CtmrReadE", fmt(c3["read_pJ"], 0)),
        mac("CtmrArea", fmt(c3["total_area_mm2"], 2)),
        mac("CtmrReadDelta", fmt(100 * (c3["hit_ns"] / c1["hit_ns"] - 1), 1)),
        mac("CresetArea", fmt(c4["total_area_mm2"], 2)),
        mac("CresetAreaDelta", fmt(100 * (1 - c4["total_area_mm2"]
                                          / c1["total_area_mm2"]), 0)),
        mac("CresetWrite", fmt(c4["write_ns"])),
        mac("CresetWriteE", fmt(c4["write_pJ"], 0)),
        mac("CrsArea", fmt(c4b["total_area_mm2"], 2)),
        mac("CrsAreaDrop", fmt(100*(1-c4b["total_area_mm2"]/c1["total_area_mm2"]), 1)),
        mac("CrsLeak", fmt(c4b["leak_mW"], 0)),
        mac("CrsLeakDrop", fmt(100*(1-c4b["leak_mW"]/c1["leak_mW"]), 1)),
        mac("CrsWriteE", fmt(c4b["write_pJ"], 0)),
        mac("CrsWriteEDrop", fmt(100*(1-c4b["write_pJ"]/c1["write_pJ"]), 1)),
        mac("CrsCellF", "38.4"),
        mac("CrsCellDrop", "28.9"),
        mac("CsenseTwo", "120"),
        mac("CsenseThree", "320"),
        mac("Bfootprint", fmt(b1.get("footprint_mm2"), 2)),
        mac("Cknee", str(int(iw[np.argmax(ia > flat + 1e-9)])) if (ia > flat + 1e-9).any() else "--"),
        mac("CareaAtMax", fmt(ia.max(), 2)),
        mac("CimaxUA", str(int(iw[-1]))),
        mac("CareaGrowthPct", fmt(100 * (ia.max() / flat - 1), 0)),
        mac("CtmrMinArea", fmt(min(r["total_area_mm2"] for r in tmr_sweep), 2)),
        mac("CtmrMaxArea", fmt(max(r["total_area_mm2"] for r in tmr_sweep), 2)),
        mac("CtmrReadSpread", fmt(100 * (max(r["hit_ns"] for r in tmr_sweep)
                                         / min(r["hit_ns"] for r in tmr_sweep) - 1), 1)),
    ]
    (DATA / "macros_partBC.tex").write_text("\n".join(macros) + "\n")

    (DATA / "partBC_summary.json").write_text(json.dumps({
        "B_baseline": b1, "B_sweep": sweep, "B_objectives": objres,
        "B_handcheck": {"L_um": L_um, "r_ohm_per_um": r_per_um,
                        "c_F_per_um": c_per_um, "t_hand_ns": t_hand_ns,
                        "t_cacti_ns": t_cacti_ns, "ratio": ratio},
        "C_baseline": c1, "C_tmr": c3, "C_reset100": c4,
        "C_tmr_sweep": tmr_sweep, "C_ireset_sweep": ireset_sweep,
    }, indent=2, default=str))

    # ---------------- console --------------------------------------------
    print(f"""
Part B  (CACTI, 2 MB SRAM, 45 nm)
  access {b1['access_ns']:.2f} ns | read {b1['read_pJ']:.0f} pJ | """
          f"""leak {b1['leak_total_mW']:.0f} mW (total) | area {b1['area_mm2']:.2f} mm^2
  organisation Ndwl/Ndbl/Nspd = {int(b1['ndwl'])}/{int(b1['ndbl'])}/{int(b1['nspd'])}
  B2 slope {m:.3f} ns per capacity doubling ({1000*m:.0f} ps)
  B3 delay-opt {objres['delay']['access_ns']:.2f} ns / """
          f"""{objres['delay']['area_mm2']:.2f} mm^2 | """
          f"""area-opt {objres['area']['access_ns']:.2f} ns / """
          f"""{objres['area']['area_mm2']:.2f} mm^2
  B4 bitline: CACTI {t_cacti_ns:.3f} ns vs hand 0.38rcL^2 = """
          f"""{t_hand_ns:.3f} ns  -> {ratio:.1f}x

Part C  (NVSim, 2 MB STT-MRAM)
  read {c1['hit_ns']:.2f} ns | write {c1['write_ns']:.2f} ns | """
          f"""readE {c1['read_pJ']:.0f} pJ | writeE {c1['write_pJ']:.0f} pJ
  leak {c1['leak_mW']:.0f} mW | area {c1['total_area_mm2']:.2f} mm^2 """
          f"""({b1['area_mm2']/c1['total_area_mm2']:.1f}x denser than SRAM)
  C3 TMR 3:1  -> read {c3['hit_ns']:.2f} ns """
          f"""({100*(c3['hit_ns']/c1['hit_ns']-1):+.1f} %), area {c3['total_area_mm2']:.2f}
  C4 Ireset/2 -> area {c4['total_area_mm2']:.2f} mm^2 """
          f"""({100*(1-c4['total_area_mm2']/c1['total_area_mm2']):+.0f} %), """
          f"""write {c4['write_ns']:.2f} ns
  C4 area vs Ireset: """
          + " ".join(f"{int(r['ireset'])}uA={r['total_area_mm2']:.2f}"
                     for r in ireset_sweep) + f"""
  C3 TMR sweep read ns: """
          + " ".join(f"{r['tmr']:.1f}:1={r['hit_ns']:.2f}" for r in tmr_sweep) + "\n")


if __name__ == "__main__":
    main()
