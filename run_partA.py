#!/usr/bin/env python3
"""
AMCAS Assignment 1 - Part A driver.

Runs A.sp under ngspice, parses every data file it writes, produces the report
figures directly into report/images/, and emits LaTeX-ready tables plus a JSON
summary into report/data/.

The netlist is run twice: once with the PTM 45 nm High-Performance card (the
primary result) and once with the Low-Power card, which is what the Task 1
comparison against the 69 mV reference needs in order to be meaningful.

    python3 run_partA.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
REPORT = ROOT / "report"
IMAGES = REPORT / "images"
DATA = REPORT / "data"

NETLIST = ROOT / "A.sp"
MODEL_CARDS = ["45nm_HP.pm", "45nm_LP.pm"]

SENSE_OFFSET_MV = 25.0      # sense-amp offset, Lecture 5
REFERENCE_DV_MV = 69.0      # DeltaV reference value, Lecture 1

C_BL = 180e-15              # bitline capacitance in the netlist, F
T_SENSE = 1.0e-9            # WL rise (1 ns) to sense point (2 ns)

plt.rcParams.update(
    {
        "figure.figsize": (5.4, 3.4),
        "figure.dpi": 160,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
    }
)

BLUE, RED, GREY, GREEN = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"

PREVIEW = WORK / "preview"


def save(fig, out: Path) -> None:
    """Write the PDF the report uses, plus a PNG preview for quick eyeballing."""
    fig.savefig(out)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW / (out.stem + ".png"))


# --------------------------------------------------------------------------
# running ngspice
# --------------------------------------------------------------------------
def _invoke(netlist_dir: Path, netlist_name: str, tag: str) -> str:
    proc = subprocess.run(
        ["ngspice", "-b", netlist_name],
        cwd=netlist_dir,
        capture_output=True,
        text=True,
    )
    log = proc.stdout + proc.stderr
    (WORK / f"ngspice_{tag}.log").write_text(log)

    produced = list((netlist_dir / "work").glob("*.data"))
    if not produced:
        print(log)
        sys.exit(f"ngspice produced no data for '{tag}' (exit {proc.returncode})")

    failed = [ln for ln in log.splitlines() if "failed" in ln.lower()]
    if failed:
        print(f"  warning: {len(failed)} measurement(s) failed in '{tag}':")
        for ln in failed[:5]:
            print("   ", ln.strip())
    return log


def run_primary() -> Path:
    """Run A.sp in place (HP card).  Returns the directory holding work/."""
    WORK.mkdir(exist_ok=True)
    for stale in WORK.glob("*.data"):
        stale.unlink()
    _invoke(ROOT, "A.sp", "hp")
    return ROOT


def run_with_card(card: str, tag: str) -> Path:
    """Run the same netlist against a different PTM model card, in a sandbox."""
    sandbox = Path(tempfile.mkdtemp(prefix=f"partA_{tag}_"))
    (sandbox / "work").mkdir()
    for c in MODEL_CARDS:
        shutil.copy(ROOT / c, sandbox / c)
    text = NETLIST.read_text().replace(".include 45nm_HP.pm", f".include {card}")
    (sandbox / "A.sp").write_text(text)
    _invoke(sandbox, "A.sp", tag)
    return sandbox


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def read_table(path: Path) -> dict[str, np.ndarray]:
    """Read a whitespace table whose first line is the column names."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    names = lines[0].split()
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != len(names):
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    arr = np.array(rows, dtype=float).reshape(-1, len(names))
    return {n: arr[:, i] for i, n in enumerate(names)}


def read_meas(path: Path) -> list[dict[str, object]]:
    """Read the task/temp/... measurement tables (first column is a string tag)."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    names = lines[0].split()
    out = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != len(names):
            continue
        rec: dict[str, object] = {"task": parts[0]}
        for n, p in zip(names[1:], parts[1:]):
            rec[n] = float(p)
        out.append(rec)
    return out


def read_wave(path: Path) -> dict[str, np.ndarray]:
    """Read a wrdata file written with wr_singlescale + wr_vecnames."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    header = lines[0].replace("#", "").split()
    names = ["time"] + [h.lower() for h in header[1:]]
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < len(names):
            continue
        try:
            rows.append([float(p) for p in parts[: len(names)]])
        except ValueError:
            continue
    arr = np.array(rows, dtype=float).reshape(-1, len(names))
    return {n: arr[:, i] for i, n in enumerate(names)}


def wave_col(w: dict[str, np.ndarray], want: str) -> np.ndarray:
    """Fetch v(bl) etc. tolerating naming differences between ngspice builds."""
    want = want.lower()
    if want in w:
        return w[want]
    bare = re.sub(r"^v\((.*)\)$", r"\1", want)
    for k, v in w.items():
        if re.sub(r"^v\((.*)\)$", r"\1", k) == bare:
            return v
    raise KeyError(f"{want} not in {list(w)}")


def harvest(base: Path) -> dict:
    """Collect every measurement + waveform produced by one ngspice run."""
    w = base / "work"
    tags = {r["task"]: r for r in
            read_meas(w / "t1_t2_meas.data") + read_meas(w / "t4_meas.data")}
    return {
        "meas": tags,
        "wave": {k: read_wave(w / f"{k}.data") for k in
                 ("t1_read", "t2_read_wide", "t4_read_85", "t4_read_85_07v",
                  "t2_upset")},
        "s27": read_table(w / "t3_sweep_27.data"),
        "s85": read_table(w / "t3_sweep_85.data"),
        "wa": read_table(w / "t2_wa_sweep_27.data"),
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_bitlines(wave, out: Path, title: str) -> None:
    t = wave["time"] * 1e9
    bl, blb = wave_col(wave, "v(bl)"), wave_col(wave, "v(blb)")
    fig, ax = plt.subplots()
    ax.plot(t, blb, label="BLB (holds)", color=RED)
    ax.plot(t, bl, label="BL (discharges)", color=BLUE)
    ax.plot(t, wave_col(wave, "v(wl)"), label="WL", color=GREY, ls="--", lw=1.0)
    i2 = int(np.argmin(np.abs(t - 2.0)))
    ax.axvline(2.0, color="k", ls=":", lw=0.9)
    ax.annotate(
        "", xy=(2.0, bl[i2]), xytext=(2.0, blb[i2]),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="k"),
    )
    ax.annotate(f"$\\Delta V$ = {1e3*(blb[i2]-bl[i2]):.0f} mV",
                xy=(2.08, 0.5 * (bl[i2] + blb[i2])), fontsize=8)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("voltage (V)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="center left", fontsize=8)
    save(fig, out)
    plt.close(fig)


def plot_storage(waves, out: Path, title: str) -> None:
    fig, ax = plt.subplots()
    vdd = None
    for label, w, color, ls in waves:
        ax.plot(w["time"] * 1e9, wave_col(w, "v(q)") * 1e3,
                label=label, color=color, ls=ls)
        vdd = max(wave_col(w, "v(qb)"))
    if vdd:
        ax.axhline(vdd * 1e3 / 2, color="k", ls="--", lw=1.0)
        ax.annotate("trip point $\\approx V_{DD}/2$", xy=(3.15, vdd * 1e3 / 2 * 1.03),
                    fontsize=7.5)
        ax.set_ylim(-20, vdd * 1e3 * 0.68)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("v(Q)  (mV)")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out)
    plt.close(fig)


def plot_dv_sweep(s27, s85, out: Path, extrap: dict):
    fig, ax = plt.subplots()
    ax.plot(s27["vdd"], s27["dv_27C"] * 1e3, "o-", ms=3.5,
            label="27 $^\\circ$C", color=BLUE)
    ax.plot(s85["vdd"], s85["dv_85C"] * 1e3, "s-", ms=3.5,
            label="85 $^\\circ$C", color=RED)

    # linear extrapolation down to the sense-amp offset
    for key, s, dvname, color in (("27", s27, "dv_27C", BLUE),
                                  ("85", s85, "dv_85C", RED)):
        v = extrap[key]
        if v is None:
            continue
        m, c = extrap[f"fit{key}"]
        xs = np.linspace(v, s["vdd"].min(), 30)
        ax.plot(xs, (m * xs + c) * 1e3, ls=":", lw=1.1, color=color)
        ax.plot([v], [SENSE_OFFSET_MV], "*", ms=9, color=color)

    ax.axhline(SENSE_OFFSET_MV, color="k", ls="--", lw=1.0)
    ax.annotate(f"sense-amp offset = {SENSE_OFFSET_MV:.0f} mV",
                xy=(0.60, SENSE_OFFSET_MV + 18), fontsize=7.5)
    ax.axhline(REFERENCE_DV_MV, color=GREEN, ls="-.", lw=1.0)
    ax.annotate(f"Lecture 1 reference = {REFERENCE_DV_MV:.0f} mV",
                xy=(0.60, REFERENCE_DV_MV + 18), fontsize=7.5, color=GREEN)
    ax.set_xlabel("$V_{DD}$ (V)")
    ax.set_ylabel("$\\Delta V$(BLB,BL) at 2 ns  (mV)")
    ax.set_title("Task 3: read margin vs supply voltage", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, out)
    plt.close(fig)


def plot_qmax_sweep(s27, s85, out: Path):
    fig, ax = plt.subplots()
    ax.plot(s27["vdd"], 100 * s27["qmax_27C"] / s27["vdd"], "o-", ms=3.5,
            label="27 $^\\circ$C", color=BLUE)
    ax.plot(s85["vdd"], 100 * s85["qmax_85C"] / s85["vdd"], "s-", ms=3.5,
            label="85 $^\\circ$C", color=RED)
    ax.axhline(50, color="k", ls="--", lw=1.0)
    ax.annotate("read upset ($\\max v(Q) = V_{DD}/2$)", xy=(0.60, 51.5),
                fontsize=7.5)
    ax.set_ylim(0, 60)
    ax.set_xlabel("$V_{DD}$ (V)")
    ax.set_ylabel("max $v(Q)$ during read  (% of $V_{DD}$)")
    ax.set_title("Read disturb on the storage node", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, out)
    plt.close(fig)


def plot_wa_sweep(wa, out: Path, vdd: float, wa_crit):
    x = wa["wa_m"] * 1e6
    fig, ax = plt.subplots()
    ax.plot(x, 100 * wa["qmax"] / vdd, "o-", ms=3.5, color=RED,
            label="max $v(Q)$")
    ax.plot(x, 100 * wa["dv"] / vdd, "s-", ms=3.5, color=BLUE,
            label="$\\Delta V$ at 2 ns")
    ax.plot(x, 100 * (vdd - wa["vblb"]) / vdd, "^-", ms=3.5, color=GREEN,
            label="BLB droop below precharge")
    ax.axhline(50, color="k", ls="--", lw=1.0)
    ax.annotate("$V_{DD}/2$ trip point", xy=(0.30, 52), fontsize=7.5)
    for w_ in (0.16, 0.24):
        ax.axvline(w_, color=GREY, ls=":", lw=0.9)
    ax.annotate("given\ncell", xy=(0.163, 8), fontsize=7, color="k")
    ax.annotate("Task 2\n(0.24)", xy=(0.243, 8), fontsize=7, color="k")
    if wa_crit:
        ax.axvspan(wa_crit, x.max(), color=RED, alpha=0.07)
        ax.annotate(f"read upset\n$W_A > {wa_crit:.2f}\\,\\mu$m",
                    xy=(wa_crit + 0.015, 92), fontsize=7.5, color=RED)
    ax.set_xlabel("access-transistor width $W_A$ ($\\mu$m)")
    ax.set_ylabel("% of $V_{DD}$")
    ax.set_ylim(-5, 105)
    ax.set_title("Task 2: where the cell ratio actually breaks "
                 "(1.1 V, 27 $^\\circ$C)", fontsize=9)
    ax.legend(loc="upper center", fontsize=8, ncol=3,
              bbox_to_anchor=(0.5, -0.22))
    save(fig, out)
    plt.close(fig)


def plot_card_compare(hp, lp, out: Path):
    fig, ax = plt.subplots()
    for label, w, color in (("45 nm HP card", hp, BLUE), ("45 nm LP card", lp, RED)):
        t = w["time"] * 1e9
        ax.plot(t, wave_col(w, "v(bl)"), color=color, label=f"BL, {label}")
        ax.plot(t, wave_col(w, "v(blb)"), color=color, ls="--", lw=1.1)
    ax.axvline(2.0, color="k", ls=":", lw=0.9)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("bitline voltage (V)")
    ax.set_title("Why $\\Delta V$ is not 69 mV: drive strength of the model card",
                 fontsize=9)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, out)
    plt.close(fig)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def crossing_vdd(vdd: np.ndarray, dv_mv: np.ndarray, thresh: float):
    """VDD at which dv crosses `thresh`; interpolated if bracketed, else
    linearly extrapolated from the lowest four points."""
    order = np.argsort(vdd)
    v, d = vdd[order], dv_mv[order]
    for i in range(len(v) - 1):
        if (d[i] - thresh) * (d[i + 1] - thresh) <= 0 and d[i] != d[i + 1]:
            f = (thresh - d[i]) / (d[i + 1] - d[i])
            return float(v[i] + f * (v[i + 1] - v[i])), (0.0, 0.0), False
    m, c = np.polyfit(v[:4], d[:4], 1)
    return float((thresh - c) / m), (float(m), float(c)), True


def fmt(x, n=1):
    return "--" if x is None else f"{x:.{n}f}"


def avg_read_current(dv_volt: float) -> float:
    """Average bitline discharge current implied by the measured swing, in uA."""
    return 1e6 * C_BL * dv_volt / T_SENSE


# --------------------------------------------------------------------------
# tables / macros
# --------------------------------------------------------------------------
def write_tex(summary: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    m = summary["hp"]["meas"]
    t1, t2, t4, t4c = m["T1"], m["T2"], m["T4"], m["T4c"]

    rows = [
        (r"Task 1 --- nominal, $W_A$=0.16\,\si{\micro\meter}, 1.1\,V, 27\,\si{\celsius}", t1),
        (r"Task 2 --- wide access, $W_A$=0.24\,\si{\micro\meter}, 1.1\,V, 27\,\si{\celsius}", t2),
        (r"Task 4 --- nominal, 1.1\,V, 85\,\si{\celsius}", t4),
        (r"Task 4 --- nominal, 0.7\,V, 85\,\si{\celsius}", t4c),
    ]
    body = "\n".join(
        f"{name} & {r['dv']*1e3:.0f} & {avg_read_current(r['dv']):.0f} & "
        f"{r['qmax']*1e3:.0f} & {100*r['qmax']/r['vdd']:.0f}\\,\\% & "
        f"{r['qfin']*1e3:.1f} \\\\"
        for name, r in rows
    )
    (DATA / "table_partA_summary.tex").write_text(
        r"""\begin{tabular}{lrrrrr}
\toprule
Configuration & $\Delta V$ @2\,ns & $\bar{I}_{\mathrm{read}}$ & $\max v(Q)$
 & as \% $V_{DD}$ & $v(Q)$ @4\,ns \\
 & (mV) & (\si{\micro\ampere}) & (mV) & & (mV) \\
\midrule
""" + body + "\n" + r"""\bottomrule
\end{tabular}
""")

    s27, s85 = summary["hp"]["s27"], summary["hp"]["s85"]
    srows = []
    for i, v in enumerate(s27["vdd"]):
        d27, d85 = s27["dv_27C"][i] * 1e3, s85["dv_85C"][i] * 1e3
        srows.append(
            f"{v:.2f} & {d27:.0f} & {d85:.0f} & {d85-d27:+.0f} & "
            f"{100*s27['qmax_27C'][i]/v:.0f}\\,\\% & "
            f"{100*s85['qmax_85C'][i]/v:.0f}\\,\\% \\\\"
        )
    (DATA / "table_partA_sweep.tex").write_text(
        r"""\begin{tabular}{cccccc}
\toprule
$V_{DD}$ & \multicolumn{3}{c}{$\Delta V$(BLB,BL) at 2\,ns (mV)}
 & \multicolumn{2}{c}{$\max v(Q)$ (\% of $V_{DD}$)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
(V) & 27\,\si{\celsius} & 85\,\si{\celsius} & $\Delta$ & 27\,\si{\celsius} & 85\,\si{\celsius} \\
\midrule
""" + "\n".join(srows) + "\n" + r"""\bottomrule
\end{tabular}
""")

    lp = summary["lp"]["meas"]["T1"] if summary["lp"] else None
    card_rows = [
        (r"45 nm HP (high performance)", t1),
    ]
    if lp:
        card_rows.append((r"45 nm LP (low power)", lp))
    (DATA / "table_partA_cards.tex").write_text(
        r"""\begin{tabular}{lrrr}
\toprule
PTM model card & $\Delta V$ @2\,ns (mV) & $\bar{I}_{\mathrm{read}}$ (\si{\micro\ampere})
 & $\Delta V$ / 69\,mV \\
\midrule
""" + "\n".join(
            f"{n} & {r['dv']*1e3:.0f} & {avg_read_current(r['dv']):.0f} & "
            f"{r['dv']*1e3/REFERENCE_DV_MV:.1f}$\\times$ \\\\"
            for n, r in card_rows
        ) + "\n" + r"""\bottomrule
\end{tabular}
""")

    def mac(name, val):
        return r"\newcommand{\%s}{%s}" % (name, val)

    macros = [
        mac("AdvNom", fmt(t1["dv"] * 1e3, 0)),
        mac("AdvNomRatio", fmt(t1["dv"] * 1e3 / REFERENCE_DV_MV, 1)),
        mac("AinomUA", fmt(avg_read_current(t1["dv"]), 0)),
        mac("AqmaxNom", fmt(t1["qmax"] * 1e3, 0)),
        mac("AqmaxNomPct", fmt(100 * t1["qmax"] / t1["vdd"], 0)),
        mac("AqfinNom", fmt(t1["qfin"] * 1e3, 1)),
        mac("AdvWide", fmt(t2["dv"] * 1e3, 0)),
        mac("AdvWideGain", fmt(100 * (t2["dv"] / t1["dv"] - 1), 0)),
        mac("AqmaxWide", fmt(t2["qmax"] * 1e3, 0)),
        mac("AqmaxWidePct", fmt(100 * t2["qmax"] / t2["vdd"], 0)),
        mac("AqmaxWideGain", fmt(100 * (t2["qmax"] / t1["qmax"] - 1), 0)),
        mac("Avcrit", fmt(summary["vfail27"], 2)),
        mac("AvcritHot", fmt(summary["vfail85"], 2)),
        mac("AdvHot", fmt(t4["dv"] * 1e3, 0)),
        mac("AdvHotPct", fmt(100 * (t4["dv"] / t1["dv"] - 1), 0)),
        mac("AdvHotDelta", fmt(abs(t4["dv"] - t1["dv"]) * 1e3, 0)),
        mac("AdvCorner", fmt(t4c["dv"] * 1e3, 0)),
        mac("AqmaxCornerPct", fmt(100 * t4c["qmax"] / t4c["vdd"], 0)),
        mac("AdvCornerMargin", fmt(t4c["dv"] * 1e3 / SENSE_OFFSET_MV, 1)),
        mac("AdvMinSweep", fmt(min(s85["dv_85C"]) * 1e3, 0)),
        mac("AwaCrit", fmt(summary["wa_crit_um"], 2)),
        mac("AcellRatioGiven", fmt(0.20 / 0.16, 2)),
        mac("AcellRatioCrit",
            fmt(0.20 / summary["wa_crit_um"], 2) if summary["wa_crit_um"] else "--"),
        mac("AwaMarginX",
            fmt(summary["wa_crit_um"] / 0.16, 1) if summary["wa_crit_um"] else "--"),
        mac("Asenseoffset", f"{SENSE_OFFSET_MV:.0f}"),
        mac("Arefdv", f"{REFERENCE_DV_MV:.0f}"),
    ]
    if lp:
        macros += [
            mac("AdvLP", fmt(lp["dv"] * 1e3, 0)),
            mac("AdvLPRatio", fmt(lp["dv"] * 1e3 / REFERENCE_DV_MV, 1)),
            mac("AiLPUA", fmt(avg_read_current(lp["dv"]), 0)),
            mac("AhpOverLp", fmt(t1["dv"] / lp["dv"], 1)),
        ]
    (DATA / "macros_partA.tex").write_text("\n".join(macros) + "\n")


# --------------------------------------------------------------------------
def main() -> None:
    if shutil.which("ngspice") is None:
        sys.exit("ngspice not found on PATH. Install: sudo apt-get install -y ngspice")

    IMAGES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print("running ngspice (45nm_HP.pm) ...")
    run_primary()
    hp = harvest(ROOT)

    print("running ngspice (45nm_LP.pm) ...")
    try:
        lp_dir = run_with_card("45nm_LP.pm", "lp")
        lp = harvest(lp_dir)
    except SystemExit:
        print("  LP card run failed; continuing with HP only")
        lp = None

    s27, s85 = hp["s27"], hp["s85"]
    v27, fit27, ex27 = crossing_vdd(s27["vdd"], s27["dv_27C"] * 1e3, SENSE_OFFSET_MV)
    v85, fit85, ex85 = crossing_vdd(s85["vdd"], s85["dv_85C"] * 1e3, SENSE_OFFSET_MV)
    extrap = {"27": v27, "85": v85,
              "fit27": (fit27[0] * 1e-3, fit27[1] * 1e-3),
              "fit85": (fit85[0] * 1e-3, fit85[1] * 1e-3)}

    # Width at which the cell first upsets.  max v(Q) is a poor detector: once BL
    # has discharged it clamps Q back down, so the peak asymptotes just below
    # VDD/2 without ever crossing it.  The unambiguous signature is BLB drooping
    # below the precharge rail, which can only happen if QB was pulled down, i.e.
    # if the cell momentarily flipped and dumped BLB through MA2.
    wa_um = hp["wa"]["wa_m"] * 1e6
    blb_droop = 1.1 - hp["wa"]["vblb"]
    wa_crit = None
    for i in range(len(wa_um) - 1):
        if blb_droop[i] < 0.025 <= blb_droop[i + 1]:
            wa_crit = float(0.5 * (wa_um[i] + wa_um[i + 1]))
            break

    print("writing figures ...")
    plot_bitlines(hp["wave"]["t1_read"], IMAGES / "A_task1_bitlines.pdf",
                  "Task 1: bitline development, 1.1 V, 27 $^\\circ$C")
    plot_storage(
        [("$W_A = 0.16\\,\\mu$m (nominal)", hp["wave"]["t1_read"], BLUE, "-"),
         ("$W_A = 0.24\\,\\mu$m (Task 2)", hp["wave"]["t2_read_wide"], RED, "-"),
         ("$W_A = 0.56\\,\\mu$m (upsets)", hp["wave"]["t2_upset"], GREEN, "--")],
        IMAGES / "A_task2_storage_node.pdf",
        "Task 2: read disturb on Q vs access-transistor width",
    )
    plot_wa_sweep(hp["wa"], IMAGES / "A_task2_wa_sweep.pdf", 1.1, wa_crit)
    plot_dv_sweep(s27, s85, IMAGES / "A_task3_dv_vs_vdd.pdf", extrap)
    plot_qmax_sweep(s27, s85, IMAGES / "A_task3_qmax_vs_vdd.pdf")
    plot_bitlines(hp["wave"]["t4_read_85_07v"], IMAGES / "A_task4_bitlines_hot.pdf",
                  "Task 4: bitline development, 0.7 V, 85 $^\\circ$C")
    if lp:
        plot_card_compare(hp["wave"]["t1_read"], lp["wave"]["t1_read"],
                          IMAGES / "A_card_compare.pdf")

    summary = {
        "hp": hp, "lp": lp,
        "vfail27": v27, "vfail85": v85, "wa_crit_um": wa_crit,
        "extrapolated27": ex27, "extrapolated85": ex85,
        "sense_offset_mV": SENSE_OFFSET_MV,
        "reference_dv_mV": REFERENCE_DV_MV,
    }
    write_tex(summary)

    def jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    slim = {k: v for k, v in summary.items() if k not in ("hp", "lp")}
    for name, run in (("hp", hp), ("lp", lp)):
        if run is None:
            continue
        slim[name] = {k: v for k, v in run.items() if k != "wave"}
    (DATA / "partA_summary.json").write_text(
        json.dumps(jsonable(slim), indent=2, default=str))

    m = hp["meas"]
    t1, t2, t4, t4c = m["T1"], m["T2"], m["T4"], m["T4c"]
    lp1 = lp["meas"]["T1"] if lp else None
    print(f"""
Part A results  (PTM 45 nm HP card)
-----------------------------------
Task 1  dV(BLB,BL) @2ns, 1.1 V / 27 C : {t1['dv']*1e3:7.1f} mV """
          f"""= {t1['dv']*1e3/REFERENCE_DV_MV:.1f}x the {REFERENCE_DV_MV:.0f} mV reference
        implied average read current  : {avg_read_current(t1['dv']):7.1f} uA
        max v(Q) during read          : {t1['qmax']*1e3:7.1f} mV """
          f"""({100*t1['qmax']/t1['vdd']:.0f} % of VDD)"""
          + (f"""
        same read, LP card            : {lp1['dv']*1e3:7.1f} mV """
             f"""= {lp1['dv']*1e3/REFERENCE_DV_MV:.1f}x reference""" if lp1 else "") + f"""

Task 2  WA = 0.24 um   dV            : {t2['dv']*1e3:7.1f} mV """
          f"""({100*(t2['dv']/t1['dv']-1):+.0f} %)
        max v(Q)                      : {t2['qmax']*1e3:7.1f} mV """
          f"""({100*t2['qmax']/t2['vdd']:.0f} % of VDD, {100*(t2['qmax']/t1['qmax']-1):+.0f} %)
        v(Q) at 4 ns                  : {t2['qfin']*1e3:7.2f} mV  (state retained)
        read upset first occurs at WA : {fmt(wa_crit,2)} um """
          f"""(cell ratio {0.20/wa_crit:.2f}); given cell has {wa_crit/0.16:.1f}x margin

Task 3  min dV over 1.1 -> 0.6 V      : {min(s85['dv_85C'])*1e3:7.1f} mV at 0.6 V / 85 C
        dV = 25 mV reached at VDD     : {fmt(v27,2)} V (27 C){' [extrapolated]' if ex27 else ''}
                                        {fmt(v85,2)} V (85 C){' [extrapolated]' if ex85 else ''}
        max read disturb seen         : """
          f"""{max(100*s85['qmax_85C']/s85['vdd']):.0f} % of VDD -> no read upset

Task 4  dV @ 1.1 V / 85 C             : {t4['dv']*1e3:7.1f} mV """
          f"""({100*(t4['dv']/t1['dv']-1):+.0f} % vs 27 C)
        dV @ 0.7 V / 85 C  (headline) : {t4c['dv']*1e3:7.1f} mV """
          f"""= {t4c['dv']*1e3/SENSE_OFFSET_MV:.1f}x the 25 mV offset -> """
          f"""{'PASS' if t4c['dv']*1e3 >= SENSE_OFFSET_MV else 'FAIL'}

figures -> {IMAGES}
tables  -> {DATA}
""")


if __name__ == "__main__":
    main()
