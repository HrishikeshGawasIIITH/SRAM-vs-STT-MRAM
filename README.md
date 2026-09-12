# AMCAS Assignment 1: Simulating Memory, Devices to Systems

ECE2.414, IIIT Hyderabad. Should the 2 MB L2 cache in an accelerator be SRAM or
STT-MRAM? The question is carried through ngspice, CACTI, NVSim, Ramulator 2.0
and gem5.

## Tools

ngspice 36 from the distribution packages. CACTI 7, NVSim, Ramulator 2.0a and
gem5 v24 are cloned and built from source; each lives in its own directory
here. GAPBS provides the `bfs` and `sssp` kernels, built static and serial.

## Running

One script per part. Each one runs its tool, parses the output, and writes
every figure, table and number used in the report into `report/`.

```bash
python3 run_partA.py    # ngspice            -> Part A
python3 run_partBC.py   # CACTI, NVSim       -> Parts B and C
python3 gen_l2trace.py  # gem5 CommMonitor   -> partD/l2miss.trace
python3 run_partD.py    # Ramulator          -> Part D
python3 run_partE.py    # gem5               -> Part E
```

Part D needs the trace from `gen_l2trace.py`, and Part E needs the scale-18
graphs plus the atomic instruction counts that set its warmup points:

```bash
cd gapbs && ./converter -g 18 -b graph18.sg && ./converter -g 18 -w -b graph18.wsg
```

Then build the report:

```bash
cd report && pdflatex main && pdflatex main
./make_report_zip.sh     # report/ packaged for Overleaf
```

Needs `numpy`, `matplotlib`, `pyyaml` for the drivers, and `siunitx`,
`booktabs`, `titlesec`, `microtype`, `caption`, `xcolor`, `listings`,
`hyperref`, `geometry` for the document.

## Files

| Path | What it is |
|---|---|
| `A.sp` | Part A netlist, all four tasks in one `.control` block |
| `45nm_HP.pm`, `45nm_LP.pm` | PTM 45 nm BSIM4 model cards |
| `partB/cache.cfg` | CACTI config for the 2 MB SRAM baseline |
| `partC/sample.cell` | STT-MRAM bitcell |
| `partC/STT_cache.cfg`, `partC/STT_cache_8MB.cfg` | NVSim configs, 2 MB and 8 MB |
| `partD/ddr4.yaml` | Ramulator config |
| `partD/l2miss.trace` | L2 miss stream from gem5 |
| `partD/decode_l2trace.py` | converts gem5's packet trace to Ramulator format |
| `gem5_l2.py` | gem5 config with a CommMonitor on the L2, used for the trace |
| `run_part*.py`, `gen_l2trace.py` | per-part drivers |
| `make_report_zip.sh` | packages `report/` for Overleaf |
| `report/main.tex`, `report/sections/` | the document |
| `report/images/`, `report/data/` | generated figures, tables and numbers |
| `work/` | raw tool output and logs, regenerated on each run |

Parts D and E share one workload: GAPBS `bfs` and `sssp` over a scale-18
Kronecker graph, 262,143 nodes and 3.8 M undirected edges, built beforehand and
loaded from file. Part E fast-forwards atomically through setup and the first
trial, then measures the second trial on the detailed core.
