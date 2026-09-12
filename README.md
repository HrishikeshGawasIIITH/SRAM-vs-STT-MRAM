# AMCAS Assignment 1 — Simulating Memory: Devices to Systems

ECE2.414, IIIT Hyderabad. Reproducible sources for the report.

<https://github.com/HrishikeshGawasIIITH/SRAM-vs-STT-MRAM>

One script per part. Each runs its tool, parses the output, and regenerates
**every** figure, table and in-prose number into `report/`. Nothing in the
report is typed by hand.

```bash
python3 run_partA.py    # ngspice   -> Part A
python3 run_partBC.py   # CACTI     -> Part B, NVSim -> Part C
python3 gen_l2trace.py  # gem5 CommMonitor -> partD/l2miss.trace
python3 run_partD.py    # Ramulator -> Part D
python3 run_partE.py    # gem5      -> Part E
cd report && pdflatex main && pdflatex main   # -> AMCAS_A1_report.pdf
./make_report_zip.sh                          # -> AMCAS_A1_report.zip (Overleaf)
```

No root needed for the report: TeX was installed with TinyTeX into
`~/.TinyTeX`, then `tlmgr install siunitx booktabs titlesec microtype lm
caption xcolor listings hyperref geometry`.

## Layout

| Path | What it is |
|---|---|
| `A.sp`, `45nm_*.pm` | Part A netlist + PTM 45 nm BSIM4 cards |
| `partB/cache.cfg` | CACTI config (assignment spec) |
| `partC/sample.cell`, `partC/STT_cache*.cfg` | NVSim STT-MRAM cell + configs |
| `partD/ddr4.yaml`, `partD/l2miss.trace` | Ramulator config + gem5-derived trace |
| `partD/decode_l2trace.py` | packet-trace decoder (see below) |
| `gem5_l2.py` | shared gem5 SE config for Parts D and E |
| `report/` | Overleaf-ready LaTeX project — zip and upload |
| `work/` | raw tool output, logs, PNG previews (regenerated) |

## Build notes

Everything below was needed to get the five tools running on Ubuntu 22.04 /
WSL2, GCC 11.

**NVSim** fails to compile: `reference to 'data' is ambiguous`. It is pre-C++17
code, and GCC 11 defaults to `gnu++17` where `std::data` collides with NVSim's
`MemoryType data` enum under `using namespace std`. Fix — pin an older standard:

```make
CXXFLAGS := -Wall -std=c++11
```

**Ramulator 2.0a** — `make` fails on a `clang-format` codegen target
(`unknown key 'InsertBraces'`) when the installed clang-format is older than the
one the `.clang-format` was written for. Delete that one line; it does not
affect the simulator. Ramulator also ships **no FCFS scheduler**, which Part D
Task 2 requires, so one was added at
`src/dram_controller/impl/scheduler/fcfs_scheduler.cpp` (~40 lines: identical to
FRFCFS minus the readiness check, giving true head-of-line blocking).

**gem5** — two problems on a 7 GB WSL box:
* The SCons zlib probe fails even with zlib present, because it executes the
  conftest binary through `/bin/sh`. `SConstruct` is patched to honour
  `GEM5_LOCAL_PREFIX` for include/lib paths and `GEM5_ASSUME_ZLIB=1` to link
  `-lz` directly.
* The final link is OOM-killed by `ld.bfd`. Use lld:

```bash
GEM5_LOCAL_PREFIX=$HOME/.local GEM5_ASSUME_ZLIB=1 \
  scons build/X86/gem5.opt --linker=lld --limit-ld-memory-usage -j4
```

**Packet trace decoding** — gem5's `util/decode_packet_trace.py` collapses every
`MemCmd` except `ReadReq`/`WriteReq` to the letter `u`. An L2 miss stream is
almost entirely `ReadSharedReq`/`ReadExReq`/`WritebackDirty`, so that decoder
discards the read/write distinction Part D needs.
`partD/decode_l2trace.py` keeps the numeric command and classifies it against
the enum in `src/mem/packet.hh`. It also needs
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, because the shipped `*_pb2.py`
predate the installed protobuf runtime.

## Deviations from the assignment text

Each is forced by the tools, and each is stated in the report where it matters.

* **Part A** — the given netlist's `.ic v(qb)='VDD'`, its capacitor `IC='VBL'`
  values and its `PWL(... 'VDD')` wordline are all resolved at parse time, so
  none of them follows `alter vdd` in the Task 3 sweep. Kept the one initial
  condition that *is* supply-independent (`.ic v(q)=0`), let the latch establish
  QB, precharged the bitlines with a real active-low PMOS precharge + equaliser,
  and scaled WL/PRE by `v(vdd)`. No `uic`. Also
  `meas ... FIND v(blb)-v(bl)` is rejected by ngspice 36.
* **Part D** — the assignment's `ddr4.yaml` pairs `Frontend: SimpleO3` with a
  one-access-per-line memory trace; SimpleO3 consumes an *instruction* trace.
  Uses `LoadStoreTrace`, which takes that format and routes byte addresses
  through the address mapper (Task 3 needs this). `ReadWriteTrace` cannot be
  used — it takes pre-decoded address vectors and its `is_finished()` is a
  stub.
* **Part E** — `configs/deprecated/example/se.py` in gem5 v24 has **no**
  `--l2-hit-latency` option and cannot host a CommMonitor. `gem5_l2.py`
  provides both; cache sizes, associativity and memory match the assignment's
  command line. L2 hit latencies are taken from our own Parts B/C
  (6 cycles SRAM, 7 cycles MRAM) rather than the assignment's assumed 7/14; the
  14-cycle case is still run as a sensitivity check.
* **Parts D/E workload** — GAPBS graphs are pre-built with `converter` and
  loaded with `-f`. Generating the Kronecker graph *inside* gem5 spends the
  whole simulation in a compute-bound generator with a tiny working set, which
  yields a 99.9\% cold-miss L2 profile that measures nothing about traversal.

## Known tool bug worth flagging

Ramulator's `avg_read_latency_N` divides accumulated latency by
`num_read_reqs_N`, which `send()` increments on **every** call — including calls
that are rejected because the queue is full and the frontend retries. The
denominator is inflated, and the reported average comes out *below*
tRCD + tCL, which is physically impossible. `run_partD.py` divides by the
authoritative `total_num_read_requests` instead.
