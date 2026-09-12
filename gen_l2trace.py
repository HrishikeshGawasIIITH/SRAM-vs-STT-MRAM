#!/usr/bin/env python3
"""
Generate partD/l2miss.trace from gem5's CommMonitor packet trace.

  1. run gem5 with partD/l2mon_se.py  -> work/d/l2miss.trc.gz (protobuf)
  2. decode with gem5/util/decode_packet_trace.py
  3. rewrite as Ramulator LoadStoreTrace lines: "LD <addr>" / "ST <addr>"

    python3 gen_l2trace.py [--kernel bfs] [--graph 14] [--maxinsts 30000000]
"""
from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEM5 = ROOT / "gem5" / "build" / "X86" / "gem5.opt"
WORK = ROOT / "work" / "d"
OUT = ROOT / "partD" / "l2miss.trace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="bfs")
    ap.add_argument("--graph-file", dest="graph_file",
                    default="gapbs/graph16.sg")
    ap.add_argument("--maxinsts", type=int, default=0,
                    help="0 = run the workload to completion")
    ap.add_argument("--max-lines", type=int, default=200_000)
    a = ap.parse_args()

    if not GEM5.exists():
        sys.exit(f"gem5 not built at {GEM5}")
    WORK.mkdir(parents=True, exist_ok=True)
    trc = WORK / "l2miss.trc.gz"
    if trc.exists():
        trc.unlink()

    cmd = [str(GEM5), "-d", str(WORK / "m5out_trace"),
           str(ROOT / "gem5_l2.py"),
           "--cpu-type", "Timing",
           "--cmd", str(ROOT / "gapbs" / a.kernel),
           "--options", f"-f {a.graph_file} -n 1",
           "--trace-file", str(trc),
           "--maxinsts", str(a.maxinsts)]
    print("running gem5 to collect the L2 miss stream ...")
    print("  " + " ".join(cmd))
    env = dict(os.environ)
    # gem5 links the locally built zlib
    env["LD_LIBRARY_PATH"] = (str(Path.home() / ".local" / "lib") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    (WORK / "gem5_trace.log").write_text(p.stdout + p.stderr)
    if not trc.exists():
        print((p.stdout + p.stderr)[-3000:])
        sys.exit("gem5 produced no packet trace")

    print("decoding packet trace ...")
    # gem5's own decoder collapses every MemCmd except ReadReq/WriteReq to 'u',
    # which destroys the read/write split in an L2 miss stream.  Use ours.
    d = subprocess.run([sys.executable, str(ROOT / "partD" / "decode_l2trace.py"),
                        str(trc), str(OUT), str(a.max_lines)],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    (WORK / "decode.log").write_text(d.stdout + d.stderr)
    print(d.stdout.strip() or d.stderr[-1500:])
    if not OUT.exists() or OUT.stat().st_size == 0:
        sys.exit("decoding produced no trace")


if __name__ == "__main__":
    main()
