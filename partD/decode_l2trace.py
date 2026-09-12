#!/usr/bin/env python3
"""
Decode a gem5 CommMonitor packet trace into a Ramulator LoadStoreTrace.

gem5's own util/decode_packet_trace.py collapses every command except ReadReq
and WriteReq to the letter 'u'.  An L2 miss stream is almost entirely
ReadSharedReq / ReadExReq / WritebackDirty, so that decoder throws away exactly
the read/write distinction Part D needs.  This one keeps the numeric MemCmd and
classifies it against the enum in src/mem/packet.hh.

  python3 partD/decode_l2trace.py <trace.trc.gz> <out.trace> [max_lines]
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UTIL = ROOT / "gem5" / "util"
sys.path.insert(0, str(UTIL))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

subprocess.check_call(["make", "--quiet", "-C", str(UTIL), "packet_pb2.py"])
import packet_pb2  # noqa: E402
import protolib  # noqa: E402

# src/mem/packet.hh, MemCmd::Command, in declaration order.
MEMCMD = """InvalidCmd ReadReq ReadResp ReadRespWithInvalidate WriteReq
WriteResp WriteCompleteResp WritebackDirty WritebackClean WriteClean
CleanEvict SoftPFReq SoftPFExReq HardPFReq SoftPFResp HardPFResp WriteLineReq
UpgradeReq SCUpgradeReq UpgradeResp SCUpgradeFailReq UpgradeFailResp ReadExReq
ReadExResp ReadCleanReq ReadSharedReq LoadLockedReq StoreCondReq
StoreCondFailReq StoreCondResp LockedRMWReadReq LockedRMWReadResp
LockedRMWWriteReq LockedRMWWriteResp SwapReq SwapResp MemSyncReq MemSyncResp
MemFenceResp CleanSharedReq CleanSharedResp CleanInvalidReq""".split()

# Commands that move data down to DRAM.
WRITE_CMDS = {"WritebackDirty", "WriteClean", "WriteReq", "WriteLineReq"}
# CleanEvict only tells the level below that a clean line was dropped; with a
# plain memory underneath it generates no DRAM traffic, so it is not replayed.
DROP_CMDS = {"CleanEvict"}


def classify(cmd_id: int):
    """-> 'ST', 'LD', or None if the command generates no DRAM traffic."""
    name = MEMCMD[cmd_id] if 0 <= cmd_id < len(MEMCMD) else ""
    if name in DROP_CMDS:
        return None
    return "ST" if name in WRITE_CMDS else "LD"


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <in.trc.gz> <out.trace> [max_lines]")
    src, dst = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    proto_in = protolib.openFileRd(src)
    if proto_in.read(4).decode() != "gem5":
        sys.exit(f"{src} is not a gem5 protobuf trace")

    from packet_pb2 import PacketHeader
    header = PacketHeader()
    protolib.decodeMessage(proto_in, header)

    pkt = packet_pb2.Packet()
    n = reads = writes = 0
    seen = {}
    with open(dst, "w") as out:
        while protolib.decodeMessage(proto_in, pkt):
            name = MEMCMD[pkt.cmd] if pkt.cmd < len(MEMCMD) else str(pkt.cmd)
            seen[name] = seen.get(name, 0) + 1
            kind = classify(pkt.cmd)
            if kind is None:
                continue
            out.write(f"{kind} 0x{pkt.addr:x}\n")
            reads += kind == "LD"
            writes += kind == "ST"
            n += 1
            if cap and n >= cap:
                break
    print(f"{dst}: {n} accesses ({reads} LD, {writes} ST)")
    print("command mix:", ", ".join(f"{k}={v}" for k, v in
                                    sorted(seen.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
