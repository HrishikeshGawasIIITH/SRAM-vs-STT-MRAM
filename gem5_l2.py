"""
gem5 SE config for Parts D and E: one core, L1I/L1D, a configurable L2, DDR4.

Why not configs/deprecated/example/se.py, which the assignment names?  Two
reasons, both hard blockers in gem5 v24:
  * se.py exposes --l2_size and --l2_assoc but has NO L2 hit-latency option,
    and --l2-hit-latency is the single most important knob in Part E.
  * se.py cannot insert a CommMonitor, which Part D needs to capture the L2
    miss stream.
Everything else (cache sizes, associativity, DDR4 memory, SE mode) matches the
assignment's command line.

  # Part E baseline (SRAM L2 from CACTI)
  gem5.opt -d m5out/x gem5_l2.py --cpu-type O3 --l2_size 2MB --l2_assoc 8 \
      --l2-hit-latency 7 --cmd gapbs/bfs --options "-g 14 -n 1" --maxinsts 5e7

  # Part D (add a monitor on the L2 memory-side port)
  gem5.opt gem5_l2.py --cpu-type Timing --trace-file work/d/l2miss.trc.gz ...
"""
import argparse

import m5
from m5.objects import *

p = argparse.ArgumentParser()
p.add_argument("--cmd", required=True)
p.add_argument("--options", default="")
p.add_argument("--cpu-type", dest="cpu_type", default="O3",
               choices=["O3", "Timing"])
p.add_argument("--l1d_size", default="32kB")
p.add_argument("--l1i_size", default="32kB")
p.add_argument("--l2_size", default="2MB")
p.add_argument("--l2_assoc", type=int, default=8)
p.add_argument("--l2-hit-latency", dest="l2_lat", type=int, default=7,
               help="L2 tag+data latency in CPU cycles (from CACTI/NVSim)")
p.add_argument("--mem_size", default="4GB")
p.add_argument("--clock", default="2GHz")
p.add_argument("--maxinsts", type=float, default=0)
p.add_argument("--trace-file", dest="trace_file", default="")
args = p.parse_args()

system = System()
system.clk_domain = SrcClockDomain(clock=args.clock,
                                   voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange(args.mem_size)]

system.cpu = X86O3CPU() if args.cpu_type == "O3" else X86TimingSimpleCPU()

system.membus = SystemXBar()
system.l2bus = L2XBar()

system.cpu.icache = Cache(size=args.l1i_size, assoc=8, tag_latency=2,
                          data_latency=2, response_latency=2,
                          mshrs=16, tgts_per_mshr=20)
system.cpu.dcache = Cache(size=args.l1d_size, assoc=8, tag_latency=2,
                          data_latency=2, response_latency=2,
                          mshrs=16, tgts_per_mshr=20)
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

# The L2 hit latency is the number Parts B and C produce.
system.l2cache = Cache(size=args.l2_size, assoc=args.l2_assoc,
                       tag_latency=args.l2_lat, data_latency=args.l2_lat,
                       response_latency=args.l2_lat,
                       mshrs=32, tgts_per_mshr=12)
system.l2cache.cpu_side = system.l2bus.mem_side_ports

if args.trace_file:
    # Everything the L2 sends memory-side is a miss or a writeback.
    system.l2mon = CommMonitor()
    system.l2mon.probes = [MemTraceProbe(trace_file=args.trace_file)]
    system.l2cache.mem_side = system.l2mon.cpu_side_port
    system.l2mon.mem_side_port = system.membus.cpu_side_ports
else:
    system.l2cache.mem_side = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR4_2400_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

process = Process()
process.cmd = [args.cmd] + (args.options.split() if args.options else [])
system.cpu.workload = process
system.cpu.createThreads()
if args.maxinsts:
    system.cpu.max_insts_any_thread = int(args.maxinsts)

system.workload = SEWorkload.init_compatible(args.cmd)

root = Root(full_system=False, system=system)
m5.instantiate()
print(f"[gem5_l2] {args.cpu_type} L2={args.l2_size} lat={args.l2_lat} "
      f"cmd={args.cmd} {args.options}")
ev = m5.simulate()
print(f"[gem5_l2] exit @ {m5.curTick()} : {ev.getCause()}")
