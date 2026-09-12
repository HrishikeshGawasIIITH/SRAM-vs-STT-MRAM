* AMCAS Assignment 1 - Part A : 6T SRAM read, bitline discharge and read margin
* 45 nm PTM BSIM4 (High Performance card).  Stored state: Q = 0, QB = 1.
*
* Run with:  ngspice -b A.sp
* All numeric results and waveforms are written into ./work/ as ASCII.
*
* NOTE ON INITIALISATION
* ---------------------
* Stored state comes from ".ic v(q)=0".  That is deliberately the only initial
* condition: v(q)=0 is supply-independent, so it stays correct at every point of
* the Task 3 VDD sweep, and the cross-coupled pair pulls QB up to whatever rail
* is applied by itself.  The assignment's extra ".ic v(qb)='VDD'" and the
* capacitor "IC='VBL'" values would be frozen at their parse-time VDD and would
* silently mis-initialise every altered-VDD run, so they are dropped and the
* bitlines are precharged by a real circuit instead (MPC1/MPC2 + equaliser MEQ).
* "uic" is therefore NOT used: each tran solves its own operating point with
* v(q) held at 0, which is exactly the state we want to read.

.include 45nm_HP.pm

.param VDD=1.1
.param WA=0.16u              $ access-transistor width (Task 2 alters this)
.param WPC=1.2u              $ precharge PMOS width
.param WEQ=0.6u              $ equaliser PMOS width

* ---------------- supplies and wordline ----------------
* The wordline must track the supply.  A literal PWL(... 'VDD') would freeze the
* WL high level at its parse-time value, so every altered-VDD run in Task 3 would
* be done with an overdriven wordline.  Shape the pulse 0 -> 1 and scale it by the
* live rail voltage instead.
Vdd  vdd 0 'VDD'
Vwls wls 0 PWL(0 0 1n 0 1.05n 1)
Bwl  wl  0 V = v(wls)*v(vdd)

* ---------------- 6T bitcell ----------------
MP1 qb q  vdd vdd pmos W=0.15u L=0.045u
MN1 qb q  0   0   nmos W=0.20u L=0.045u
MP2 q  qb vdd vdd pmos W=0.15u L=0.045u
MN2 q  qb 0   0   nmos W=0.20u L=0.045u
MA1 bl  wl q  0   nmos W='WA' L=0.045u
MA2 blb wl qb 0   nmos W='WA' L=0.045u

* ---------------- bitline capacitance ----------------
Cbl  bl  0 180f
Cblb blb 0 180f

* ---------------- bitline precharge and equalise ----------------
* Active-low precharge: PRE sits at 0 (devices on) until t = 0.7 ns, then rises
* to the rail by 0.75 ns, leaving the bitlines floating at VDD a full 0.25 ns
* before the wordline asserts at t = 1.0 ns.  PRE has to swing to the *live*
* rail to switch the PMOS off, so it is shaped 0 -> 1 and scaled by v(vdd) for
* the same reason the wordline is.
Vpres pres 0 PWL(0 0 0.70n 0 0.75n 1)
Bpre  pre  0 V = v(pres)*v(vdd)

MPC1 bl  pre vdd vdd pmos W='WPC' L=0.045u   $ pull BL  up to VDD
MPC2 blb pre vdd vdd pmos W='WPC' L=0.045u   $ pull BLB up to VDD
MEQ  bl  pre blb vdd pmos W='WEQ' L=0.045u   $ equalise BL and BLB

* ---------------- stored state ----------------
* Held only during the operating-point solve, then released.
.ic v(q)=0

.control
set filetype = ascii
set wr_singlescale
set wr_vecnames

* ================= TASK 1 : nominal read, VDD = 1.1 V, 27 C =================
option temp = 27
alter vdd = 1.1
tran 5p 4n
meas tran vblb  FIND v(blb) AT=2.0n
meas tran vbl   FIND v(bl)  AT=2.0n
let dv = vblb - vbl
meas tran qmax  MAX  v(q)  FROM=1n TO=4n
meas tran qfin  FIND v(q)  AT=4n
wrdata work/t1_read.data v(bl) v(blb) v(q) v(qb) v(wl)
echo "task temp_C wa_um vdd dv qmax qfin vbl2" > work/t1_t2_meas.data
echo "T1 27 0.16 1.1 $&dv $&qmax $&qfin $&vbl" >> work/t1_t2_meas.data

* ================= TASK 2 : grow access transistors to W = 0.24 um =========
alter @ma1[w] = 0.24u
alter @ma2[w] = 0.24u
tran 5p 4n
meas tran vblb  FIND v(blb) AT=2.0n
meas tran vbl   FIND v(bl)  AT=2.0n
let dv = vblb - vbl
meas tran qmax  MAX  v(q)  FROM=1n TO=4n
meas tran qfin  FIND v(q)  AT=4n
wrdata work/t2_read_wide.data v(bl) v(blb) v(q) v(qb) v(wl)
echo "T2 27 0.24 1.1 $&dv $&qmax $&qfin $&vbl" >> work/t1_t2_meas.data

* --- Task 2 extension: the given cell ratio does NOT fail, so find the access
* --- width that actually upsets the cell.  Note max v(Q) is a poor detector: once
* --- BL has discharged it clamps Q back down, so the peak asymptotes just below
* --- VDD/2 without crossing it.  The clean signature is BLB drooping below the
* --- precharge rail, which needs QB to have been pulled down, i.e. a real flip.
* --- vblb is recorded here for exactly that test.  Widths are in metres.
echo "wa_m dv qmax qfin vblb" > work/t2_wa_sweep_27.data
foreach wa 0.16e-6 0.20e-6 0.24e-6 0.28e-6 0.32e-6 0.36e-6 0.40e-6 0.44e-6 0.48e-6 0.50e-6 0.52e-6 0.54e-6 0.56e-6 0.60e-6 0.64e-6 0.70e-6
  alter @ma1[w] = $wa
  alter @ma2[w] = $wa
  tran 5p 4n
  meas tran vblb FIND v(blb) AT=2.0n
  meas tran vbl  FIND v(bl)  AT=2.0n
  let dv = vblb - vbl
  meas tran qmax MAX  v(q) FROM=1n TO=4n
  meas tran qfin FIND v(q) AT=4n
  echo "$wa $&dv $&qmax $&qfin $&vblb" >> work/t2_wa_sweep_27.data
end

* capture the first upsetting width as the Task 2 failure waveform
alter @ma1[w] = 0.56e-6
alter @ma2[w] = 0.56e-6
tran 5p 4n
wrdata work/t2_upset.data v(bl) v(blb) v(q) v(qb) v(wl)

* restore nominal conditions for the remaining tasks
alter @ma1[w] = 0.16u
alter @ma2[w] = 0.16u

* ================= TASK 3 : VDD sweep, 1.1 V -> 0.6 V in 50 mV steps =======
echo "vdd dv_27C qmax_27C qfin_27C" > work/t3_sweep_27.data
foreach vd 1.10 1.05 1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60
  alter vdd = $vd
  tran 5p 4n
  meas tran vblb FIND v(blb) AT=2.0n
  meas tran vbl  FIND v(bl)  AT=2.0n
  let dv = vblb - vbl
  meas tran qmax MAX  v(q) FROM=1n TO=4n
  meas tran qfin FIND v(q) AT=4n
  echo "$vd $&dv $&qmax $&qfin" >> work/t3_sweep_27.data
end

* ================= TASK 4 : repeat at 85 C =================================
option temp = 85

alter vdd = 1.1
tran 5p 4n
meas tran vblb  FIND v(blb) AT=2.0n
meas tran vbl   FIND v(bl)  AT=2.0n
let dv = vblb - vbl
meas tran qmax  MAX  v(q)  FROM=1n TO=4n
meas tran qfin  FIND v(q)  AT=4n
wrdata work/t4_read_85.data v(bl) v(blb) v(q) v(qb) v(wl)
echo "task temp_C wa_um vdd dv qmax qfin vbl2" > work/t4_meas.data
echo "T4 85 0.16 1.1 $&dv $&qmax $&qfin $&vbl" >> work/t4_meas.data

* 85 C VDD sweep, so the Task 3 margin plot can be drawn at both corners
echo "vdd dv_85C qmax_85C qfin_85C" > work/t3_sweep_85.data
foreach vd 1.10 1.05 1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60
  alter vdd = $vd
  tran 5p 4n
  meas tran vblb FIND v(blb) AT=2.0n
  meas tran vbl  FIND v(bl)  AT=2.0n
  let dv = vblb - vbl
  meas tran qmax MAX  v(q) FROM=1n TO=4n
  meas tran qfin FIND v(q) AT=4n
  echo "$vd $&dv $&qmax $&qfin" >> work/t3_sweep_85.data
end

* the headline corner asked for in the central question: 0.7 V, 85 C
alter vdd = 0.7
tran 5p 4n
meas tran vblb  FIND v(blb) AT=2.0n
meas tran vbl   FIND v(bl)  AT=2.0n
let dv = vblb - vbl
meas tran qmax  MAX  v(q)  FROM=1n TO=4n
meas tran qfin  FIND v(q)  AT=4n
wrdata work/t4_read_85_07v.data v(bl) v(blb) v(q) v(qb) v(wl)
echo "T4c 85 0.16 0.7 $&dv $&qmax $&qfin $&vbl" >> work/t4_meas.data

quit
.endc
.end
