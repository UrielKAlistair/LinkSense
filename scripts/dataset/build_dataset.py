#!/usr/bin/env python3
"""Turn ns-3 run directories into the tabular training table.

One row is one AP as seen from one scan, carrying the throughput the client
actually achieved by joining it. Rows sharing a scan_id are the APs available
from where the client stood:

    scan_id       ap_index   feat_* ...   label_throughput_mbps
    t00007__c01   0          ...           18.4
    t00007__c01   1          ...           31.2   <- best
    t00007__c01   2          ...            9.7

The simulator replayed that one scan once per AP, so every option's outcome
is known and not just the one that was taken. Regression and ranking models
both train on this table - a regressor fits the label per row, a ranker
compares rows within a scan_id - and both are scored by decision quality
within the scan. See models/evaluate.py.

The work of this file is converting what the simulator RECORDED into what a
real client would have KNOWN. The simulator over-observes on purpose: one
scanner radio per channel, all listening for the whole pre-association
window. No commodity station has that, so most of what follows is discarding
information faithfully.
 
INPUT - one run directory per (scan, target AP)
  observation.csv  one decoded frame per row: t (reception END), freq_mhz,
                   bssid, ta, cat (0 mgmt / 1 ctrl / 2 data), is_beacon,
                   retry, len, signal_dbm, noise_dbm, duration_us, rate_mbps.
                   Roughly 8,000 frames across all channels. No payload, and
                   no receiver address.
  chanbusy.csv     NOT frames. The scanner PHY's carrier-sense state per
                   interval per channel: the medium was unavailable, whether
                   or not anything was decodable. Real chipsets expose this.
  metadata.json    run parameters and ground truth.
  result.json      the label.

A scan's runs share a byte-identical observation, so only the first records it.

OUTPUT - column prefixes are a contract
  feat_*   the ONLY columns a model may read
  gt_*     simulator ground truth: for validating the feat_* proxies and for
           slicing results, never as input. models/data.py enforces that.
  label_*  the answer
  meta_*   which observation settings produced this table

Three filters cut the recording back to one radio's worth of attention. They
live here rather than in the simulator so that changing an assumption costs
seconds instead of re-running hours of ns-3:

  1. --single-radio-sweep keeps only frames that fell inside the dwell
     scheduled on their channel, and normalises rates by time actually spent
     listening to that channel rather than by the whole window.
  2. --rssi-noise-db / --rssi-quant-db degrade exact simulator RSSI to what a
     chipset reports, per frame, before any statistic is taken.
  3. An AP whose beacon never decoded is dropped: a client cannot join a BSS
     it never heard of.

Features come in three families. feat_chan_* describes the CHANNEL an option
occupies, feat_ap_* the option's own BSS, and feat_rel_* the option against
its alternatives. The third is a repair rather than a third measurement:
filtering frames down to one AP destroys every cross-option comparison, and
feat_rel_* adds it back by hand. build_frame_corpus.py exists because a model
handed the frames never loses that context to begin with.

The 802.11 BSS Load element - the station count and utilisation an AP
advertises about itself - is absent because the simulator does not emit it.
feat_ap_n_clients reconstructs a noisier version by counting distinct
transmitters, which cannot see a station that only downloads. See
report/open_questions.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import random
import statistics
import csv
import sys
from collections import defaultdict
from pathlib import Path

TYPE_MGMT, TYPE_CTRL, TYPE_DATA = 0, 1, 2

# ns-3's default beacon interval. A passive dwell shorter than this can miss
# an AP entirely.
BEACON_INTERVAL_S = 0.1024

# The channels a client sweeps: a property of the RADIO, not of the deployment.
# Deriving this from the occupied channels would hand the model a ground-truth
# quantity correlated with the label. Must match kApChannels in
# sim/my-wifi-test.cc.
SCAN_CHANNELS = (36, 40, 44, 48)

# What devices ship, not what papers recommend. Linux mac80211
# (net/mac80211/scan.c) uses IEEE80211_PASSIVE_CHANNEL_TIME = HZ/9 ~ 111 ms and
# IEEE80211_CHANNEL_TIME = HZ/33 ~ 30 ms; Qualcomm's Android WLAN config ships
# gPassiveMaxChannelTime=110, gActiveMaxChannelTime=40.
PASSIVE_DWELL_MS = 110.0
ACTIVE_DWELL_MS = 30.0


@dataclasses.dataclass(frozen=True)
class ScanConfig:
    """How much of the simulator's observation a real client would have got.

    The defaults describe a passive scan ending at the moment the client must
    decide, which is what a station holds in its scan cache when it picks an
    AP. Retune time is measured: 72-150 us across eight commodity radios
    (Goovaerts et al., NordSec 2019), a tenth of a percent of the dwell.
    """

    sweep: bool = False
    mode: str = "passive"         # passive (listen for beacons) | active (probe)
    dwell_ms: float | None = None  # None -> the shipping default for the mode
    retune_ms: float = 0.15       # blind time while the synthesiser settles
    probe_retries: int = 4        # unicast retry budget for a probe response
    passes: int = 0               # 0 = as many full passes as the window fits
    order: str = "random"         # channel visit order: random | ascending
    align: str = "end"            # place the sweep at the end | start of window
    rssi_noise_db: float = 3.0    # +/- this much per-frame measurement error
    rssi_noise_model: str = "uniform"   # uniform | gaussian (as a std dev)
    rssi_bias_db: float = 0.0     # per-scan systematic offset, +/- this much
    rssi_quant_db: float = 1.0    # report resolution; 1.0 = whole dBm
    seed: int = 20250817

    @property
    def dwell(self) -> float:
        """Dwell in ms: the explicit setting, else what devices ship."""
        if self.dwell_ms is not None:
            return self.dwell_ms
        return ACTIVE_DWELL_MS if self.mode == "active" else PASSIVE_DWELL_MS

    @property
    def dwell_s(self) -> float:
        return self.dwell / 1000.0

    @property
    def listen_s(self) -> float:
        """Dwell time actually spent receiving, after retuning."""
        return max(0.0, (self.dwell - self.retune_ms) / 1000.0)

    def describe(self) -> str:
        rssi = (f"RSSI +/-{self.rssi_noise_db:g} dB ({self.rssi_noise_model})"
                f" quantised to {self.rssi_quant_db:g} dB"
                if (self.rssi_noise_db or self.rssi_quant_db) else "RSSI raw")
        if self.rssi_bias_db:
            rssi += f", per-scan bias +/-{self.rssi_bias_db:g} dB"
        if not self.sweep:
            return f"all channels observed in parallel for the full window; {rssi}"
        passes = f"{self.passes} pass(es)" if self.passes else "window-filling passes"
        probe = (f", probe responses with {self.probe_retries} retries"
                 if self.mode == "active" else "")
        return (f"single-radio {self.mode} sweep: {self.dwell:g} ms dwell "
                f"({self.retune_ms:g} ms retune){probe}, {passes}, {self.order} order, "
                f"aligned to window {self.align}; {rssi}")


def _rng(cfg: ScanConfig, scan_id: str, stream: str) -> random.Random:
    """Deterministic per-(scan, stream) RNG.

    Seeded from a digest rather than hash(), which is randomised per process
    and would make the worker pool non-reproducible. Noise and sweep
    scheduling draw from separate streams, so changing the dwell does not also
    reshuffle every RSSI sample.
    """
    d = hashlib.blake2b(f"{cfg.seed}:{scan_id}:{stream}".encode(), digest_size=8)
    return random.Random(int.from_bytes(d.digest(), "big"))


def apply_rssi_realism(rows: list[dict], cfg: ScanConfig, rng: random.Random) -> None:
    """Degrade exact simulator RSSI to what a chipset would have reported.

    Per frame and in place, before any aggregation, so every statistic taken
    afterwards inherits the same resolution limit. Averaging pulls the error
    back down, which is why quantisation bites hardest once a sweep leaves
    only a beacon or two per AP.
    """
    if not (cfg.rssi_noise_db or cfg.rssi_quant_db or cfg.rssi_bias_db):
        return
    bias = rng.uniform(-cfg.rssi_bias_db, cfg.rssi_bias_db) if cfg.rssi_bias_db else 0.0
    for r in rows:
        v = r["rssi"] + bias
        if cfg.rssi_noise_db:
            v += (rng.gauss(0.0, cfg.rssi_noise_db)
                  if cfg.rssi_noise_model == "gaussian"
                  else rng.uniform(-cfg.rssi_noise_db, cfg.rssi_noise_db))
        if cfg.rssi_quant_db:
            v = round(v / cfg.rssi_quant_db) * cfg.rssi_quant_db
        r["rssi"] = v


def sweep_schedule(channels: list[int], window: float, cfg: ScanConfig,
                   rng: random.Random) -> tuple[list[int], float]:
    """Channel visited in each dwell slot, and where the first slot starts.

    cfg.passes is the setting to understand:

      passes=0  fill the window - floor(window / dwell) slots, truncated to a
                whole number of passes. For 5.5 s and four channels that is 12
                passes, ~1.3 s per channel. This is the rotating radio a real
                client runs, and it is the default.
      passes=1  one visit per channel: 440 ms of a 5.5 s window, about one
                beacon per AP, so most per-AP statistics collapse to a single
                sample.
      passes=n  exactly n passes.

    Order is randomised per scan. The simulator assigns channels round-robin by
    AP index, so a fixed ascending order would tie "how stale my load estimate
    is" to the AP's index - an artefact no real deployment has.
    """
    order = list(channels)
    if cfg.order == "random":
        rng.shuffle(order)

    max_slots = int(window / cfg.dwell_s) if cfg.dwell_s > 0 else 0
    want = cfg.passes * len(order) if cfg.passes else (max_slots // len(order)) * len(order)
    n_slots = min(want, max_slots)
    if n_slots < len(order) and max_slots >= 1:
        # not even one full pass fits; visit as many channels as it does,
        # leaving the rest genuinely unheard
        n_slots = max_slots

    span = n_slots * cfg.dwell_s
    # "end" finishes the sweep at the decision instant: the least stale view of
    # the medium the client could hold.
    t0 = max(0.0, window - span) if cfg.align == "end" else 0.0
    return [order[i % len(order)] for i in range(n_slots)], t0


def synthesise_probe_responses(rows: list[dict], macs_by_chan: dict[int, list[str]],
                               slots: list[int], t0: float, window: float,
                               cfg: ScanConfig, rng: random.Random) -> list[dict]:
    """Reconstruct the probe responses an active scan would have elicited.

    THIS SYNTHESISES FRAMES THAT ARE NOT IN THE CAPTURE. The simulator's
    scanner radios are passive (ActiveProbing=false), so no probe response
    exists to be found. The reconstruction is sound rather than invented
    because a probe response and a beacon are the same management frame from
    the same radio at the same power, so a beacon recorded near the probe is a
    valid draw for what the response would have looked like.

    Unlike a beacon, a probe response is unicast, so it is acknowledged and
    retried. The per-AP beacon delivery ratio over the window gives the
    per-attempt success probability and the retry budget compounds it, which
    keeps the range dependence while weakening it.
    """
    beacons_of: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["beacon"] and r["bssid"]:
            beacons_of[r["bssid"]].append(r)

    expected_beacons = max(1.0, window / BEACON_INTERVAL_S)
    retune_s = cfg.retune_ms / 1000.0

    out: list[dict] = []
    for i, ch in enumerate(slots):
        # the request goes out as soon as the synthesiser has settled
        t_probe = t0 + i * cfg.dwell_s + retune_s
        for mac in macs_by_chan.get(ch, []):
            heard = beacons_of.get(mac)
            if not heard:
                continue  # never decoded once all window - out of range, no reply
            pdr = min(1.0, len(heard) / expected_beacons)
            if rng.random() > 1.0 - (1.0 - pdr) ** (cfg.probe_retries + 1):
                continue  # every retry lost
            near = min(heard, key=lambda b: abs(b["t"] - t_probe))
            out.append({**near, "t": t_probe})
    return out


def single_radio_sweep(rows: list[dict], freq_of_chan: dict[int, int], window: float,
                       cfg: ScanConfig, rng: random.Random,
                       macs_by_chan: dict[int, list[str]] | None = None
                       ) -> tuple[list[dict], list[dict], dict[int, float], list[int], float]:
    """Keep only the frames one sweeping radio could have received.

    Returns the surviving frames, any synthesised probe responses, per-channel
    listening time, and the exact dwell schedule. The schedule is reused for
    CCA-busy intervals so packet features and PHY-state features describe the
    same one-radio scan.
    """
    channels = sorted(freq_of_chan)
    slots, t0 = sweep_schedule(channels, window, cfg, rng)

    listen_s = {ch: 0.0 for ch in channels}
    for ch in slots:
        listen_s[ch] += cfg.listen_s

    chan_of_freq = {f: ch for ch, f in freq_of_chan.items()}
    retune_s = cfg.retune_ms / 1000.0
    kept = []
    for r in rows:
        # The simulator timestamps a frame when reception ENDS, so it occupied
        # [t - dur, t]. Membership keys on the start: a radio tuning in
        # mid-frame missed the preamble and decodes nothing, while one already
        # listening finishes what it is receiving.
        start = r["t"] - r["dur"] / 1e6
        # floor, not int(): int() truncates toward zero, so a frame starting
        # before t0 would land in slot 0 instead of being rejected.
        i = math.floor((start - t0) / cfg.dwell_s) if cfg.dwell_s > 0 else -1
        if not (0 <= i < len(slots)):
            continue
        if chan_of_freq.get(r["freq"]) != slots[i]:
            continue
        if start - (t0 + i * cfg.dwell_s) < retune_s:
            continue  # radio was still retuning
        kept.append(r)

    probes = []
    if cfg.mode == "active" and macs_by_chan:
        probes = synthesise_probe_responses(rows, macs_by_chan, slots, t0, window,
                                            cfg, rng)
    return kept, probes, listen_s, slots, t0


def read_observation(obs_csv: Path) -> list[dict]:
    """Read the frames the simulator's scanner radios decoded.

    Already restricted to the pre-association window, but not to one radio's
    worth of attention - single_radio_sweep() does that afterwards.
    """
    rows = []
    with obs_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["t"]),
                "freq": int(r["freq_mhz"]),
                "bssid": r["bssid"].lower() or None,
                "ta": r["ta"].lower() or None,
                "cat": int(r["cat"]),
                "beacon": r["is_beacon"] == "1",
                "retry": r["retry"] == "1",
                "len": int(r["len"]),
                "rssi": float(r["signal_dbm"]),
                "noise": float(r["noise_dbm"]),
                "dur": float(r["duration_us"]),
                "rate": float(r["rate_mbps"]),
            })
    return rows


def read_chanbusy(path: Path) -> list[dict]:
    """Read scanner PHY busy intervals written by the simulator.

    Not decoded frames: the radio's carrier-sense state, where ns-3 considered
    the medium unavailable because the PHY was transmitting, receiving or in
    CCA_BUSY. Real chipsets expose this even when no header was decoded.
    """
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            busy_frac = float(r["busy_frac"]) if "busy_frac" in r else 1.0
            rows.append({
                "start": float(r["start"]),
                "end": float(r["end"]),
                "channel": int(r["channel"]),
                "freq": int(r["freq_mhz"]),
                "busy_frac": busy_frac,
            })
    return rows


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def cca_busy_by_channel(events: list[dict], channels: list[int], window: float,
                        cfg: ScanConfig, obs_s: dict[int, float],
                        slots: list[int] | None = None, t0: float = 0.0) -> dict[int, float]:
    """Return CCA-busy fraction per channel under the chosen observation model.

    A better occupancy signal than counting frames, since it registers energy
    that was never decodable - and the one signal here with no equivalent in
    the raw frame trace. Without a sweep every channel was watched for the
    whole window; with one, only the scheduled dwells count and the
    denominator is time actually spent on that channel.
    """
    busy_s = {ch: 0.0 for ch in channels}
    if not events:
        return {ch: 0.0 for ch in channels}

    if not cfg.sweep:
        for e in events:
            if e["channel"] in busy_s:
                busy_s[e["channel"]] += (
                    _overlap(e["start"], e["end"], 0.0, window) * e["busy_frac"])
    else:
        slots = slots or []
        retune_s = cfg.retune_ms / 1000.0
        by_channel: dict[int, list[dict]] = defaultdict(list)
        for e in events:
            by_channel[e["channel"]].append(e)
        for i, ch in enumerate(slots):
            listen_start = t0 + i * cfg.dwell_s + retune_s
            listen_end = t0 + (i + 1) * cfg.dwell_s
            for e in by_channel.get(ch, []):
                busy_s[ch] += (
                    _overlap(e["start"], e["end"], listen_start, listen_end) *
                    e["busy_frac"])

    return {ch: (busy_s[ch] / obs_s[ch] if obs_s.get(ch) else 0.0) for ch in channels}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _or_default(value, default):
    """Substitute only for a genuinely absent value, never for a zero."""
    return default if value is None else value


def _std(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def channel_features(rows: list[dict], window: float,
                     probes: list[dict] | None = None,
                     cca_busy_frac: float = 0.0) -> tuple[dict, dict[str, dict]]:
    """Return (channel-level features, per-BSSID features) for one channel.

    `window` is time spent listening to THIS channel - the whole feature window
    with parallel scanners, the accumulated dwell under a sweep. Every rate and
    fraction is normalised by it, so a channel watched for 110 ms and one
    watched for 4.5 s share a scale, differing only in sampling error.

    `probes` feed the RSSI and detection features but are excluded from every
    occupancy, rate and client-count statistic: they are airtime the CLIENT
    provoked, not background load it will contend with after joining.
    """
    probes = probes or []
    total_airtime_us = sum(r["dur"] for r in rows)
    data_rows = [r for r in rows if r["cat"] == TYPE_DATA]

    # With APs spread across channels these differ per option, and they are
    # what makes AP choice more than a signal-strength comparison.
    chan = {
        "feat_chan_busy_frac": total_airtime_us / (window * 1e6) if window else 0.0,
        "feat_chan_cca_busy_frac": cca_busy_frac,
        "feat_chan_frames_per_s": len(rows) / window if window else 0.0,
        "feat_chan_bytes_per_s": sum(r["len"] for r in rows) / window if window else 0.0,
        "feat_chan_data_frac": len(data_rows) / len(rows) if rows else 0.0,
        "feat_chan_mgmt_frac": sum(1 for r in rows if r["cat"] == TYPE_MGMT) / len(rows) if rows else 0.0,
        "feat_chan_ctrl_frac": sum(1 for r in rows if r["cat"] == TYPE_CTRL) / len(rows) if rows else 0.0,
        "feat_chan_retry_frac": sum(1 for r in rows if r["retry"]) / len(rows) if rows else 0.0,
        # `or 0.0` would also fire on a legitimate mean of exactly zero; be explicit
        "feat_chan_mean_data_rate": _or_default(_mean([r["rate"] for r in data_rows]), 0.0),
        "feat_chan_n_bssids": len({r["bssid"] for r in rows if r["bssid"]}),
        "feat_chan_n_tas": len({r["ta"] for r in rows if r["ta"]}),
        "feat_chan_mean_rssi": _or_default(_mean([r["rssi"] for r in rows]), -100.0),
        "_airtime_us": total_airtime_us,
    }

    by_bssid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["bssid"]:
            by_bssid[r["bssid"]].append(r)
    probes_of: dict[str, list[dict]] = defaultdict(list)
    for p in probes:
        if p["bssid"]:
            probes_of[p["bssid"]].append(p)

    per_ap: dict[str, dict] = {}
    # an AP that only answered a probe still has to appear: under an active
    # scan that is the normal way to discover a BSS whose beacons never arrived
    for bssid in sorted(set(by_bssid) | set(probes_of)):
        frames = by_bssid.get(bssid, [])
        # a probe response carries the same elements as a beacon from the same
        # radio, so it counts as one wherever "what did the AP tell me" is asked
        beacons = sorted([f for f in frames if f["beacon"]] + probes_of.get(bssid, []),
                         key=lambda f: f["t"])
        beacon_rssi = [f["rssi"] for f in beacons]
        btimes = [f["t"] for f in beacons]
        gaps = [b - a for a, b in zip(btimes, btimes[1:])]
        airtime = sum(f["dur"] for f in frames)
        # transmitters in this BSS other than the AP itself: a client-count
        # estimate a scanning radio really can form
        clients = {f["ta"] for f in frames if f["ta"] and f["ta"] != bssid}
        dframes = [f for f in frames if f["cat"] == TYPE_DATA]

        per_ap[bssid] = {
            "feat_ap_rssi_mean": _mean(beacon_rssi),
            "feat_ap_rssi_std": _std(beacon_rssi),
            "feat_ap_rssi_max": max(beacon_rssi) if beacon_rssi else None,
            "feat_ap_rssi_min": min(beacon_rssi) if beacon_rssi else None,
            "feat_ap_rssi_last": beacon_rssi[-1] if beacon_rssi else None,
            "feat_ap_beacons": len(beacons),
            # This is the interval between beacons the radio DECODED, set by
            # how often the sweep revisits the channel (~0.55 s), not by the
            # AP's 0.1024 s beacon period - a property of the scan as much as
            # of the AP. A gap also needs two beacons; with one it is None and
            # models/data.py fills 0.0, which is indistinguishable from a
            # genuinely tiny gap, so the _known indicator carries that apart.
            "feat_ap_beacon_gap_mean": _mean(gaps),
            "feat_ap_beacon_gap_std": _std(gaps),
            "feat_ap_beacon_gap_known": float(len(gaps) > 0),
            "feat_ap_frames": len(frames),
            "feat_ap_frames_per_s": len(frames) / window if window else 0.0,
            "feat_ap_bytes_per_s": sum(f["len"] for f in frames) / window if window else 0.0,
            "feat_ap_airtime_frac": airtime / (window * 1e6) if window else 0.0,
            "feat_ap_n_clients": len(clients),
            "feat_ap_data_frames": len(dframes),
            "feat_ap_mean_data_rate": _or_default(_mean([f["rate"] for f in dframes]), 0.0),
            "feat_ap_retry_frac": sum(1 for f in frames if f["retry"]) / len(frames) if frames else 0.0,
        }
    return chan, per_ap


def build_scan_rows(scan_id: str, runs: list[tuple[int, Path, dict]],
                     cfg: ScanConfig, topology_id: str | None = None,
                     candidate_stratum: str | None = None) -> list[dict]:
    """runs: (target_ap, run_dir, metadata) for every AP option in one scan.

    The observation is identical across a scan's runs - the simulator parks the
    association radio so the choice cannot influence what was heard, verified
    byte-for-byte in validate_sim.py V7 - so it is read once here and only the
    labels come from the individual runs.

    The realism filters are applied to that shared observation once, so every
    option still sees the same frames, sweep schedule and RSSI draws. Applying
    them per run would break the one property the design rests on.
    """
    runs = sorted(runs)
    ref_dir = next((d for _, d, _ in runs if (d / "observation.csv").exists()), None)
    if ref_dir is None:
        raise FileNotFoundError("no run in this scan recorded an observation")
    ref_meta = next(m for _, d, m in runs if d == ref_dir)
    window = ref_meta["params"]["feature_window_end"]

    ap_meta = {ap["index"]: ap for ap in ref_meta["aps"]}
    mac_of = {ap["index"]: ap["mac"].lower() for ap in ref_meta["aps"]}
    chan_of = {ap["index"]: int(ap["channel"]) for ap in ref_meta["aps"]}

    rows = read_observation(ref_dir / "observation.csv")
    chanbusy = read_chanbusy(ref_dir / "chanbusy.csv")
    freq_of_chan = {ch: 5000 + 5 * ch for ch in SCAN_CHANNELS}

    apply_rssi_realism(rows, cfg, _rng(cfg, scan_id, "rssi"))

    # How long the client listened to each channel. Without the sweep every
    # channel had its own radio for the whole window.
    obs_s = {ch: window for ch in freq_of_chan}
    probes: list[dict] = []
    slots: list[int] | None = None
    sweep_t0 = 0.0
    if cfg.sweep:
        macs_by_chan: dict[int, list[str]] = defaultdict(list)
        for idx, mac in mac_of.items():
            macs_by_chan[chan_of[idx]].append(mac)
        rows, probes, obs_s, slots, sweep_t0 = single_radio_sweep(rows, freq_of_chan,
                                                                  window, cfg,
                                                                  _rng(cfg, scan_id, "sweep"),
                                                                  macs_by_chan)
    cca_busy = cca_busy_by_channel(chanbusy, sorted(set(chan_of.values())), window,
                                   cfg, obs_s, slots, sweep_t0)

    by_freq: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_freq[r["freq"]].append(r)
    probes_by_freq: dict[int, list[dict]] = defaultdict(list)
    for p in probes:
        probes_by_freq[p["freq"]].append(p)

    chan_feats: dict[int, dict] = {}
    per_ap: dict[str, dict] = {}
    for ch in sorted(set(chan_of.values())):
        cf, ap_f = channel_features(by_freq.get(freq_of_chan[ch], []), obs_s[ch],
                                    probes_by_freq.get(freq_of_chan[ch], []),
                                    cca_busy.get(ch, 0.0))
        chan_feats[ch] = cf
        per_ap.update(ap_f)

    # Filter 3: an AP is an option only if a beacon from it decoded (or an
    # active scan reconstructed a probe response). A client cannot join a BSS
    # it never heard of.
    heard = {idx: per_ap[mac]["feat_ap_rssi_mean"]
             for idx, mac in mac_of.items()
             if mac in per_ap and per_ap[mac]["feat_ap_rssi_mean"] is not None}
    if len(heard) < 2:
        print(f"WARN: scan {scan_id} discovered only {len(heard)} AP(s); skipping",
              file=sys.stderr)
        return []

    # Sums over every BSSID heard, including APs that are not options here.
    # Their traffic still occupies the medium the client will contend for, so
    # the wider denominator is deliberate - but it means feat_rel_airtime_share
    # does NOT sum to 1 across a scan's options, which the name invites you to
    # assume. The window-normalised fraction is used rather than raw
    # microseconds because two channels need not have been watched equally long.
    total_airtime = sum(v["feat_ap_airtime_frac"] for v in per_ap.values()) or 1.0

    rows = []
    for target_ap, _dir, meta in runs:
        if target_ap not in heard:
            continue

        mac = mac_of[target_ap]
        chan = dict(chan_feats[chan_of[target_ap]])
        chan.pop("_airtime_us", None)
        apf = dict(per_ap[mac])

        mine = heard[target_ap]
        others = [v for k, v in heard.items() if k != target_ap]
        best_other = max(others) if others else None

        rel = {
            "feat_rel_n_options": len(heard),
            # How many options are STRICTLY stronger, so tied options share a
            # rank. RSSI is quantised to whole dBm, so ties are common.
            "feat_rel_rssi_rank": sum(1 for v in heard.values() if v > mine),
            "feat_rel_is_strongest": int(mine == max(heard.values())),
            "feat_rel_rssi_margin_best_other": (
                mine - best_other if best_other is not None else None),
            "feat_rel_rssi_minus_mean": mine - statistics.fmean(heard.values()),
            "feat_rel_airtime_share": per_ap[mac]["feat_ap_airtime_frac"] / total_airtime,
            "feat_rel_clients_share": None,
        }
        tot_clients = sum(per_ap[m]["feat_ap_n_clients"] for m in per_ap)
        rel["feat_rel_clients_share"] = (
            per_ap[mac]["feat_ap_n_clients"] / tot_clients if tot_clients else 0.0)

        c = meta["candidate"]
        gt_ap = ap_meta[target_ap]
        row = {
            "topology_id": topology_id or scan_id,
            "scan_id": scan_id,
            "ap_index": target_ap,
            "run_id": meta["run_id"],
            "meta_window_s": window,  # constant across runs; not a feature
            # what this option's channel was actually watched for, and under
            # which realism settings - provenance, not model input
            "meta_scan_sweep": int(cfg.sweep),
            "meta_scan_mode": cfg.mode if cfg.sweep else "parallel",
            "meta_scan_seconds": obs_s[chan_of[target_ap]],
            "meta_scan_dwell_ms": cfg.dwell if cfg.sweep else None,
            "meta_scan_retune_ms": cfg.retune_ms if cfg.sweep else None,
            "meta_scan_passes": cfg.passes if cfg.sweep else None,
            "meta_scan_order": cfg.order if cfg.sweep else None,
            "meta_scan_align": cfg.align if cfg.sweep else None,
            "meta_scan_seed": cfg.seed,
            "meta_rssi_noise_db": cfg.rssi_noise_db,
            "meta_rssi_noise_model": cfg.rssi_noise_model,
            "meta_rssi_bias_db": cfg.rssi_bias_db,
            "meta_rssi_quant_db": cfg.rssi_quant_db,
            **chan, **apf, **rel,
            "label_throughput_mbps": c["throughput_mbps"],
            "label_associated": int(c["associated"]),
            "gt_rng_seed": meta["rng_seed"],
            "gt_topology_seed": meta["params"]["topology_seed"],
            "gt_n_aps": meta["params"]["n_aps"],
            "gt_n_stas": meta["params"]["n_stas"],
            "gt_ap_spacing": meta["params"]["ap_spacing"],
            "gt_bg_mean_per_sta_mbps": meta["params"]["bg_mean_per_sta_mbps"],
            "gt_bg_total_offered_mbps": meta["params"]["bg_total_offered_mbps"],
            "gt_packet_size": meta["params"]["packet_size"],
            "gt_n_hotspots": meta["params"]["n_hotspots"],
            "gt_hotspot_aps": ",".join(map(str, meta["params"]["hotspot_aps"])),
            "gt_candidate_x": meta["candidate_position"]["x"],
            "gt_candidate_y": meta["candidate_position"]["y"],
            "gt_candidate_stratum": candidate_stratum,
            "gt_true_distance": gt_ap["candidate_distance"],
            "gt_ap_sta_count": gt_ap["sta_count"],
            "gt_ap_offered_mbps": gt_ap["offered_mbps"],
            "gt_ap_background_mbps": gt_ap["background_mbps"],
            "gt_assoc_delay": c["assoc_delay"],
            "gt_observed_seconds": c["observed_seconds"],
            "gt_n_channels": meta["params"]["n_channels"],
            "gt_ap_channel": chan_of[target_ap],
        }
        rows.append(row)
    return rows


def _scan_key(run_dir: Path) -> str:
    """The scan token in a run tag: t00007__c01__ap1 -> "c01"."""
    return run_dir.name.split("__")[1]


def reference_dir(runs_dirs, topology_id: str, scan_key: str) -> Path:
    """The run directory holding one scan's observation, across sweep roots.

    A dataset can be stitched from sweeps written to different roots, so all of
    them are searched. ap0 normally carries the recording; the scan is still
    found if a different run recorded it.
    """
    if isinstance(runs_dirs, Path):
        runs_dirs = [runs_dirs]
    prefix = f"{topology_id}__{scan_key}__"
    for runs_dir in runs_dirs:
        exact = runs_dir / f"{prefix}ap0"
        if (exact / "observation.csv").exists():
            return exact
        refs = [d for d in runs_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
                and (d / "observation.csv").exists()]
        if refs:
            return sorted(refs)[0]
    raise FileNotFoundError(
        f"no observation for {topology_id}__{scan_key} under "
        f"{[str(d) for d in runs_dirs]}")


def process_topology(item) -> list[dict]:
    """Build every row for one topology, across all of its scans.

    Both integrity checks below fail the build rather than warn. A half-finished
    sweep would otherwise yield a quietly smaller dataset, and a scan missing one
    of its runs would hand the ranking task a truncated option set - making the
    label the best of whatever finished rather than the best available. That is
    a wrong answer, not a missing one.
    """
    topology_id, run_dirs, cfg, manifest_entry = item
    by_scan: dict[str, list[tuple[int, Path, dict]]] = defaultdict(list)
    for d in run_dirs:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        by_scan[_scan_key(d)].append((meta["candidate"]["target_ap"], d, meta))
    if not by_scan:
        raise ValueError(f"{topology_id}: no readable scans")

    expected_keys = {f"c{index:02d}"
                     for index in range(len(manifest_entry["candidates"]))}
    if set(by_scan) != expected_keys:
        raise ValueError(
            f"{topology_id}: scans {sorted(by_scan)} do not match "
            f"manifest {sorted(expected_keys)}")

    rows: list[dict] = []
    for scan_key, runs in sorted(by_scan.items()):
        expected = runs[0][2]["params"]["n_aps"]
        if len(runs) != expected:
            raise ValueError(
                f"scan {topology_id}__{scan_key} has {len(runs)}/{expected} runs")
        rows.extend(build_scan_rows(
            f"{topology_id}__{scan_key}",
            runs,
            cfg,
            topology_id=topology_id,
            candidate_stratum=runs[0][2]["candidate_stratum"],
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--workers", type=int, default=8)

    dflt = ScanConfig()
    sweep_grp = parser.add_argument_group(
        "single-radio sweep",
        "Reduce the simulator's per-channel scanner radios to the one radio a "
        "real client has, by keeping only the frames that fell inside the "
        "dwell scheduled on their channel.")
    sweep_grp.add_argument("--single-radio-sweep", action="store_true",
                   help="enable the sweep (default: keep the full parallel capture)")
    sweep_grp.add_argument("--sweep-mode", choices=["passive", "active"], default=dflt.mode,
                   help=f"passive listens for beacons; active probes and reconstructs "
                        f"responses for comparison work (default {dflt.mode})")
    sweep_grp.add_argument("--sweep-dwell-ms", type=float, default=None,
                   help=f"dwell per channel visit (default: {PASSIVE_DWELL_MS:g} passive, "
                        f"{ACTIVE_DWELL_MS:g} active - the values Linux mac80211 and "
                        "Qualcomm's Android WLAN config ship; below the "
                        f"{BEACON_INTERVAL_S * 1000:g} ms beacon interval a passive dwell "
                        "misses APs outright)")
    sweep_grp.add_argument("--sweep-probe-retries", type=int, default=dflt.probe_retries,
                   help=f"retry budget for a unicast probe response (default "
                        f"{dflt.probe_retries}; active mode only)")
    sweep_grp.add_argument("--sweep-retune-ms", type=float, default=dflt.retune_ms,
                   help=f"blind time at the start of each dwell (default {dflt.retune_ms:g})")
    sweep_grp.add_argument("--sweep-passes", type=int, default=dflt.passes,
                   help=f"full sweeps of the channel list (default {dflt.passes} "
                        "= as many as the window fits)")
    sweep_grp.add_argument("--sweep-order", choices=["random", "ascending"], default=dflt.order,
                   help=f"channel visit order (default {dflt.order})")
    sweep_grp.add_argument("--sweep-align", choices=["end", "start"], default=dflt.align,
                   help=f"put the scan at the end or start of the window (default {dflt.align})")

    rssi_grp = parser.add_argument_group(
        "RSSI realism",
        "Simulator RSSI is exact to millidecibels; chipsets report whole dBm "
        "with several dB of error. Applied per frame before any aggregation. "
        "Pass 0 to both to recover the raw simulator values.")
    rssi_grp.add_argument("--rssi-noise-db", type=float, default=dflt.rssi_noise_db,
                   help=f"per-frame measurement error, +/- this (default {dflt.rssi_noise_db:g})")
    rssi_grp.add_argument("--rssi-noise-model", choices=["uniform", "gaussian"],
                   default=dflt.rssi_noise_model,
                   help=f"uniform bound or gaussian std dev (default {dflt.rssi_noise_model})")
    rssi_grp.add_argument("--rssi-bias-db", type=float, default=dflt.rssi_bias_db,
                   help="per-scan systematic offset, +/- this (default 0). Unlike "
                        "per-frame noise this does NOT average out over beacons")
    rssi_grp.add_argument("--rssi-quant-db", type=float, default=dflt.rssi_quant_db,
                   help=f"report resolution in dB (default {dflt.rssi_quant_db:g} = whole dBm)")
    rssi_grp.add_argument("--scan-seed", type=int, default=dflt.seed,
                   help="seed for sweep scheduling and RSSI noise (per scan; reproducible)")

    args = parser.parse_args()
    cfg = ScanConfig(
        sweep=args.single_radio_sweep, mode=args.sweep_mode,
        dwell_ms=args.sweep_dwell_ms, probe_retries=args.sweep_probe_retries,
        retune_ms=args.sweep_retune_ms, passes=args.sweep_passes,
        order=args.sweep_order, align=args.sweep_align,
        rssi_noise_db=args.rssi_noise_db, rssi_noise_model=args.rssi_noise_model,
        rssi_bias_db=args.rssi_bias_db, rssi_quant_db=args.rssi_quant_db,
        seed=args.scan_seed)

    by_topology: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        by_topology[d.name.split("__")[0]].append(d)
    print(f"{len(by_topology)} topologies under {args.runs_dir}")
    print(f"observation model: {cfg.describe()}")

    manifest_path = args.runs_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    missing_topologies = sorted(set(manifest) - set(by_topology))
    if missing_topologies:
        sys.exit(f"manifest topologies have no run directories: {missing_topologies[:10]}")
    work = [(topology_id, dirs, cfg, manifest.get(topology_id, {}))
            for topology_id, dirs in by_topology.items()]
    rows: list[dict] = []
    if args.workers <= 1:
        iterator = map(process_topology, work)
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        iterator = pool.map(process_topology, work)
    try:
        for i, res in enumerate(iterator, 1):
            rows.extend(res)
            if i % 50 == 0 or i == len(by_topology):
                print(f"[{i}/{len(by_topology)}] {len(rows)} rows")
    finally:
        if args.workers > 1:
            pool.shutdown()

    if not rows:
        sys.exit("no rows extracted")

    # imported here, not at module scope: the worker processes never need it
    import pandas as pd

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns)} cols "
          f"({df.scan_id.nunique()} scans) to {args.out}")


if __name__ == "__main__":
    main()
