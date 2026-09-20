from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tf.cache_dataset import (AP_CELL, CCA_SAMPLES, CHANNEL_CELL,
                                         NO_AP, Cell, DescriptorTable, Dwell, FrameTable,
                                         build_scan)
from scripts.common.projection import (DWELL_S, MISSING_RSSI_DBM, TYPE_CTRL, TYPE_DATA,
                                       TYPE_MGMT, dwell_bounds, dwell_of, sweep_rng,
                                       sweep_schedule)
from scripts.common.parse_scans import Scan, ValidAP

AP_ONE, AP_TWO, STATION = "00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:09"


def frame(start, end, bssid=AP_ONE, ta=STATION, cat=TYPE_DATA, beacon=False,
          retry=False, length=1000, rssi=-60.0, dur=1000.0, rate=65.0, channel=36):
    return {"tx_start": start, "tx_end": end, "channel": channel, "bssid": bssid,
            "ta": ta, "cat": cat, "beacon": beacon, "retry": retry, "len": length,
            "rssi": rssi, "dur": dur, "rate": rate}




def cell(frames, cca=(), ap_mac=None):
    """A cell of a one-dwell scan that listened from 0 to 0.1 s: its channel
    cell, or, given AP_ONE, that AP's cell."""
    dwell = Dwell(0, 36, 0.0, 0.0, 0.1, list(frames), list(cca))
    table = FrameTable([dwell], [ValidAP(0, AP_ONE, 36, 12.0, True, {})])
    return Cell(table, dwell) if ap_mac is None else Cell(table, dwell, 0, ap_mac)


class AirTimeTests(unittest.TestCase):
    def test_overlapping_transmissions_are_unioned_not_summed(self):
        busy, gaps = Cell.air_time([frame(0.0, 1.0), frame(0.5, 2.0)], 0.0, 4.0)
        self.assertAlmostEqual(busy, 2.0)
        self.assertEqual(gaps, [2.0])

    def test_gaps_bracket_the_frames_inside_the_span(self):
        busy, gaps = Cell.air_time([frame(1.0, 2.0)], 0.0, 4.0)
        self.assertAlmostEqual(busy, 1.0)
        self.assertEqual([round(g, 6) for g in gaps], [1.0, 2.0])

    def test_a_transmission_is_clipped_to_the_span_it_straddles(self):
        busy, _ = Cell.air_time([frame(3.5, 5.0)], 0.0, 4.0)
        self.assertAlmostEqual(busy, 0.5)


class DwellTests(unittest.TestCase):
    def test_dwell_edges_are_whole_milliseconds(self):
        start, listen_start, end = dwell_bounds(0, 280.0)
        self.assertEqual((start, end), (0.28, 0.39))
        self.assertAlmostEqual(listen_start, 0.28015)

    def test_a_frame_just_after_retuning_belongs_to_its_own_dwell(self):
        _, listen_start, end = dwell_bounds(3, 280.0)
        self.assertEqual(dwell_of(listen_start, 280.0), 3)
        self.assertEqual(dwell_of(end, 280.0), 4)

    def test_a_moment_before_the_sweep_falls_outside_it(self):
        self.assertEqual(dwell_of(0.2799, 280.0), -1)


class CellAggregateTests(unittest.TestCase):
    def test_an_empty_cell_reads_as_a_clear_medium_with_no_level(self):
        row = cell([]).aggregates
        self.assertEqual(row["frames_log1p"], 0.0)
        # the radio was here and the medium was clear, which is a reading
        self.assertEqual(row["longest_gap_fraction"], 1.0)
        for name in ("rssi_mean", "rssi_max", "beacon_rssi_mean"):
            self.assertEqual(row[name], MISSING_RSSI_DBM, name)

    def test_beacon_columns_stay_empty_on_a_channel_cell(self):
        frames = [frame(0.01, 0.02, ta=AP_ONE, beacon=True, cat=TYPE_MGMT, rssi=-50.0)]
        channel = cell(frames).aggregates
        self.assertEqual(channel["beacons_log1p"], 0.0)
        self.assertEqual(channel["beacon_rssi_mean"], MISSING_RSSI_DBM)
        self.assertEqual(channel["rssi_mean"], -50.0)
        ap = cell(frames, ap_mac=AP_ONE).aggregates
        self.assertEqual(ap["beacon_rssi_mean"], -50.0)

    def test_carrier_sense_columns_stay_empty_on_an_ap_cell(self):
        frames = [frame(0.01, 0.02)]
        ap = cell(frames, ap_mac=AP_ONE).aggregates
        for name in ("cca_mean", "cca_max"):
            self.assertEqual(ap[name], 0.0, name)

    def test_carrier_sense_is_read_apart_from_decoded_busy(self):
        frames = [frame(0.01, 0.02)]
        channel = cell(frames, cca=[0.6] * CCA_SAMPLES).aggregates
        self.assertAlmostEqual(channel["busy_fraction"], 0.1)
        self.assertAlmostEqual(channel["cca_mean"], 0.6)

    def test_an_ap_cell_counts_stations_without_counting_the_ap(self):
        frames = [frame(0.01, 0.02, ta=AP_ONE, beacon=True, cat=TYPE_MGMT),
                  frame(0.03, 0.04, ta=STATION)]
        row = cell(frames, ap_mac=AP_ONE).aggregates
        self.assertAlmostEqual(row["transmitters_log1p"], np.log1p(1))
        self.assertAlmostEqual(row["beacons_log1p"], np.log1p(1))


class FrameRowTests(unittest.TestCase):
    def test_flags_separate_who_sent_a_frame_from_which_bss_it_names(self):
        frames = [frame(0.281, 0.2811, ta=AP_ONE, beacon=True, cat=TYPE_MGMT),
                  frame(0.282, 0.2821, bssid=None, ta=None, cat=TYPE_CTRL),
                  frame(0.283, 0.2831, bssid=AP_TWO)]
        dwell = Dwell(0, 36, 0.28, 0.28015, 0.39, frames, [])
        flags = [(row["from_ap"], row["of_valid_ap"])
                 for row in FrameTable.frame_features(dwell, {AP_ONE})]
        self.assertEqual(flags, [(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)])
        table = FrameTable([dwell], [ValidAP(0, AP_ONE, 36, 12.0, True, {})])
        self.assertEqual(table.categories.tolist(), [TYPE_MGMT, TYPE_CTRL, TYPE_DATA])

    def test_the_gap_is_idle_time_since_the_last_transmission_ended(self):
        aggregate = [frame(0.281, 0.286), frame(0.281, 0.286)]
        ack = frame(0.286016, 0.28605, bssid=None, ta=None, cat=TYPE_CTRL)
        rows = FrameTable.frame_features(Dwell(0, 36, 0.28, 0.28015, 0.39, aggregate + [ack], []), set())
        # the second subframe shares the first's transmission, and the ACK
        # follows it after a 16 us SIFS, not 5 ms after it started
        self.assertEqual(rows[1]["gap_log1p"], 0.0)
        self.assertAlmostEqual(rows[2]["gap_log1p"], float(np.log1p(16.0)), places=3)

    def test_the_first_gap_runs_from_the_moment_the_radio_finished_retuning(self):
        dwell = Dwell(0, 36, 0.28, 0.28015, 0.39, [frame(0.28115, 0.2812)], [])
        first = FrameTable.frame_features(dwell, set())[0]
        self.assertAlmostEqual(first["gap_log1p"], float(np.log1p(1000.0)), places=3)
        self.assertAlmostEqual(first["offset_in_dwell"], 0.00115 / DWELL_S, places=5)


class DescriptorTests(unittest.TestCase):
    def test_levels_come_from_beacons_not_from_the_stations_of_the_bss(self):
        ap = ValidAP(0, AP_ONE, 36, 12.0, True, {})
        frames = ([frame(0.3, 0.31, ta=AP_ONE, beacon=True, cat=TYPE_MGMT, rssi=-50.0),
                   frame(0.4, 0.41, ta=AP_ONE, beacon=True, cat=TYPE_MGMT, rssi=-52.0)]
                  + [frame(0.5 + i / 100, 0.51 + i / 100, rssi=-80.0) for i in range(10)])
        row = DescriptorTable.ap_descriptor(ap, frames, 1.428, cochannel=1, n_valid_aps=3)
        self.assertAlmostEqual(row["beacon_rssi_mean"], -51.0)
        self.assertAlmostEqual(row["beacon_rssi_last"], -52.0)
        self.assertAlmostEqual(row["beacons_log1p"], float(np.log1p(2)))
        self.assertAlmostEqual(row["bss_transmitters_log1p"], float(np.log1p(1)))

    def test_one_beacon_gives_a_level_and_no_spread(self):
        ap = ValidAP(0, AP_ONE, 36, 12.0, True, {})
        beacon = frame(0.3, 0.31, ta=AP_ONE, beacon=True, cat=TYPE_MGMT, rssi=-50.0)
        row = DescriptorTable.ap_descriptor(ap, [beacon], 1.428, cochannel=0, n_valid_aps=2)
        self.assertEqual(row["beacon_rssi_last"], -50.0)
        self.assertEqual(row["beacon_rssi_std"], 0.0)


class BuildScanTests(unittest.TestCase):
    """One synthetic scan through the whole builder."""

    def _write_scan(self, root: Path, scan_id: str) -> Scan:
        schedule, sweep_start_ms = sweep_schedule(6.0, sweep_rng(scan_id))
        channels = {AP_ONE: schedule[0], AP_TWO: next(c for c in schedule if c != schedule[0])}
        runs = {}
        for index, mac in enumerate((AP_ONE, AP_TWO)):
            run = root / f"{scan_id}__ap{index}"
            run.mkdir()
            runs[index] = run
            run.joinpath("metadata.json").write_text(json.dumps({
                "run_id": f"{scan_id}__ap{index}",
                "params": {"feature_window_end": 6.0, "n_aps": 2, "n_hotspots": 0},
                "aps": [{"index": 0, "mac": AP_ONE}, {"index": 1, "mac": AP_TWO}],
                "candidate": {"target_ap": index, "throughput_mbps": 10.0 + index,
                              "associated": 1},
            }))

        # One beacon per AP, in the first dwell on its own channel, plus one
        # station frame there, so both APs are valid.
        lines = ["tx_start,tx_end,freq_mhz,bssid,ta,cat,is_beacon,retry,len,"
                 "signal_dbm,duration_us,rate_mbps"]
        for mac, channel in channels.items():
            dwell = schedule.index(channel)
            start = (sweep_start_ms + dwell * 110.0) / 1000.0 + 0.001
            freq = 5000 + 5 * channel
            lines.append(f"{start},{start + 0.0002},{freq},{mac},{mac},"
                         f"{TYPE_MGMT},1,0,200,-55,200,24")
            lines.append(f"{start + 0.01},{start + 0.0105},{freq},{mac},{STATION},"
                         f"{TYPE_DATA},0,0,1200,-70,500,65")
        runs[0].joinpath("observation.csv").write_text("\n".join(lines) + "\n")

        busy = ["start,end,channel,freq_mhz,busy_frac"]
        for channel in set(channels.values()):
            for millisecond in range(6000):
                busy.append(f"{millisecond / 1000},{(millisecond + 1) / 1000},{channel},"
                            f"{5000 + 5 * channel},0.25")
        runs[0].joinpath("chanbusy.csv").write_text("\n".join(busy) + "\n")
        return Scan(scan_id, scan_id.split("__")[0], runs)

    def test_a_scan_becomes_one_cell_per_dwell_plus_one_per_ap_dwell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = self._write_scan(root, "t00000__c00")
            out = root / "cells"
            out.mkdir()
            row = build_scan(scan, out)

            self.assertEqual(row["n_valid_aps"], 2)
            cached = np.load(out / "t00000__c00.npz", allow_pickle=False)
            dwells = len(cached["dwell_channel"])
            self.assertEqual(row["n_cells"], dwells + 13 * 2)
            self.assertEqual(len(cached["cell_kind"]), dwells + 13 * 2)

            kind, dwell, ap = cached["cell_kind"], cached["cell_dwell"], cached["cell_ap"]
            dwell_numbers, ap_numbers = cached["frame_dwell"], cached["frame_ap"]

            def frames_of(dwell_number, ap_number):
                mine = dwell_numbers == dwell_number
                if ap_number != NO_AP:
                    mine &= ap_numbers == ap_number
                return np.flatnonzero(mine)

            # every frame sits in exactly one channel cell
            owned = sorted(int(m) for i in np.flatnonzero(kind == CHANNEL_CELL)
                           for m in frames_of(dwell[i], NO_AP))
            self.assertEqual(owned, list(range(len(cached["frames"]))))
            # each AP heard its beacon and one station frame in a single dwell,
            # so exactly one of its 13 cells holds frames, and it holds both
            for number in range(2):
                mine = np.flatnonzero((kind == AP_CELL) & (ap == number))
                self.assertEqual(len(mine), 13)
                held = [len(frames_of(dwell[i], number)) for i in mine]
                self.assertEqual(sorted(held), [0] * 12 + [2])
                frames_log1p = row["feature_names"][1].index("frames_log1p")
                heard = cached["cell_aggregates"][mine, frames_log1p] > 0
                self.assertEqual(sorted(heard.tolist()), [False] * 12 + [True])

            self.assertEqual(cached["cca_samples"].shape, (dwells, CCA_SAMPLES))
            self.assertEqual(sorted(cached["labels"].tolist()), [10.0, 11.0])

    def test_a_dwell_on_an_unoccupied_channel_senses_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = self._write_scan(root, "t00001__c00")
            out = root / "cells"
            out.mkdir()
            build_scan(scan, out)
            cached = np.load(out / "t00001__c00.npz", allow_pickle=False)
            samples = cached["cca_samples"]
            occupied = set(np.unique(cached["ap_channels"]).tolist())
            for index, channel in enumerate(cached["dwell_channel"]):
                if int(channel) in occupied:
                    self.assertGreater(samples[index].sum(), 0.0)
                else:
                    self.assertEqual(samples[index].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
