from __future__ import annotations

import argparse
import unittest

import numpy as np

from scripts.tf.cache_dataset import AP_CELL, CHANNEL_CELL, NO_AP
from scripts.tf.train import CachedScan, Scale, make_batch, subsample

ARGS = argparse.Namespace(frame_cap=4, slots=16, device="cpu")


def scan(seed=0) -> CachedScan:
    """Two dwells on channel 36, each with a channel cell and the cells of
    two valid APs; frame i is in dwell i // 6 and belongs to AP i % 3, or to
    none when that is 2."""
    rng = np.random.default_rng(seed)
    frame_dwell = np.repeat([0, 1], 6)
    frame_ap = np.array([0, 1, NO_AP] * 4)
    cell_dwell = np.repeat([0, 1], 3)
    cell_ap = np.array([NO_AP, 0, 1] * 2)
    members = []
    for dwell, ap in zip(cell_dwell, cell_ap):
        mine = frame_dwell == dwell
        if ap != NO_AP:
            mine &= frame_ap == ap
        members.append(np.flatnonzero(mine))
    aggregates = rng.normal(size=(6, 3)).astype(np.float32)
    aggregates[cell_ap == NO_AP, 2] = -100.0          # empty on channel cells
    return CachedScan(
        scan_id=f"s{seed}", topology_id=f"t{seed}",
        frames=rng.normal(size=(12, 5)).astype(np.float32),
        frame_categories=rng.integers(0, 3, 12).astype(np.uint8),
        members=members,
        cell_kind=np.where(cell_ap == NO_AP, CHANNEL_CELL, AP_CELL).astype(np.uint8),
        cell_dwell=cell_dwell.astype(np.uint8), cell_channel=np.full(6, 36, np.uint8),
        cell_ap=cell_ap, cell_aggregates=aggregates,
        cca_samples=rng.random((2, 109)).astype(np.float32),
        dwell_channel=np.full(2, 36, np.uint8),
        descriptors=rng.normal(size=(2, 4)).astype(np.float32),
        labels=np.array([3.0, 8.0], np.float32), ap_channels=np.full(2, 36, np.uint8))


class SubsampleTests(unittest.TestCase):
    def test_evaluation_keeps_an_even_stride_in_the_order_heard(self):
        kept = subsample(np.arange(10, 20), 4, rng=None)
        self.assertEqual(kept.tolist(), [10, 13, 16, 19])

    def test_training_keeps_a_random_subset_in_the_order_heard(self):
        kept = subsample(np.arange(100), 10, np.random.default_rng(0))
        self.assertEqual(len(set(kept.tolist())), 10)
        self.assertEqual(kept.tolist(), sorted(kept.tolist()))

    def test_a_cell_under_the_cap_keeps_every_frame(self):
        self.assertEqual(subsample(np.arange(3), 4, np.random.default_rng(0)).tolist(), [0, 1, 2])


class BatchTests(unittest.TestCase):
    def test_an_ap_cell_gathers_its_own_frames_and_carries_its_aps_slot(self):
        scans = [scan(0), scan(1)]
        for rng in (None, np.random.default_rng(3)):
            batch, _, _ = make_batch(scans, Scale(scans), ARGS, rng)
            for g, cached in enumerate(scans):
                for c, ap in enumerate(cached.cell_ap):
                    real = batch.cell_member_mask[g, c].numpy()
                    rows = batch.cell_members[g, c].numpy()[real]
                    self.assertEqual(len(rows), min(ARGS.frame_cap, len(cached.members[c])))
                    self.assertTrue(set(rows.tolist()) <= set(cached.members[c].tolist()))
                    self.assertEqual(rows.tolist(), sorted(rows.tolist()))
                    if ap != NO_AP:
                        self.assertEqual(int(batch.cell_slot[g, c]), int(batch.ap_slot[g, ap]))

    def test_carrier_sense_reaches_channel_cells_only(self):
        cached = scan()
        batch, _, _ = make_batch([cached], Scale([cached]), ARGS, None)
        channel = cached.cell_kind == CHANNEL_CELL
        cca = batch.cell_cca[0].numpy()
        self.assertTrue(np.allclose(cca[channel], cached.cca_samples[cached.cell_dwell[channel]]))
        self.assertEqual(float(np.abs(cca[~channel]).sum()), 0.0)

    def test_a_column_empty_on_one_kind_of_cell_scales_to_zero_there(self):
        scans = [scan(0), scan(1)]
        batch, _, _ = make_batch(scans, Scale(scans), ARGS, None)
        channel = scans[0].cell_kind == CHANNEL_CELL
        self.assertEqual(float(np.abs(batch.cell_aggregates[0].numpy()[channel, 2]).max()), 0.0)

    def test_the_target_is_standardised_log1p_throughput(self):
        scans = [scan(0), scan(1)]
        scale = Scale(scans)
        _, target, mask = make_batch(scans, scale, ARGS, None)
        logs = np.log1p(np.concatenate([s.labels for s in scans]))
        expected = (logs - logs.mean()) / logs.std()
        self.assertTrue(np.allclose(target[mask].numpy(), expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
