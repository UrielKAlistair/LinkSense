from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

from scripts.tf.cache_dataset import NO_AP
from scripts.tf.train import CachedScan, Scale, gaussian_nll, make_batch, subsample

ARGS = argparse.Namespace(frame_cap=4, identities=16, device="cpu")


def scan(seed=0) -> CachedScan:
    """Two dwells on channel 36, each with a channel cell and the cells of two
    discovered APs; frame i is in dwell i // 6 and belongs to AP i % 3, or to
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
        frame_offsets=rng.random(12).astype(np.float32),
        frame_categories=rng.integers(0, 3, 12).astype(np.uint8),
        members=members,
        cell_is_channel=cell_ap == NO_AP,
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
    def test_an_ap_cell_gathers_its_own_frames_and_carries_its_aps_identity(self):
        scans = [scan(0), scan(1)]
        starts = np.cumsum([0] + [len(s.frames) for s in scans[:-1]])
        for rng in (None, np.random.default_rng(3)):
            batch, _, _ = make_batch(scans, Scale(scans), ARGS, rng)
            for g, cached in enumerate(scans):
                for c, ap in enumerate(cached.cell_ap):
                    real = batch.cell_member_mask[g, c].numpy()
                    rows = batch.cell_members[g, c].numpy()[real] - starts[g]
                    self.assertEqual(len(rows), min(ARGS.frame_cap, len(cached.members[c])))
                    self.assertTrue(set(rows.tolist()) <= set(cached.members[c].tolist()))
                    self.assertEqual(rows.tolist(), sorted(rows.tolist()))
                    if ap != NO_AP:
                        self.assertEqual(int(batch.cell_ap_identity[g, c]),
                                         int(batch.ap_identity[g, ap]))

    def test_carrier_sense_reaches_channel_cells_only(self):
        cached = scan()
        batch, _, _ = make_batch([cached], Scale([cached]), ARGS, None)
        channel = cached.cell_is_channel
        cca = batch.cell_cca[0].numpy()
        self.assertTrue(np.allclose(cca[channel], cached.cca_samples[cached.cell_dwell[channel]]))
        self.assertEqual(float(np.abs(cca[~channel]).sum()), 0.0)

    def test_frame_offsets_reach_the_batch_unstandardised(self):
        cached = scan()
        batch, _, _ = make_batch([cached], Scale([cached]), ARGS, None)
        self.assertTrue(np.allclose(batch.frame_offsets.numpy(), cached.frame_offsets))

    def test_a_column_empty_on_one_kind_of_cell_scales_to_zero_there(self):
        scans = [scan(0), scan(1)]
        batch, _, _ = make_batch(scans, Scale(scans), ARGS, None)
        channel = scans[0].cell_is_channel
        self.assertEqual(float(np.abs(batch.cell_aggregates[0].numpy()[channel, 2]).max()), 0.0)

    def test_the_target_is_standardised_log1p_throughput(self):
        scans = [scan(0), scan(1)]
        scale = Scale(scans)
        _, target, mask = make_batch(scans, scale, ARGS, None)
        logs = np.log1p(np.concatenate([s.labels for s in scans]))
        expected = (logs - logs.mean()) / logs.std()
        self.assertTrue(np.allclose(target[mask].numpy(), expected, atol=1e-5))


class LossTests(unittest.TestCase):
    def test_padded_aps_do_not_enter_the_loss(self):
        mu = torch.zeros(2, 3)
        log_var = torch.zeros(2, 3)
        target = torch.tensor([[1.0, 1.0, 50.0], [1.0, 1.0, -50.0]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        self.assertAlmostEqual(float(gaussian_nll(mu, log_var, target, mask)), 0.5)

    def test_every_ap_weighs_the_same_whatever_its_scan_holds(self):
        mu, log_var = torch.zeros(2, 4), torch.zeros(2, 4)
        # The one-AP scan is far off and the four-AP scan is close. One flat mean
        # over the five discovered APs gives 4.0; averaging within each scan first
        # would let the single bad AP carry half the loss and give 9.25.
        target = torch.tensor([[6.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        few = torch.tensor([[True, False, False, False], [True, True, True, True]])
        loss = gaussian_nll(mu, log_var, target, few, variance=False)
        self.assertAlmostEqual(float(loss), 4.0)

    def test_widening_the_variance_is_only_worth_it_when_the_mean_is_wrong(self):
        target = torch.zeros(1, 1)
        mask = torch.ones(1, 1, dtype=torch.bool)
        right = torch.zeros(1, 1)
        wrong = torch.full((1, 1), 3.0)
        narrow, wide = torch.zeros(1, 1), torch.full((1, 1), 2.0)
        self.assertLess(float(gaussian_nll(right, narrow, target, mask)),
                        float(gaussian_nll(right, wide, target, mask)))
        self.assertGreater(float(gaussian_nll(wrong, narrow, target, mask)),
                           float(gaussian_nll(wrong, wide, target, mask)))


if __name__ == "__main__":
    unittest.main()
