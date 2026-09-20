from __future__ import annotations

import dataclasses
import unittest

import torch

from scripts.tf.binned_model import BinnedModel, binned_view
from scripts.tf.cache_dataset import AP_CELL, CHANNEL_CELL
from scripts.tf.joint_ap_model import CellBatch, JointAPModel, gaussian_nll
from scripts.tf.target_ap_model import (CO_CHANNEL_AP, OTHER_AP, TARGET_AP,
                                           TargetAPModel, roles)

FRAME_FEATURES, AGGREGATES, DESCRIPTOR, CCA_SAMPLES = 10, 16, 10, 109
CHANNELS = torch.tensor([36, 40, 44, 48])
MEMBERS, FRAMES, SCANS = 12, 80, 2
# The grid models read any batch; the binned model assumes what every cached
# scan holds, one channel cell per dwell and one cell per AP per dwell on its
# channel, which a random batch does not.
GRID_MODELS = (JointAPModel, TargetAPModel)
MODELS = GRID_MODELS + (BinnedModel,)


def make_batch(aps=4, cells=24, seed=0) -> CellBatch:
    g = torch.Generator().manual_seed(seed)
    member_mask = torch.rand(SCANS, cells, MEMBERS, generator=g) > 0.3
    member_mask[0, 0] = False                      # a dwell that decoded nothing
    return CellBatch(
        frames=torch.randn(SCANS, FRAMES, FRAME_FEATURES, generator=g),
        frame_categories=torch.randint(0, 3, (SCANS, FRAMES), generator=g),
        cell_members=torch.randint(0, FRAMES, (SCANS, cells, MEMBERS), generator=g),
        cell_member_mask=member_mask,
        cell_aggregates=torch.randn(SCANS, cells, AGGREGATES, generator=g),
        cell_cca=torch.rand(SCANS, cells, CCA_SAMPLES, generator=g),
        cell_kind=(torch.rand(SCANS, cells, generator=g) > 0.5).long(),
        cell_dwell=torch.randint(0, 52, (SCANS, cells), generator=g),
        cell_channel=CHANNELS[torch.randint(0, 4, (SCANS, cells), generator=g)],
        cell_slot=torch.randint(0, aps, (SCANS, cells), generator=g),
        cell_mask=torch.ones(SCANS, cells, dtype=torch.bool),
        descriptors=torch.randn(SCANS, aps, DESCRIPTOR, generator=g),
        ap_channel=CHANNELS[torch.randint(0, 4, (SCANS, aps), generator=g)],
        ap_slot=torch.arange(aps).repeat(SCANS, 1),
        ap_mask=torch.ones(SCANS, aps, dtype=torch.bool))


def with_padded_cells(batch: CellBatch, extra: int) -> CellBatch:
    g = torch.Generator().manual_seed(99)
    fields = {}
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if not field.name.startswith("cell_"):
            fields[field.name] = value
            continue
        # Padding still has to be in range: the model looks its tags up in
        # embeddings before ever consulting the mask.
        limits = {"cell_kind": 2, "cell_dwell": 52, "cell_slot": 16,
                  "cell_members": FRAMES}
        shape = (value.shape[0], extra) + value.shape[2:]
        if field.name == "cell_mask":
            tail = torch.zeros(shape, dtype=torch.bool)
        elif value.dtype == torch.bool:
            tail = torch.rand(shape, generator=g) > 0.5
        elif value.dtype.is_floating_point:
            tail = torch.randn(shape, generator=g)
        elif field.name == "cell_channel":
            tail = CHANNELS[torch.randint(0, 4, shape, generator=g)]
        else:
            tail = torch.randint(0, limits[field.name], shape, generator=g,
                                 dtype=value.dtype)
        fields[field.name] = torch.cat([value, tail], dim=1)
    return CellBatch(**fields)


def with_padded_aps(batch: CellBatch, extra: int) -> CellBatch:
    g = torch.Generator().manual_seed(7)
    return dataclasses.replace(
        batch,
        descriptors=torch.cat([batch.descriptors,
                               torch.randn(SCANS, extra, DESCRIPTOR, generator=g)], dim=1),
        ap_channel=torch.cat(
            [batch.ap_channel,
             CHANNELS[torch.randint(0, 4, (SCANS, extra), generator=g)]], dim=1),
        ap_slot=torch.cat(
            [batch.ap_slot,
             torch.randint(0, 16, (SCANS, extra), generator=g)], dim=1),
        ap_mask=torch.cat([batch.ap_mask,
                               torch.zeros(SCANS, extra, dtype=torch.bool)], dim=1))


def model_for(model_class, seed=0):
    torch.manual_seed(seed)
    if model_class is BinnedModel:
        return BinnedModel(AGGREGATES).eval()
    return model_class(FRAME_FEATURES, AGGREGATES, DESCRIPTOR).eval()


def two_channel_scan() -> CellBatch:
    """One scan of three dwells on channels 36, 40 and 36, and two valid APs
    in slots 0 and 9, on channels 36 and 40; channel cells carry slot 0 too, as
    make_batch writes them. Cell i's aggregates are all i + 1."""
    kind = torch.tensor([[CHANNEL_CELL, CHANNEL_CELL, CHANNEL_CELL, AP_CELL, AP_CELL, AP_CELL]])
    cells = kind.shape[1]
    return CellBatch(
        frames=torch.zeros(1, 1, FRAME_FEATURES),
        frame_categories=torch.zeros(1, 1, dtype=torch.long),
        cell_members=torch.zeros(1, cells, 1, dtype=torch.long),
        cell_member_mask=torch.zeros(1, cells, 1, dtype=torch.bool),
        cell_aggregates=torch.arange(1.0, cells + 1)[None, :, None].repeat(1, 1, AGGREGATES),
        cell_cca=torch.zeros(1, cells, CCA_SAMPLES),
        cell_kind=kind,
        cell_dwell=torch.tensor([[0, 1, 2, 0, 2, 1]]),
        cell_channel=torch.tensor([[36, 40, 36, 36, 36, 40]]),
        cell_slot=torch.tensor([[0, 0, 0, 0, 0, 9]]),
        cell_mask=torch.ones(1, cells, dtype=torch.bool),
        descriptors=torch.zeros(1, 2, DESCRIPTOR),
        ap_channel=torch.tensor([[36, 40]]),
        ap_slot=torch.tensor([[0, 9]]),
        ap_mask=torch.ones(1, 2, dtype=torch.bool))


def scores(model, batch: CellBatch) -> torch.Tensor:
    with torch.no_grad():
        return model(batch)[0]


class InvarianceTests(unittest.TestCase):
    """Properties both models share."""

    def test_aps_permute_with_their_slots(self):
        batch = make_batch(aps=4)
        order = torch.tensor([2, 0, 3, 1])
        shuffled = dataclasses.replace(
            batch,
            descriptors=batch.descriptors[:, order],
            ap_channel=batch.ap_channel[:, order],
            ap_slot=batch.ap_slot[:, order],
            ap_mask=batch.ap_mask[:, order])
        for model_class in MODELS:
            with self.subTest(model_class.__name__):
                model = model_for(model_class)
                self.assertTrue(torch.allclose(scores(model, batch)[:, order],
                                               scores(model, shuffled), atol=1e-5))

    def test_cells_in_any_order_score_the_same(self):
        batch = make_batch(aps=4)
        order = torch.randperm(batch.cell_mask.shape[1], generator=torch.Generator().manual_seed(1))
        shuffled = dataclasses.replace(batch, **{
            field.name: getattr(batch, field.name)[:, order]
            for field in dataclasses.fields(batch) if field.name.startswith("cell_")})
        for model_class in GRID_MODELS:
            with self.subTest(model_class.__name__):
                model = model_for(model_class)
                self.assertTrue(torch.allclose(scores(model, batch), scores(model, shuffled),
                                               atol=1e-5))

    def test_padded_aps_leave_the_real_ones_alone(self):
        batch = make_batch(aps=3)
        for model_class in MODELS:
            with self.subTest(model_class.__name__):
                model = model_for(model_class)
                padded = scores(model, with_padded_aps(batch, 5))
                self.assertTrue(torch.allclose(scores(model, batch), padded[:, :3], atol=1e-5))

    def test_padded_cells_leave_the_scores_alone(self):
        batch = make_batch(aps=4, cells=20)
        for model_class in MODELS:
            with self.subTest(model_class.__name__):
                model = model_for(model_class)
                padded = scores(model, with_padded_cells(batch, 6))
                self.assertTrue(torch.allclose(scores(model, batch), padded, atol=1e-5))

    def test_frames_a_cell_does_not_own_never_reach_it(self):
        batch = make_batch(aps=4)
        g = torch.Generator().manual_seed(5)
        rewritten = torch.randint(0, FRAMES, batch.cell_members.shape, generator=g)
        elsewhere = dataclasses.replace(
            batch,
            cell_members=torch.where(batch.cell_member_mask, batch.cell_members,
                                     rewritten))
        for model_class in MODELS:
            with self.subTest(model_class.__name__):
                model = model_for(model_class)
                self.assertTrue(torch.allclose(scores(model, batch), scores(model, elsewhere),
                                               atol=1e-5))

    def test_an_aps_channel_reaches_its_score(self):
        batch = make_batch(aps=4)
        moved = dataclasses.replace(
            batch, ap_channel=torch.full_like(batch.ap_channel, 48))
        for model_class in GRID_MODELS:
            with self.subTest(model_class.__name__):
                model = model_for(model_class)
                # The distance bias starts at zero, so it has no effect at all
                # until training gives it one. Setting it here tests the wiring.
                with torch.no_grad():
                    model.distance_bias.copy_(
                        torch.tensor([0.0, -1.0, -2.0, -3.0]).repeat(model.heads, 1))
                self.assertFalse(torch.allclose(scores(model, batch), scores(model, moved),
                                                atol=1e-5))

    def test_every_ap_count_and_grid_size_runs(self):
        for model_class in MODELS:
            for aps in (2, 5, 8):
                for cells in (13, 52, 156):
                    batch = make_batch(aps=aps, cells=cells, seed=aps + cells)
                    mu, log_var = model_for(model_class)(batch)
                    self.assertEqual(mu.shape, (SCANS, aps))
                    self.assertTrue(torch.isfinite(mu).all())
                    self.assertTrue(torch.isfinite(log_var).all())


class JointAgainstTargetTests(unittest.TestCase):
    def test_a_rivals_descriptor_reaches_the_joint_score_and_not_the_target_score(self):
        batch = make_batch(aps=4)
        descriptors = batch.descriptors.clone()
        descriptors[:, 1:] = torch.randn(descriptors[:, 1:].shape,
                                         generator=torch.Generator().manual_seed(3))
        rivals_changed = dataclasses.replace(batch, descriptors=descriptors)
        joint, target = model_for(JointAPModel), model_for(TargetAPModel)
        self.assertFalse(torch.allclose(scores(joint, batch)[:, 0],
                                        scores(joint, rivals_changed)[:, 0], atol=1e-5))
        self.assertTrue(torch.allclose(scores(target, batch)[:, 0],
                                       scores(target, rivals_changed)[:, 0], atol=1e-5))

    def test_ap_cells_take_their_role_from_the_target(self):
        cell_channel = torch.tensor([[36, 36, 40, 36]])
        cell_slot = torch.tensor([[2, 5, 5, 7]])
        found = roles(torch.tensor([[36]]), torch.tensor([[2]]), cell_channel, cell_slot)
        self.assertEqual(found.tolist(), [[TARGET_AP, CO_CHANNEL_AP, OTHER_AP, CO_CHANNEL_AP]])


class BinnedViewTests(unittest.TestCase):
    def test_each_ap_sees_every_dwell_and_its_own_cells_on_its_channel(self):
        steps, exists = binned_view(two_channel_scan(), n_dwells=4)
        self.assertEqual(exists.tolist(), [[True, True, True, False]])
        # channel cells hold 1, 2, 3 by dwell; AP cells 4 and 5 are slot 0's,
        # at dwells 0 and 2, and 6 is slot 9's, at dwell 1
        heard = steps[0, :, :3, 0]
        own = steps[0, :, :3, AGGREGATES]
        flags = steps[0, :, :3, 2 * AGGREGATES:]
        self.assertEqual(heard.tolist(), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        self.assertEqual(own.tolist(), [[4.0, 0.0, 5.0], [0.0, 6.0, 0.0]])
        self.assertEqual(flags[..., 0].tolist(), [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertEqual(flags[..., 1].tolist(), [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    def test_cells_in_any_order_give_the_same_view(self):
        batch = two_channel_scan()
        order = torch.tensor([4, 2, 5, 0, 3, 1])
        shuffled = dataclasses.replace(batch, **{
            field.name: getattr(batch, field.name)[:, order]
            for field in dataclasses.fields(batch) if field.name.startswith("cell_")})
        self.assertTrue(torch.equal(binned_view(batch, 4)[0], binned_view(shuffled, 4)[0]))


class LossTests(unittest.TestCase):
    def test_padded_aps_do_not_enter_the_loss(self):
        mu = torch.zeros(2, 3)
        log_var = torch.zeros(2, 3)
        target = torch.tensor([[1.0, 1.0, 50.0], [1.0, 1.0, -50.0]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        self.assertAlmostEqual(float(gaussian_nll(mu, log_var, target, mask)), 0.5)

    def test_each_scan_weighs_the_same_whatever_its_ap_count(self):
        mu, log_var = torch.zeros(2, 4), torch.zeros(2, 4)
        # The one-AP scan is far off and the four-AP scan is close, so averaging
        # within a scan before averaging over scans gives 9.25, where one flat
        # mean over every valid AP would give 4.0.
        target = torch.tensor([[6.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        few = torch.tensor([[True, False, False, False], [True, True, True, True]])
        loss = gaussian_nll(mu, log_var, target, few, variance=False)
        self.assertAlmostEqual(float(loss), 9.25)

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
