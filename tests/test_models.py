from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.data import (scan_sample_weights, split_by_topology,
                         split_rows_by_topology)
from models.evaluate import (baseline_predictions, fit_rssi_busy_k,
                             random_selection_metrics, selection_metrics)
from models.frames import FrameCorpus
from models.ranker import SetRanker, pack_scans, pointwise_loss, ranking_loss
from models.temporal import (TemporalCorpus, TemporalSetTransformer,
                             load_temporal_checkpoint)
from scripts.simulate.run_sweep import Run, already_done
from scripts.dataset.combine_datasets import combine_csv, combine_npz
from scripts.train.learning_curve import subset_topologies
from scripts.train.train_eval import choose_by_validation_regret, stratified_report


class ScanSplitTests(unittest.TestCase):
    def test_repeated_seeds_of_topology_never_cross_splits(self):
        rows = []
        for topology in range(20):
            for seed in range(5):
                for ap in range(2):
                    rows.append({
                        "topology_id": f"g{topology:02d}",
                        "scan_id": f"g{topology:02d}__s{seed:02d}",
                        "ap_index": ap,
                        "label_throughput_mbps": float(ap),
                    })
        frame = pd.DataFrame(rows)
        train, val, test = split_by_topology(frame, seed=7)
        topology_sets = [set(part.topology_id) for part in (train, val, test)]
        self.assertTrue(topology_sets[0].isdisjoint(topology_sets[1]))
        self.assertTrue(topology_sets[0].isdisjoint(topology_sets[2]))
        self.assertTrue(topology_sets[1].isdisjoint(topology_sets[2]))

    def test_too_few_topologies_fails_clearly(self):
        frame = pd.DataFrame({
            "topology_id": ["a", "a", "b", "b"],
            "scan_id": ["a", "a", "b", "b"],
            "ap_index": [0, 1, 0, 1],
            "label_throughput_mbps": [1.0, 0.0, 1.0, 0.0],
        })
        with self.assertRaisesRegex(ValueError, "at least 5 independent"):
            split_by_topology(frame)

    def test_sample_weights_give_each_choice_set_equal_mass(self):
        frame = pd.DataFrame({
            "scan_id": ["a", "a", "b", "b", "b", "b"],
        })
        frame["weight"] = scan_sample_weights(frame)
        totals = frame.groupby("scan_id").weight.sum()
        self.assertAlmostEqual(float(totals["a"]), float(totals["b"]))
        self.assertAlmostEqual(float(frame.weight.mean()), 1.0)

    def test_large_dataset_stratifies_every_ap_count(self):
        rows = []
        for n_aps in (2, 3, 4, 6, 8):
            for topology in range(10):
                topology_id = f"n{n_aps}_g{topology}"
                for ap in range(2):
                    rows.append({
                        "topology_id": topology_id,
                        "scan_id": topology_id,
                        "ap_index": ap,
                        "gt_n_aps": n_aps,
                        "label_throughput_mbps": float(ap),
                    })
        train, val, test = split_by_topology(pd.DataFrame(rows), seed=4)
        for part in (train, val, test):
            self.assertEqual(set(part.gt_n_aps), {2, 3, 4, 6, 8})

    def test_learning_curve_subsets_are_nested_by_topology(self):
        rows = []
        for n_aps in (2, 3):
            for topology in range(10):
                topology_id = f"n{n_aps}_g{topology}"
                for seed in range(3):
                    rows.append({
                        "topology_id": topology_id,
                        "scan_id": f"{topology_id}_s{seed}",
                        "gt_n_aps": n_aps,
                    })
        frame = pd.DataFrame(rows)
        small = subset_topologies(frame, 0.25, seed=8)
        large = subset_topologies(frame, 0.5, seed=8)
        self.assertTrue(set(small.topology_id) <= set(large.topology_id))
        self.assertTrue((small.groupby("topology_id").size() == 3).all())


class DatasetCombinationTests(unittest.TestCase):
    def test_csv_combination_rejects_duplicate_topologies(self):
        columns = ["topology_id", "scan_id", "ap_index", "run_id",
                   "label_throughput_mbps"]
        first = pd.DataFrame([["a", "a_s0", 0, "a0", 1.0]], columns=columns)
        second = pd.DataFrame([["b", "b_s0", 0, "b0", 2.0]], columns=columns)
        duplicate = pd.DataFrame([["a", "a_s1", 0, "a1", 3.0]], columns=columns)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.csv", root / "b.csv", root / "duplicate.csv"]
            for frame, path in zip((first, second, duplicate), paths):
                frame.to_csv(path, index=False)
            out = root / "combined.csv"
            combine_csv(paths[:2], out)
            self.assertEqual(set(pd.read_csv(out).topology_id), {"a", "b"})
            with self.assertRaisesRegex(ValueError, "duplicate topology IDs"):
                combine_csv([paths[0], paths[2]], out)

    def test_temporal_combination_pads_compatible_corpora(self):
        def payload(prefix, options, steps):
            return {
                "schema_version": np.array(2, np.int16),
                "temporal": np.ones((1, options, steps, 2), np.float32),
                "static": np.ones((1, options, 3), np.float32),
                "labels": np.ones((1, options), np.float32),
                "option_indices": np.arange(options, dtype=np.int16)[None],
                "option_mask": np.ones((1, options), bool),
                "time_mask": np.ones((1, steps), bool),
                "scan_ids": np.array([f"{prefix}_s0"]),
                "topology_ids": np.array([prefix]),
                "configured_n_aps": np.array([options], np.int16),
                "n_hotspots": np.array([0], np.int16),
                "candidate_strata": np.array(["boundary"]),
                "temporal_features": np.array(["t0", "t1"]),
                "static_features": np.array(["s0", "s1", "s2"]),
                "bin_ms": np.array(10.0, np.float32),
                "scan_description": np.array("passive"),
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.npz", root / "b.npz"]
            np.savez_compressed(paths[0], **payload("a", 2, 3))
            np.savez_compressed(paths[1], **payload("b", 3, 4))
            out = root / "combined.npz"
            combine_npz(paths, out)
            with np.load(out, allow_pickle=False) as combined:
                self.assertEqual(combined["temporal"].shape, (2, 3, 4, 2))
                self.assertEqual(int(combined["option_mask"].sum()), 5)
                self.assertFalse(bool(combined["option_mask"][0, 2]))
                self.assertFalse(bool(combined["time_mask"][0, 3]))


class SweepResumeTests(unittest.TestCase):
    def test_resume_requires_matching_scenario_and_complete_shared_trace(self):
        topology = {
            "nAPs": 2,
            "nSTAs": 4,
            "hotspotAPs": "1",
            "topologySeed": 19,
        }
        candidate = {"candidateStratum": "boundary", "candidateSeed": 77}
        metadata = {
            "rng_seed": 23,
            "params": {
                "topology_seed": 19,
                "n_aps": 2,
                "n_stas": 4,
                "bg_mean_per_sta_mbps": 2.5,
                "hotspot_aps": [1],
            },
            "candidate_seed": 77,
            "candidate_stratum": "boundary",
            "candidate_position": {"x": 12.5, "y": 3.0},
            "candidate": {"target_ap": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "t00000__c00__ap0"
            run.mkdir()
            (run / "metadata.json").write_text(json.dumps(metadata))
            (run / "observation.csv").write_text("x" * 1001)
            (run / "chanbusy.csv").write_text("x" * 101)
            job = Run(run.name, topology, candidate, 0, 23)
            self.assertTrue(already_done(root, job))
            self.assertFalse(already_done(root, job._replace(rng_seed=24)))
            # a different candidate seed is a different position, not a resume
            self.assertFalse(already_done(root, job._replace(
                candidate=dict(candidate, candidateSeed=99))))
            self.assertFalse(already_done(root, job._replace(
                candidate=dict(candidate, candidateStratum="ap_near"))))
            (run / "chanbusy.csv").write_text("short")
            self.assertFalse(already_done(root, job))


class RankingLossTests(unittest.TestCase):
    def test_correct_order_has_lower_loss(self):
        labels = torch.tensor([[20.0, 10.0, 9.8, 0.0]])
        mask = torch.tensor([[True, True, True, False]])
        correct = torch.tensor([[2.0, 1.0, 0.0, 99.0]])
        reversed_order = torch.tensor([[0.0, 1.0, 2.0, -99.0]])
        self.assertLess(ranking_loss(correct, labels, mask),
                        ranking_loss(reversed_order, labels, mask))

    def test_near_ties_and_padding_are_ignored(self):
        labels = torch.tensor([[1.0, 1.2, 999.0]])
        mask = torch.tensor([[True, True, False]])
        scores = torch.tensor([[0.0, 100.0, -100.0]], requires_grad=True)
        loss = ranking_loss(scores, labels, mask)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_pointwise_loss_weights_groups_not_rows(self):
        scores = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                               [3.0, 3.0, 3.0, 3.0]])
        labels = torch.zeros_like(scores)
        mask = torch.tensor([[True, True, False, False],
                             [True, True, True, True]])
        self.assertAlmostEqual(float(pointwise_loss(scores, labels, mask)), 5.0)


class SetRankerTests(unittest.TestCase):
    def test_option_permutation_equivariance_and_padding_isolation(self):
        torch.manual_seed(3)
        model = SetRanker(5, hidden=12, embed=8, dropout=0.0).eval()
        values = torch.randn(1, 4, 5)
        mask = torch.tensor([[True, True, True, False]])
        with torch.no_grad():
            reference = model(values, mask)

            permutation = torch.tensor([2, 0, 1, 3])
            permuted = model(values[:, permutation], mask[:, permutation])
            changed_padding = values.clone()
            changed_padding[:, 3] = 1e6
            padded = model(changed_padding, mask)

        inverse = torch.argsort(permutation)
        self.assertTrue(torch.allclose(reference[:, :3],
                                       permuted[:, inverse][:, :3], atol=1e-6))
        self.assertTrue(torch.allclose(reference[:, :3], padded[:, :3], atol=1e-6))


class EvaluationTests(unittest.TestCase):
    def test_validation_selection_uses_equal_topology_weight(self):
        frame = pd.DataFrame({
            "topology_id": ["a", "a", "a", "a", "b", "b"],
            "scan_id": ["a0", "a0", "a1", "a1", "b0", "b0"],
            "label_throughput_mbps": [10.0, 0.0, 10.0, 0.0, 14.0, 0.0],
        })
        topology_fair = np.array([0, 1, 0, 1, 1, 0])
        group_fair = np.array([1, 0, 1, 0, 0, 1])
        _, selected = choose_by_validation_regret(
            [("topology_fair", topology_fair), ("group_fair", group_fair)], frame)
        self.assertEqual(selected, "topology_fair")

    def test_equal_scores_use_order_invariant_expected_tie_break(self):
        frame = pd.DataFrame({
            "scan_id": ["g", "g"],
            "label_throughput_mbps": [10.0, 0.0],
        })
        metrics = selection_metrics(frame, np.array([1.0, 1.0]))
        shuffled = selection_metrics(frame.iloc[::-1], np.array([1.0, 1.0]))
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_regret_mbps"], 5.0)
        self.assertEqual(metrics, shuffled)

    def test_topology_regret_does_not_overweight_extra_seed_groups(self):
        frame = pd.DataFrame({
            "topology_id": ["a", "a", "a", "a", "b", "b"],
            "scan_id": ["a0", "a0", "a1", "a1", "b0", "b0"],
            "label_throughput_mbps": [10.0, 0.0, 10.0, 0.0, 10.0, 0.0],
        })
        metrics = selection_metrics(frame, np.array([0, 1, 0, 1, 1, 0]))
        self.assertAlmostEqual(metrics["mean_regret_mbps"], 20.0 / 3.0)
        self.assertAlmostEqual(metrics["topology_mean_regret_mbps"], 5.0)
        self.assertEqual(metrics["topologies"], 2)

    def test_busy_baselines_prefer_cca_over_decoded_airtime(self):
        frame = pd.DataFrame({
            "scan_id": ["g", "g"],
            "ap_index": [0, 1],
            "label_throughput_mbps": [1.0, 2.0],
            "feat_ap_rssi_mean": [-60.0, -60.0],
            "feat_chan_busy_frac": [0.1, 0.9],
            "feat_chan_cca_busy_frac": [0.8, 0.2],
        })
        predictions = baseline_predictions(frame, fit_frame=frame)
        self.assertGreater(predictions["least_busy_channel"][1],
                           predictions["least_busy_channel"][0])
        self.assertGreater(predictions["rssi_minus_busy"][1],
                           predictions["rssi_minus_busy"][0])

    def test_nonfinite_predictions_are_rejected(self):
        frame = pd.DataFrame({
            "scan_id": ["a", "a"],
            "label_throughput_mbps": [1.0, 0.0],
        })
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            selection_metrics(frame, np.array([0.0, np.nan]))

    def test_random_metrics_are_exact_and_row_order_invariant(self):
        frame = pd.DataFrame({
            "scan_id": ["a", "a", "b", "b", "b"],
            "ap_index": [0, 1, 0, 1, 2],
            "label_throughput_mbps": [10.0, 0.0, 9.0, 6.0, 0.0],
        })
        metrics = random_selection_metrics(frame)
        self.assertAlmostEqual(metrics["mean_regret_mbps"], (5.0 + 4.0) / 2)
        shuffled = frame.sample(frac=1.0, random_state=2)
        self.assertEqual(metrics, random_selection_metrics(shuffled))

        original = dict(zip(zip(frame.scan_id, frame.ap_index),
                            baseline_predictions(frame, frame)["random"]))
        reordered = dict(zip(zip(shuffled.scan_id, shuffled.ap_index),
                             baseline_predictions(shuffled, shuffled)["random"]))
        self.assertEqual(original, reordered)

    def test_pooled_strata_keep_repeated_split_scans_separate(self):
        frame = pd.DataFrame({
            "split_seed": [0, 0, 1, 1],
            "scan_id": ["g", "g", "g", "g"],
            "stratum": ["x", "x", "x", "x"],
            "label_throughput_mbps": [10.0, 0.0, 10.0, 0.0],
            "pred_model": [1.0, 0.0, 0.0, 1.0],
            "pred_strongest_rssi": [1.0, 0.0, 1.0, 0.0],
        })
        report = stratified_report(frame, ["model"])
        self.assertEqual(int(report.loc[0, "scans"]), 2)
        self.assertAlmostEqual(float(report.loc[0, "model"]), 5.0)


class TemporalTransformerTests(unittest.TestCase):
    def test_checkpoint_loads_through_safe_weights_only_path(self):
        model = TemporalSetTransformer(3, 4, n_static_features=2,
                                       model_dim=8, heads=2,
                                       temporal_layers=1, set_layers=1)
        payload = {
            "state_dict": model.state_dict(),
            "objective": "regression",
            "model": {
                "n_temporal_features": 3,
                "max_steps": 4,
                "n_static_features": 2,
                "model_dim": 8,
                "heads": 2,
                "temporal_layers": 1,
                "set_layers": 1,
                "dropout": 0.1,
            },
            "temporal_features": ["a", "b", "c"],
            "static_features": ["x", "y"],
            "scaler_mean": torch.zeros(3),
            "scaler_std": torch.ones(3),
            "static_scaler_mean": torch.zeros(2),
            "static_scaler_std": torch.ones(2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(payload, path)
            loaded, metadata = load_temporal_checkpoint(path)
        self.assertIsInstance(loaded, TemporalSetTransformer)
        self.assertEqual(metadata["objective"], "regression")

    def setUp(self):
        torch.manual_seed(3)
        self.model = TemporalSetTransformer(
            n_temporal_features=6, max_steps=5, n_static_features=3,
            model_dim=16, heads=4,
            temporal_layers=1, set_layers=1, dropout=0.0)
        self.model.eval()
        self.temporal = torch.randn(2, 4, 5, 6)
        self.option_mask = torch.tensor([
            [True, True, True, False],
            [True, True, True, True],
        ])
        self.time_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, True],
        ])
        self.static = torch.randn(2, 4, 3)

    def test_option_permutation_equivariance(self):
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            original = self.model(self.temporal, self.option_mask, self.time_mask,
                                  self.static)
            permuted = self.model(self.temporal[:, permutation],
                                  self.option_mask[:, permutation], self.time_mask,
                                  self.static[:, permutation])
        np.testing.assert_allclose(
            permuted.numpy(), original[:, permutation].numpy(), rtol=1e-5, atol=1e-6)

    def test_masked_values_cannot_change_real_scores(self):
        changed = self.temporal.clone()
        changed[0, 3] = 1e6
        changed[0, :3, 3:] = -1e6
        with torch.no_grad():
            original = self.model(self.temporal, self.option_mask, self.time_mask,
                                  self.static)
            changed_static = self.static.clone()
            changed_static[0, 3] = 1e6
            altered = self.model(changed, self.option_mask, self.time_mask,
                                 changed_static)
        np.testing.assert_allclose(
            altered[0, :3].numpy(), original[0, :3].numpy(), rtol=1e-5, atol=1e-6)


class CorpusSplitTests(unittest.TestCase):
    """The array corpora split by the same rule as the flat dataframe."""

    @staticmethod
    def _ids(n_topologies=10, seeds_per_topology=3):
        topology_ids = np.array([f"t{t:02d}"
                                 for t in range(n_topologies)
                                 for _ in range(seeds_per_topology)])
        # two AP counts, each with enough topologies to stratify
        configured = np.array([2 if t < n_topologies // 2 else 4
                               for t in range(n_topologies)
                               for _ in range(seeds_per_topology)])
        return topology_ids, configured

    def test_every_group_lands_in_exactly_one_part(self):
        topology_ids, configured = self._ids()
        parts = split_rows_by_topology(topology_ids, configured, seed=1)
        covered = np.concatenate(parts)
        self.assertEqual(sorted(covered.tolist()), list(range(len(topology_ids))))

    def test_repeated_seeds_of_a_topology_never_cross_parts(self):
        topology_ids, configured = self._ids()
        for seed in range(4):
            parts = split_rows_by_topology(topology_ids, configured, seed=seed)
            seen = [set(topology_ids[part]) for part in parts]
            self.assertTrue(seen[0].isdisjoint(seen[1]))
            self.assertTrue(seen[0].isdisjoint(seen[2]))
            self.assertTrue(seen[1].isdisjoint(seen[2]))

    def test_too_few_topologies_fails_clearly(self):
        topology_ids = np.array(["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "at least 5 independent"):
            split_rows_by_topology(topology_ids, None)

    def _temporal_corpus(self):
        topology_ids, configured = self._ids()
        n = len(topology_ids)
        return TemporalCorpus(
            temporal=np.zeros((n, 2, 3, 4), np.float32),
            static=np.zeros((n, 2, 1), np.float32),
            labels=np.zeros((n, 2), np.float32),
            option_indices=np.zeros((n, 2), np.int16),
            option_mask=np.ones((n, 2), bool),
            time_mask=np.ones((n, 3), bool),
            scan_ids=np.array([f"g{i}" for i in range(n)]),
            topology_ids=topology_ids, configured_n_aps=configured,
            n_hotspots=np.zeros(n, np.int16),
            candidate_strata=np.array(["boundary"] * n),
            temporal_features=["a", "b", "c", "d"], static_features=["s"])

    def _frame_corpus(self):
        topology_ids, configured = self._ids()
        n = len(topology_ids)
        return FrameCorpus(
            frames=np.zeros((n, 5, 2), np.float32),
            relations=np.zeros((n, 2, 5), np.uint8),
            frame_mask=np.ones((n, 5), bool),
            static=np.zeros((n, 2, 1), np.float32),
            labels=np.zeros((n, 2), np.float32),
            option_indices=np.zeros((n, 2), np.int16),
            option_mask=np.ones((n, 2), bool),
            scan_ids=np.array([f"g{i}" for i in range(n)]),
            topology_ids=topology_ids, configured_n_aps=configured,
            n_hotspots=np.zeros(n, np.int16),
            candidate_strata=np.array(["boundary"] * n),
            frame_features=["a", "b"], static_features=["s"])

    def test_binned_and_frame_corpora_split_identically(self):
        temporal, frames = self._temporal_corpus(), self._frame_corpus()
        for seed in range(3):
            for from_temporal, from_frames in zip(temporal.split(seed=seed),
                                                  frames.split(seed=seed)):
                np.testing.assert_array_equal(from_temporal, from_frames)

    def test_corpus_split_keeps_both_ap_counts_in_every_part(self):
        corpus = self._temporal_corpus()
        for part in corpus.split(seed=2):
            self.assertEqual(set(corpus.configured_n_aps[part].tolist()), {2, 4})


class SpearmanTests(unittest.TestCase):
    def test_a_model_that_orders_nothing_scores_zero_rather_than_being_excluded(self):
        frame = pd.DataFrame({
            "scan_id": ["a", "a", "b", "b"],
            "label_throughput_mbps": [10.0, 0.0, 10.0, 0.0],
        })
        # perfect on set "a", completely undecided on set "b"
        metrics = selection_metrics(frame, np.array([1.0, 0.0, 5.0, 5.0]))
        self.assertAlmostEqual(metrics["mean_spearman"], 0.5)

    def test_a_choice_set_with_nothing_to_order_is_left_out_of_the_average(self):
        frame = pd.DataFrame({
            "scan_id": ["a", "a", "b", "b"],
            "label_throughput_mbps": [10.0, 0.0, 7.0, 7.0],
        })
        # set "b" has identical labels, so there is no ordering to get right
        metrics = selection_metrics(frame, np.array([1.0, 0.0, 1.0, 0.0]))
        self.assertAlmostEqual(metrics["mean_spearman"], 1.0)


class BusyHeuristicFitTests(unittest.TestCase):
    @staticmethod
    def _frame():
        # signal is identical everywhere, so only occupancy can decide
        return pd.DataFrame({
            "scan_id": ["a", "a", "b", "b"],
            "ap_index": [0, 1, 0, 1],
            "label_throughput_mbps": [1.0, 9.0, 8.0, 2.0],
            "feat_ap_rssi_mean": [-60.0, -60.0, -60.0, -60.0],
            "feat_chan_cca_busy_frac": [0.9, 0.1, 0.1, 0.9],
        })

    def test_fitted_k_beats_ignoring_occupancy(self):
        frame = self._frame()
        k = fit_rssi_busy_k(frame, "feat_chan_cca_busy_frac")
        self.assertGreater(k, 0.0)
        scores = frame.feat_ap_rssi_mean - k * frame.feat_chan_cca_busy_frac
        self.assertAlmostEqual(
            selection_metrics(frame, scores.to_numpy())["mean_regret_mbps"], 0.0)

    def test_an_unusable_fit_frame_raises_instead_of_defaulting(self):
        with self.assertRaisesRegex(ValueError, "empty frame"):
            fit_rssi_busy_k(self._frame().iloc[:0], "feat_chan_cca_busy_frac")
        with self.assertRaisesRegex(ValueError, "missing"):
            fit_rssi_busy_k(self._frame().drop(columns=["feat_ap_rssi_mean"]),
                            "feat_chan_cca_busy_frac")


class MakeGroupsTests(unittest.TestCase):
    def test_a_frame_that_was_not_reindexed_is_refused(self):
        frame = pd.DataFrame({
            "scan_id": ["a", "a"],
            "label_throughput_mbps": [1.0, 2.0],
            "feat_x": [0.0, 1.0],
        }, index=[7, 8])
        with self.assertRaisesRegex(ValueError, "indexed 0..n-1"):
            pack_scans(frame, ["feat_x"])


if __name__ == "__main__":
    unittest.main()
