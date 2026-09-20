from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.splits import N_FOLDS, split_by_topology, split_rows_by_topology
from scripts.common.evaluate import selection_metrics
from scripts.baselines.train import fit_rssi_busy_k, heuristic_predictions
from scripts.simulate.run_sweep import Run, already_done
from scripts.baselines.train import (choose_by_validation_regret, scan_sample_weights,
                                 stratified_report, subset_topologies)


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
                        "gt_n_aps": 2,
                        "label_throughput_mbps": float(ap),
                    })
        frame = pd.DataFrame(rows)
        train, val, test = split_by_topology(frame, test_fold=3)
        topology_sets = [set(part.topology_id) for part in (train, val, test)]
        self.assertTrue(topology_sets[0].isdisjoint(topology_sets[1]))
        self.assertTrue(topology_sets[0].isdisjoint(topology_sets[2]))
        self.assertTrue(topology_sets[1].isdisjoint(topology_sets[2]))

    def test_too_few_topologies_fails_clearly(self):
        frame = pd.DataFrame({
            "topology_id": ["a", "a", "b", "b"],
            "scan_id": ["a", "a", "b", "b"],
            "ap_index": [0, 1, 0, 1],
            "gt_n_aps": [2, 2, 2, 2],
            "label_throughput_mbps": [1.0, 0.0, 1.0, 0.0],
        })
        with self.assertRaisesRegex(ValueError, "every AP count needs at least 5"):
            split_by_topology(frame)

    def test_a_frame_without_ap_counts_fails_clearly(self):
        frame = pd.DataFrame({
            "topology_id": [f"g{i:02d}" for i in range(10)],
            "scan_id": [f"g{i:02d}" for i in range(10)],
            "ap_index": [0] * 10,
            "label_throughput_mbps": [1.0] * 10,
        })
        with self.assertRaisesRegex(ValueError, "no gt_n_aps column"):
            split_by_topology(frame)

    def test_sample_weights_give_each_scan_equal_mass(self):
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
        train, val, test = split_by_topology(pd.DataFrame(rows), test_fold=4)
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


class SweepResumeTests(unittest.TestCase):
    def test_resume_requires_matching_scenario_and_complete_shared_trace(self):
        topology = {
            "nAPs": 2,
            "nSTAs": 4,
            "hotspotAPs": "1",
            "topologySeed": 19,
        }
        candidate = {"candidateSeed": 77}
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
            (run / "chanbusy.csv").write_text("short")
            self.assertFalse(already_done(root, job))


class EvaluationTests(unittest.TestCase):
    def test_validation_selection_prefers_the_lower_regret_candidate(self):
        frame = pd.DataFrame({
            "scan_id": ["a0", "a0", "a1", "a1", "b0", "b0"],
            "label_throughput_mbps": [10.0, 0.0, 10.0, 0.0, 14.0, 0.0],
        })
        two_of_three = np.array([1, 0, 1, 0, 0, 1])
        one_of_three = np.array([0, 1, 0, 1, 1, 0])
        _, selected = choose_by_validation_regret(
            [("one_of_three", one_of_three), ("two_of_three", two_of_three)], frame)
        self.assertEqual(selected, "two_of_three")

    def test_equal_scores_use_order_invariant_expected_tie_break(self):
        frame = pd.DataFrame({
            "scan_id": ["g", "g"],
            "label_throughput_mbps": [10.0, 0.0],
        })
        metrics = selection_metrics(frame, np.array([1.0, 1.0]))
        shuffled = selection_metrics(frame.iloc[::-1], np.array([1.0, 1.0]))
        self.assertAlmostEqual(metrics["mean_regret_mbps"], 5.0)
        self.assertEqual(metrics, shuffled)

    def test_busy_baselines_prefer_cca_over_decoded_airtime(self):
        frame = pd.DataFrame({
            "scan_id": ["g", "g"],
            "ap_index": [0, 1],
            "label_throughput_mbps": [1.0, 2.0],
            "feat_ap_rssi_mean": [-60.0, -60.0],
            "feat_chan_busy_frac": [0.1, 0.9],
            "feat_chan_cca_busy_frac": [0.8, 0.2],
        })
        predictions = heuristic_predictions(frame, fit_frame=frame)
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
        flat = heuristic_predictions(frame, frame)["random"]
        metrics = selection_metrics(frame, flat)
        # a flat score ties every AP, so regret is the mean over the set
        self.assertAlmostEqual(metrics["mean_regret_mbps"], (5.0 + 4.0) / 2)
        shuffled = frame.sample(frac=1.0, random_state=2)
        self.assertEqual(metrics, selection_metrics(
            shuffled, heuristic_predictions(shuffled, shuffled)["random"]))

        original = dict(zip(zip(frame.scan_id, frame.ap_index),
                            heuristic_predictions(frame, frame)["random"]))
        reordered = dict(zip(zip(shuffled.scan_id, shuffled.ap_index),
                             heuristic_predictions(shuffled, shuffled)["random"]))
        self.assertEqual(original, reordered)

    def test_pooled_strata_count_each_fold_scan_once(self):
        # The folds are disjoint, so the pooled frame holds a scan once per
        # fold it belongs to, and a stratum's scan count is its scans.
        frame = pd.DataFrame({
            "fold": [0, 0, 1, 1],
            "scan_id": ["g", "g", "h", "h"],
            "stratum": ["x", "x", "x", "x"],
            "label_throughput_mbps": [10.0, 0.0, 10.0, 0.0],
            "pred_model": [1.0, 0.0, 0.0, 1.0],
            "pred_strongest_rssi": [1.0, 0.0, 1.0, 0.0],
        })
        report = stratified_report(frame, ["model"])
        self.assertEqual(int(report.loc[0, "scans"]), 2)
        self.assertAlmostEqual(float(report.loc[0, "model"]), 5.0)


class CorpusSplitTests(unittest.TestCase):
    """Row positions split by the same rule as the flat dataframe."""

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
        parts = split_rows_by_topology(topology_ids, configured, test_fold=1)
        covered = np.concatenate(parts)
        self.assertEqual(sorted(covered.tolist()), list(range(len(topology_ids))))

    def test_repeated_seeds_of_a_topology_never_cross_parts(self):
        topology_ids, configured = self._ids()
        for fold in range(N_FOLDS):
            parts = split_rows_by_topology(topology_ids, configured, test_fold=fold)
            seen = [set(topology_ids[part]) for part in parts]
            self.assertTrue(seen[0].isdisjoint(seen[1]))
            self.assertTrue(seen[0].isdisjoint(seen[2]))
            self.assertTrue(seen[1].isdisjoint(seen[2]))

    def test_every_topology_is_tested_exactly_once_over_a_rotation(self):
        topology_ids, configured = self._ids()
        tested = []
        for fold in range(N_FOLDS):
            _, _, test = split_rows_by_topology(topology_ids, configured, test_fold=fold)
            tested.extend(np.unique(topology_ids[test]).tolist())
        self.assertEqual(sorted(tested), sorted(np.unique(topology_ids).tolist()))

    def test_a_topology_never_validates_and_tests_in_the_same_fold(self):
        topology_ids, configured = self._ids()
        for fold in range(N_FOLDS):
            _, val, test = split_rows_by_topology(topology_ids, configured, test_fold=fold)
            self.assertTrue(set(topology_ids[val]).isdisjoint(set(topology_ids[test])))

    def test_the_deal_does_not_depend_on_the_order_the_ids_arrive_in(self):
        topology_ids, configured = self._ids()
        order = np.argsort(topology_ids[::-1], kind="stable")
        shuffled, shuffled_n_aps = topology_ids[::-1][order], configured[::-1][order]
        _, _, test = split_rows_by_topology(topology_ids, configured, test_fold=0)
        _, _, other = split_rows_by_topology(shuffled, shuffled_n_aps, test_fold=0)
        self.assertEqual(set(topology_ids[test]), set(shuffled[other]))

    def test_too_few_folds_to_leave_a_training_part_fails_clearly(self):
        topology_ids, configured = self._ids()
        with self.assertRaisesRegex(ValueError, "at least 3 folds"):
            split_rows_by_topology(topology_ids, configured, test_fold=0, n_folds=2)

    def test_a_fold_outside_the_rotation_fails_clearly(self):
        topology_ids, configured = self._ids()
        with self.assertRaisesRegex(ValueError, "not one of the"):
            split_rows_by_topology(topology_ids, configured, test_fold=N_FOLDS)

    def test_too_few_topologies_fails_clearly(self):
        topology_ids = np.array(["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "every AP count needs at least 5"):
            split_rows_by_topology(topology_ids, np.array([2, 2, 4]))

    def test_an_ap_count_too_rare_to_reach_every_fold_fails_clearly(self):
        # ten topologies pass any check on the total, but five AP counts of two
        # each leave three of the five folds with nothing in them
        topology_ids = np.array([f"t{t:02d}" for t in range(10)])
        n_aps = np.array([2, 2, 3, 3, 4, 4, 6, 6, 8, 8])
        with self.assertRaisesRegex(ValueError, "the rarest has 2"):
            split_rows_by_topology(topology_ids, n_aps)

    def test_a_nan_ap_count_fails_clearly_rather_than_dropping_a_topology(self):
        topology_ids, configured = self._ids()
        n_aps = configured.astype(float)
        n_aps[0] = np.nan
        with self.assertRaisesRegex(ValueError, "AP count is NaN"):
            split_rows_by_topology(topology_ids, n_aps)

    def test_the_frame_and_array_splits_cut_topologies_the_same_way(self):
        topology_ids, configured = self._ids()
        frame = pd.DataFrame({"topology_id": topology_ids, "gt_n_aps": configured})
        by_frame = split_by_topology(frame, test_fold=3)
        by_rows = split_rows_by_topology(topology_ids, configured, test_fold=3)
        for part_frame, part_rows in zip(by_frame, by_rows):
            self.assertEqual(set(part_frame["topology_id"]), set(topology_ids[part_rows]))

    def test_every_part_keeps_both_ap_counts(self):
        topology_ids, configured = self._ids()
        for part in split_rows_by_topology(topology_ids, configured, test_fold=2):
            self.assertEqual(set(configured[part].tolist()), {2, 4})


class SpearmanTests(unittest.TestCase):
    def test_a_model_that_orders_nothing_scores_zero_rather_than_being_excluded(self):
        frame = pd.DataFrame({
            "scan_id": ["a", "a", "b", "b"],
            "label_throughput_mbps": [10.0, 0.0, 10.0, 0.0],
        })
        # perfect on set "a", completely undecided on set "b"
        metrics = selection_metrics(frame, np.array([1.0, 0.0, 5.0, 5.0]))
        self.assertAlmostEqual(metrics["mean_spearman"], 0.5)

    def test_a_scan_with_nothing_to_order_is_left_out_of_the_average(self):
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


if __name__ == "__main__":
    unittest.main()
