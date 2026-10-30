"""Scientific selection, interpolation and normalization for campaign plots."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MTS_Management.config import Config
from MTS_Plotting.campaign_plot import prepare, plot, prepared_plot, validate_axes


class CampaignPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        entries = []
        for threads in (1, 2):
            for seed in (0, 1):
                config = Config(rows=2, cols=2, nrThreads=threads, seed=seed)
                entries.append({"label": "Example", "settings": config.as_dict()})
                folder = self.root / config.folder_name()
                folder.mkdir()
                config.write(folder / "config.conf")
                (folder / "manifest.json").write_text(json.dumps({
                    "settings": config.as_dict(), "config_hash": config.config_hash,
                    "executable_sha256": "fixture", "campaign": "Example"}))
                (folder / "termination.json").write_text(json.dumps({"termination": {
                    "stopReason": "strain_limit", "targetsMet": False}}))
                with (folder / "macroData.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["load", "avg_energy", "avg_sigma12"])
                    for step in range(3):
                        load = step + seed / 2
                        writer.writerow([load, threads * (4 * load + 2 * seed), 2 * load])
        self.campaign = self.root / "simulations.json"
        self.campaign.write_text(json.dumps({"name": "Example", "runs": entries}))
        self.spec = dict(x="load", y="avg_energy", fixed={"nrThreads": 1},
                         color=None, gradient=None, seed="all", normalization="N")

    def test_seed_mean_interpolates_only_common_range_and_normalizes_before_statistics(self):
        result = prepare(self.campaign, self.root, self.spec)
        item, = result["series"]
        np.testing.assert_allclose(item["x"], [.5, 1, 1.5, 2])
        np.testing.assert_allclose(item["mean"], [.75, 1.25, 1.75, 2.25])
        np.testing.assert_allclose(item["sd"], [2**.5 / 4] * 4)
        self.assertEqual(item["seed_ids"], [0, 1])
        self.assertTrue(any("interpolation" in warning for warning in result["warnings"]))
        single = prepare(self.campaign, self.root, self.spec | {"seed": 0})
        self.assertEqual(single["series"][0]["x"], [0, 1, 2])
        self.assertEqual(single["series"][0]["mean"], [0, 1, 2])
        self.assertIsNone(single["series"][0]["sd"])
        self.assertFalse(any("interpolation" in warning for warning in single["warnings"]))

    def test_scalar_reduction_and_unused_campaign_parameter_guard(self):
        spec = self.spec | {"x": "nrThreads", "fixed": {}, "reduction": "last"}
        result = prepare(self.campaign, self.root, spec)
        self.assertEqual([(s["x"], s["mean"]) for s in result["series"]], [(1, 2.5), (2, 5)])
        with self.assertRaisesRegex(ValueError, "hold it fixed"):
            prepare(self.campaign, self.root, self.spec | {"fixed": {}})
        with self.assertRaisesRegex(ValueError, "requires a campaign parameter"):
            plot(self.campaign, self.root, self.root / "invalid.pdf", self.spec | {"y": "total_runtime_s"})
        self.assertFalse((self.root / "invalid.pdf").exists())

    def test_missing_seed_is_reported_and_distinct_executables_are_rejected(self):
        folders = sorted(self.root.glob("*/manifest.json"))
        folder = next(p.parent for p in folders if json.loads(p.read_text())["settings"]["seed"] == 1
                      and json.loads(p.read_text())["settings"]["nrThreads"] == 1)
        (folder / "manifest.json").rename(folder / "manifest.saved")
        result = prepare(self.campaign, self.root, self.spec)
        self.assertEqual(result["runs"], 1)
        self.assertTrue(any("absent" in warning for warning in result["warnings"]))
        (folder / "manifest.saved").rename(folder / "manifest.json")
        path = folder / "manifest.json"
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps(manifest | {"executable_sha256": "different"}))
        with self.assertRaisesRegex(ValueError, "different executables"):
            prepare(self.campaign, self.root, self.spec)

    def test_preview_cache_tracks_source_changes_and_log_axes_reject_nonpositive_data(self):
        spec = self.spec | {"seed": 0}
        _, result = prepared_plot(self.campaign, self.root, spec)
        self.assertEqual(result["series"][0]["mean"], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_axes(result, spec | {"yscale": "log"})
        folder = self.root / Config(rows=2, cols=2, nrThreads=1, seed=0).folder_name()
        path = folder / "macroData.csv"
        path.write_text(path.read_text().replace("2.0,8.0,4.0", "2.0,16.0,4.0"))
        _, updated = prepared_plot(self.campaign, self.root, spec | {"legend_style": "full"})
        self.assertEqual(updated["series"][0]["mean"], [0, 1, 4])
        self.assertEqual(result["series"][0]["mean"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
