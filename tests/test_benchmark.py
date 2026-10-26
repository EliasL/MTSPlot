import csv
import json
from pathlib import Path
import tempfile
import unittest

from MTS_Management.config import Config
from MTS_Plotting.benchmark import plot_benchmark, summarize, timing_summary
from MTS_Plotting.runs import Run


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fixture(self, *, finished=True, target=True, **settings):
        config = Config(stopAfterPlasticSteps=1, **settings)
        folder = self.root / config.folder_name()
        folder.mkdir()
        config.write(folder / "config.conf")
        (folder / "manifest.json").write_text(json.dumps(dict(
            settings=config.as_dict(), config_hash=config.config_hash,
            executable_sha256="fixture", campaign="Test")))
        rows = [dict(load=".15", completed_steps=0, plastic_steps=0, elastic_steps=0,
                     step_class="initialization", minimization_time="2s",
                     tangent_preparation_time_s=0, run_time="3s"),
                dict(load=".15001", completed_steps=1, plastic_steps=0, elastic_steps=1,
                     step_class="elastic", minimization_time="5s",
                     tangent_preparation_time_s=1, run_time="6s"),
                dict(load=".15002", completed_steps=2, plastic_steps=1, elastic_steps=1,
                     step_class="plastic", minimization_time="12s",
                     tangent_preparation_time_s=1, run_time="14s")]
        with (folder / "macroData.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        if finished:
            (folder / "termination.json").write_text(json.dumps(dict(termination=dict(
                steps=2, plasticSteps=1, targetsMet=target,
                stopReason="sample_targets" if target else "strain_limit"))))
        return Run.read(folder)

    def test_differences_initialization_and_tangent(self):
        row = timing_summary(self.run_fixture())
        self.assertEqual([row[k] for k in ("first_minimization_s", "elastic_minimization_s",
                         "plastic_minimization_s", "tangent_preparation_s")], [2, 3, 7, 1])
        self.assertTrue(row["eligible"])

    def test_exclude_partial_cap_and_diagnostic(self):
        self.assertFalse(timing_summary(self.run_fixture(finished=False))["eligible"])
        self.assertFalse(timing_summary(self.run_fixture(target=False, seed=1))["eligible"])
        self.assertFalse(timing_summary(self.run_fixture(logDuringMinimization=True))["eligible"])

    def test_malformed_row_and_counter_errors(self):
        run = self.run_fixture()
        path = run.folder / "macroData.csv"
        original = path.read_text()
        path.write_text(original + ".15,3")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            timing_summary(run)
        path.write_text(original.replace(",5s,", ",1s,"))
        with self.assertRaisesRegex(ValueError, "cumulative"):
            timing_summary(run)

    def test_campaign_missing_and_foreign_data(self):
        run = self.run_fixture()
        config2 = Config(**(run.config.as_dict() | {"seed": 1}))
        campaign = self.root / "campaign.json"
        campaign.write_text(json.dumps(dict(name="Test", runs=[
            dict(label="Run0", settings=run.config.as_dict()),
            dict(label="Run1", settings=config2.as_dict())])))
        _, audit = summarize(self.root, campaign=campaign)
        self.assertEqual(audit["missing"], 1)
        self.run_fixture(seed=2)
        with self.assertRaisesRegex(ValueError, "outside"):
            summarize(self.root, campaign=campaign)

    def test_reject_mixed_solver_settings(self):
        self.run_fixture()
        self.run_fixture(seed=1, epsR=1e-8)
        with self.assertRaisesRegex(ValueError, "non-sweep"):
            summarize(self.root)

    def test_slots_require_explicit_unplotted_dimensions(self):
        for size, threads in ((4, 1), (8, 1), (4, 2)):
            self.run_fixture(rows=size, cols=size, nrThreads=threads)
        self.run_fixture(rows=4, cols=4, nrThreads=1, seed=1)
        rows, _ = summarize(self.root)
        for row in rows:
            if row["rows"] == 4 and row["threads"] == 1:
                row["elastic_minimization_s"] = 2 if row["seed"] == 0 else 10
        options = dict(y="elastic_minimization_s", x="size", color="preconditioner")
        path = self.root / "size.pdf"
        with self.assertRaisesRegex(ValueError, "Assign threads"):
            plot_benchmark(rows, path, **options)
        points = plot_benchmark(rows, path, gradient="threads", **options)
        self.assertEqual({(p["x"], p["gradient"], p["seeds"]) for p in points},
                         {(4, 1, 2), (8, 1, 1), (4, 2, 1)})
        replicated = next(p for p in points if p["x"] == 4 and p["gradient"] == 1)
        self.assertEqual(replicated["mean"], 6)
        self.assertAlmostEqual(replicated["sd"], 32**.5)
        self.assertEqual(replicated["seed_ids"], [0, 1])
        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            plot_benchmark(rows + rows, path, gradient="threads", **options)


if __name__ == "__main__":
    unittest.main()
