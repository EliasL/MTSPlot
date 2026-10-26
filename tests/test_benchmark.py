import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from MTS_Management.config import Config
from MTS_Plotting.benchmark import plot_benchmark, plot_suite, seconds, summarize, timing_summary
from MTS_Plotting.runs import Run, discover


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

    def test_identity_errors(self):
        run = self.run_fixture()
        manifest = run.manifest | {"config_hash": "wrong"}
        (run.folder / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            discover(self.root)

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

    def test_pdf_and_sample_counts(self):
        self.run_fixture(rows=4, cols=4)
        self.run_fixture(rows=4, cols=4, seed=1)
        rows, audit = summarize(self.root)
        path = self.root / "elastic.pdf"
        points = plot_benchmark(rows, path, y="elastic_minimization_s",
                                x="size", color="preconditioner")
        self.assertEqual(audit["eligible"], 2)
        self.assertTrue(path.read_bytes().startswith(b"%PDF"))
        self.assertEqual(points[0]["seeds"], 2)
        self.assertEqual(points[0]["mean"], 3)
        self.assertEqual(points[0]["sd"], 0)

    def test_slots_require_explicit_unplotted_dimensions(self):
        for size, threads in ((4, 1), (8, 1), (4, 2)):
            self.run_fixture(rows=size, cols=size, nrThreads=threads)
        rows, _ = summarize(self.root)
        options = dict(y="elastic_minimization_s", x="size", color="preconditioner")
        path = self.root / "size.pdf"
        with self.assertRaisesRegex(ValueError, "Assign threads"):
            plot_benchmark(rows, path, **options)
        points = plot_benchmark(rows, path, gradient="threads", **options)
        self.assertEqual({(p["x"], p["gradient"], p["seeds"]) for p in points},
                         {(4, 1, 1), (8, 1, 1), (4, 2, 1)})
        with self.assertRaisesRegex(ValueError, "distinct"):
            plot_benchmark(rows, path, gradient="size", **options)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            plot_benchmark(rows, path, fixed={"threads": 99}, **options)
        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            plot_benchmark(rows + rows, path, gradient="threads", **options)
        rows[0]["cols"] += 1
        with self.assertRaisesRegex(ValueError, "square"):
            plot_benchmark(rows, path, gradient="threads", **options)

    def test_suite_has_32_distinct_figures_and_interchangeable_slots(self):
        from itertools import product
        for size, threads, preconditioner, reconnection in product(
                (4, 8), (1, 2), ("none", "exactTangent"), ("none", "edgeFlip")):
            self.run_fixture(rows=size, cols=size, nrThreads=threads,
                             preconditioner=preconditioner, reconnectionMethod=reconnection)
        rows, audit = summarize(self.root)
        with patch("MTS_Plotting.benchmark.plot_benchmark", return_value=[]) as render:
            report = plot_suite(rows, self.root / "figures",
                metrics=("elastic_minimization_s", "plastic_minimization_s"),
                comparisons=(dict(color="preconditioner", split="reconnection"),
                             dict(color="reconnection", split="preconditioner")),
                views=(dict(x="size", fixed={"threads": 1}),
                       dict(x="size", gradient="threads"),
                       dict(x="threads", fixed={"size": 4}),
                       dict(x="threads", gradient="size")), audit=audit)
        self.assertEqual(render.call_count, 32)
        self.assertEqual(len({f["file"] for f in report["figures"]}), 32)
        self.assertIn("elastic_threads_reconnection_exactTangent_L4.pdf",
                      {f["file"] for f in report["figures"]})
        self.assertEqual(list((self.root/"figures").glob("*.csv")), [])
        self.assertFalse((self.root/"figures/summary.json").exists())
        # The same renderer also accepts categorical X and a numeric color property.
        points = plot_benchmark(rows, self.root / "swapped.pdf", y="plastic_minimization_s",
            x="preconditioner", color="threads", gradient="size", fixed={"reconnection": "none"})
        self.assertEqual(len(points), 8)
        self.assertTrue(all(p["sd"] is None for p in points))

    def test_duration_errors(self):
        self.assertEqual(seconds("1d 2h 3m 4.5s"), 93784.5)
        for text in ("", "1", "-1s", "nan", "1s 1s", "1s 1m"):
            with self.assertRaises(ValueError):
                seconds(text)



if __name__ == "__main__":
    unittest.main()
