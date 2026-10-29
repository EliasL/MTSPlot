"""The benchmark's complete 24-figure specification. Run from any directory."""

from pathlib import Path

from MTS_Plotting.benchmark import plot_suite, summarize


def main():
    workspace = Path(__file__).resolve().parents[2]
    rows, audit = summarize(
        workspace / "MTSPlot/data/magi_benchmark_oct26",
        campaign=workspace / "MTSManager/campaigns/magi_benchmark_oct26/simulations.json",
    )
    plot_suite(
        rows,
        workspace / "MTSPlot/figures/magi_benchmark_oct26",
        metrics=("elastic_minimization_s", "plastic_minimization_s"),
        comparisons=(
            dict(color="preconditioner", split="reconnection"),
            dict(color="reconnection", split="preconditioner"),
        ),
        views=(
            dict(x="size", fixed={"threads": 16}),
            dict(x="size", gradient="threads"),
            dict(x="threads", gradient="size"),
        ),
        legend=dict(loc="best", ncols=1),
        audit=audit,
    )


if __name__ == "__main__":
    main()
