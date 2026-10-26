import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .benchmark import DIMENSIONS, METRICS, figure_name, plot_benchmark, summarize


def main():
    parser = argparse.ArgumentParser(description="Summarize current benchmark snapshots and save compact PDFs")
    parser.add_argument("data", type=Path, help="A managed run or root containing downloaded run folders")
    parser.add_argument("output", type=Path)
    parser.add_argument("--campaign", type=Path, help="Manager campaign JSON, for missing-run accounting")
    parser.add_argument("--metric", choices=METRICS, default="elastic_minimization_s")
    parser.add_argument("--x", choices=DIMENSIONS, default="threads")
    parser.add_argument("--color", choices=DIMENSIONS, default="preconditioner")
    parser.add_argument("--gradient", choices=DIMENSIONS)
    parser.add_argument("--fixed", default="{}", help='JSON selection, e.g. {"size":400,"reconnection":"none"}')
    parser.add_argument("--legend-loc", default="best")
    parser.add_argument("--legend-ncols", type=int, default=1)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--save-summary", type=Path, help="Optionally save the printed JSON report outside figures")
    args = parser.parse_args()
    rows, audit = summarize(args.data, campaign=args.campaign)
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    if not args.summary_only:
        slots = dict(y=args.metric, x=args.x, color=args.color,
                     gradient=args.gradient, fixed=json.loads(args.fixed))
        path = args.output / figure_name(**slots)
        plot_benchmark(rows, path, **slots, legend=dict(loc=args.legend_loc, ncols=args.legend_ncols),
                       xlim=args.xlim, ylim=args.ylim)
        files.append(path)
    report = audit | dict(created_at=datetime.now(timezone.utc).isoformat(),
                          data=str(args.data.resolve()),
                          campaign=str(args.campaign.resolve()) if args.campaign else None,
                          metric=args.metric, x=args.x, color=args.color, gradient=args.gradient,
                          fixed=json.loads(args.fixed), figures=[str(p) for p in files])
    if args.save_summary:
        args.save_summary.parent.mkdir(parents=True, exist_ok=True)
        args.save_summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
