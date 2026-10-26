"""Explicitly fetch completed benchmark macro snapshots using MTSManager.

Run from any directory, passing the manager's sites.toml and campaign JSON.
No production jobs are submitted. Each download reuses the manager's identity checks.
"""

import argparse
from pathlib import Path

from MTS_Management.campaign import campaign_entries
from MTS_Management.cluster import fetch_many, load_sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sites", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--site", default="magi")
    parser.add_argument("--size", type=int, help="Limit downloads to this square size")
    args = parser.parse_args()
    _, entries = campaign_entries(args.campaign)
    hashes = {c.config_hash for c, _ in entries if not c["logDuringMinimization"]
              and not c["fullMinimizationLogging"]
              and (args.size is None or c["rows"] == c["cols"] == args.size)}
    site = load_sites(args.sites)[args.site]
    paths = fetch_many(site, hashes, args.destination, completed_only=True)
    print(f"Fetched {len(paths)} completed runs to {args.destination}", flush=True)



if __name__ == "__main__":
    main()
