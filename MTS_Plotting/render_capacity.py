"""Estimate how many frame renderers fit on the local machine."""

import os
from pathlib import Path
import re
import resource
import subprocess
import sys


def peak_memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def available_memory():
    if sys.platform == "darwin":
        try:
            stats = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return 0
        page_size = re.search(r"page size of (\d+) bytes", stats)
        pages = re.findall(r"^Pages (?:free|inactive|speculative):\s*(\d+)\.", stats, re.MULTILINE)
        return int(page_size[1]) * sum(map(int, pages)) if page_size and pages else 0
    try:
        match = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.MULTILINE)
    except OSError:
        return 0
    return int(match[1]) * 1024 if match else 0


def worker_count(peak, available, cpus):
    # Reserve RAM for FFmpeg, the desktop and larger later frames. Keep one
    # CPU outside the render pool whenever the machine has more than one.
    memory_limit = available * 2 // (3 * peak) if peak > 0 else 0
    return max(1, min(8, cpus - 1, memory_limit))


def automatic_workers(peak):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return worker_count(peak, available_memory(), cpus)
