#!/usr/bin/env python3
"""Run a command; write wall clock, peak RSS of its process tree and exit status. Standard library only."""
import resource
import subprocess
import sys
import time

out, cmd = sys.argv[1], sys.argv[2:]
t0 = time.time()
rc = subprocess.call(cmd)
sec = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
open(out, "w").write(f"wall_s: {sec:.1f}\npeak_rss_kb: {rss}\nexit: {rc}\ncmd: {' '.join(cmd)}\n")
