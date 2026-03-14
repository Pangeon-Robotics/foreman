"""Regex patterns and output parsing for fast_test.py.

Parses the scenario report table output from run_scenario.py.
"""
import re

# Summary table line: "scattered   1/2   50%   1   16.4   0.36m   n/a   n/a   3/3   PASS"
# Fields: scenario, targets (N/M), success%, falls, ATO, SLAM drift, 3DS, 3DS-G, critics, result
REPORT_LINE = re.compile(
    r"^\s*(\w+)"            # scenario name
    r"\s+(\d+)/(\d+)"      # targets reached/total
    r"\s+\d+%"             # success rate (ignore)
    r"\s+(\d+)"            # falls
    r"\s+([\d.]+|n/a)"     # ATO score
    r"\s+([\d.]+m|n/a)"    # SLAM drift
)

FALL_LINE = re.compile(r"FALL|fell", re.IGNORECASE)

# Telemetry line: fit=X  stride=Xm  ... v=X.Xm/s
TELEMETRY_LINE = re.compile(
    r"T(\d+)/(\d+)\s+fit=([\d.]+).*v=([\d.]+)m/s"
)


def parse_output(stdout: str) -> dict:
    """Parse scenario runner stdout for ATO scores and target results."""
    result = {
        "targets": [],       # per-target info
        "aggregate_ato": None,
        "reached": 0,
        "total": 0,
        "falls": 0,
        "error": None,
        "path_efficiency": None,
        "v_avg": None,
        "slip_efficiency": None,
        "regression": None,
        "stall": None,
    }

    lines = stdout.split("\n")

    # Parse the scenario report table line
    in_report = False
    for line in lines:
        if "SCENARIO TEST REPORT" in line:
            in_report = True
            continue
        if in_report:
            m = REPORT_LINE.match(line)
            if m:
                result["reached"] = int(m.group(2))
                result["total"] = int(m.group(3))
                result["falls"] = int(m.group(4))
                ato_str = m.group(5)
                if ato_str != "n/a":
                    result["aggregate_ato"] = float(ato_str)

    # Extract per-target data from telemetry lines
    # Track which targets we've seen by looking at target transitions
    target_times = {}  # target_idx -> last seen time
    target_vavgs = {}  # target_idx -> list of velocities
    for line in lines:
        m = TELEMETRY_LINE.search(line)
        if m:
            tidx = int(m.group(1))
            total = int(m.group(2))
            v = float(m.group(4))
            if tidx not in target_vavgs:
                target_vavgs[tidx] = []
            target_vavgs[tidx].append(v)

    # Build per-target entries
    for tidx in sorted(target_vavgs.keys()):
        vavgs = target_vavgs[tidx]
        avg_v = sum(vavgs) / len(vavgs) if vavgs else 0.0
        reached = tidx < result["reached"] + 1 if result["total"] > 0 else False
        timeout = tidx >= result["reached"] + 1 if result["total"] > 0 else True
        result["targets"].append({
            "idx": tidx,
            "ato": result["aggregate_ato"] or 0.0 if not timeout else 0.0,
            "timeout": timeout,
        })

    # Compute mean v_avg from all telemetry
    all_vavgs = []
    for vlist in target_vavgs.values():
        all_vavgs.extend(vlist)
    if all_vavgs:
        result["v_avg"] = sum(all_vavgs) / len(all_vavgs)

    # Count falls from output
    fall_count = 0
    for line in lines:
        if "FALL DETECTED" in line or "fell" in line.lower():
            fall_count += 1
    if fall_count > 0 and result["falls"] == 0:
        result["falls"] = fall_count

    return result
