"""Regex patterns and output parsing for fast_test.py.

Extracted from fast_test.py to keep files under 400 lines.
"""
import re

# Regex patterns for parsing scenario runner stdout
ATO_TABLE_LINE = re.compile(
    r"^\s*(\d+)\s+"          # target number
    r"([\d.]+)m\s+"          # A* distance
    r"([\d.]+)s\s+"          # time
    r"([\d.]+)\s+"           # ATO score
    r"(\d+%)\s+"             # path efficiency
    r"([\d.]+)\s+"           # v_avg
    r"([\d.]+)m\s+"          # regression
    r"([\d.]+)\s+"           # regression gate
    r"([\d.]+)s"             # stall
    r"(.*)"                  # optional TIMEOUT suffix
)
ATO_AGG_LINE = re.compile(
    r"^\s+([\d.]+)m\s+"      # total A* distance
    r"([\d.]+)s\s+"          # total time
    r"([\d.]+)\s+"           # aggregate ATO
    r"(\d+%)\s+"             # aggregate path efficiency
    r"([\d.]+)\s+"           # aggregate v_avg
    r"([\d.]+)m\s+"          # total regression
    r"([\d.]+)s"             # total stall
)
TARGET_REACHED = re.compile(r"TARGET (\d+) REACHED in ([\d.]+)s")
TARGET_TIMEOUT = re.compile(r"TARGET (\d+) TIMEOUT after ([\d.]+)s")
PER_TARGET_ATO = re.compile(r"ATO=([\d.]+)\s+agg=([\d.]+)")
TARGETS_LINE = re.compile(r"Targets:\s*(\d+)/(\d+)\s+reached")


def parse_output(stdout: str) -> dict:
    """Parse scenario runner stdout for ATO scores and target results."""
    result = {
        "targets": [],       # per-target info
        "aggregate_ato": None,
        "reached": 0,
        "total": 0,
        "falls": 0,
        "error": None,
        # ATO component breakdown (from aggregate line)
        "path_efficiency": None,
        "v_avg": None,
        "regression": None,
        "stall": None,
    }

    lines = stdout.split("\n")

    # Parse per-target reached/timeout events with their ATO
    for i, line in enumerate(lines):
        m = TARGET_REACHED.search(line)
        if m:
            idx, t = int(m.group(1)), float(m.group(2))
            ato = 0.0
            # Next line should have ATO details
            if i + 1 < len(lines):
                am = PER_TARGET_ATO.search(lines[i + 1])
                if am:
                    ato = float(am.group(1))
            result["targets"].append({"idx": idx, "time": t, "ato": ato, "timeout": False})
            continue
        m = TARGET_TIMEOUT.search(line)
        if m:
            idx, t = int(m.group(1)), float(m.group(2))
            result["targets"].append({"idx": idx, "time": t, "ato": 0.0, "timeout": True})

    # Parse the ATO FITNESS aggregate line (last numbers line in the table)
    in_ato_table = False
    for line in lines:
        if "=== ATO FITNESS" in line:
            in_ato_table = True
            continue
        if in_ato_table:
            # Try aggregate line (no target number, just starts with distance)
            m = ATO_AGG_LINE.match(line)
            if m:
                result["aggregate_ato"] = float(m.group(3))
                # Extract components
                pe_str = m.group(4)  # e.g., "95%"
                result["path_efficiency"] = int(pe_str.rstrip("%")) / 100.0
                result["v_avg"] = float(m.group(5))
                result["regression"] = float(m.group(6))
                result["stall"] = float(m.group(7))

    # Parse targets line ("Targets: 3/4 reached (75%)")
    for line in lines:
        m = TARGETS_LINE.search(line)
        if m:
            result["reached"] = int(m.group(1))
            result["total"] = int(m.group(2))

    # Check for falls
    for line in lines:
        if "FALL DETECTED" in line:
            result["falls"] += 1

    return result
