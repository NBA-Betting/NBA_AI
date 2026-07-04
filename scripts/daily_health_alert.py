#!/usr/bin/env python3
"""
Run the health check and alert only on NEW or WORSENING issues.

The May 2026 finalization bug went unnoticed for a month because the pipeline
logged the same warning every day into a log nobody reads, while reporting
overall success. This wrapper makes regressions loud without daily noise:

- Runs `src.health_check --skip-pipeline --json` for the given season.
- Compares against the previous run's state (data/collection_state/health_state.json).
- Criticals always alert. Warnings alert only when the check is newly
  failing or its `actual` value got worse since the last run.
- Alerts append to logs/ALERTS.log AND print to stdout (so they land in the
  cron log too). A desktop notification is attempted best-effort.

Exit codes: 0 = no new issues, 1 = new/worsening issues, 2 = health check
itself failed to run (also alerts — silence must never look like success).

Usage:
    python scripts/daily_health_alert.py [--season Current]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "data" / "collection_state" / "health_state.json"
ALERTS_LOG = PROJECT_ROOT / "logs" / "ALERTS.log"
PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"


def run_health_check(season: str) -> dict | None:
    cmd = [
        str(PYTHON),
        "-m",
        "src.health_check",
        f"--season={season}",
        "--skip-pipeline",
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=1800
        )
    except subprocess.TimeoutExpired:
        return None
    # stdout has a human preamble before the JSON object
    out = proc.stdout
    brace = out.find("{")
    if brace == -1:
        return None
    try:
        return json.loads(out[brace:])
    except json.JSONDecodeError:
        return None


def issue_key(result: dict) -> str:
    return f"{result['stage']}/{result['check_name']}"


CRITICAL_REMINDER_DAYS = 7


def find_new_issues(report: dict, prev_state: dict) -> tuple[list[str], dict]:
    """
    Return (alert lines, alert timestamps to carry into saved state).

    Warnings alert when newly failing or when their value changes. Criticals
    alert when new/changed, then remind weekly while unresolved — daily
    repeats of a known issue train the reader to ignore alerts, but a stuck
    critical must not go permanently silent either.
    """
    prev_issues = prev_state.get("issues", {})
    now = datetime.now()
    alerts = []
    alerted_at = {}
    for r in report.get("results", []):
        status = r.get("status")
        if status not in ("warn", "critical"):
            continue
        key = issue_key(r)
        prev = prev_issues.get(key)
        changed = prev is not None and _worsened(prev.get("actual"), r.get("actual"))
        prev_alert = _parse_dt(prev.get("alerted_at")) if prev else None
        stale = prev_alert is None or (now - prev_alert).days >= CRITICAL_REMINDER_DAYS
        if status == "critical":
            if prev is None or changed:
                tag = "CRITICAL"
            elif stale:
                tag = "CRITICAL (still unresolved)"
            else:
                alerted_at[key] = prev.get("alerted_at")
                continue
        elif prev is None:
            tag = "NEW WARNING"
        elif changed:
            tag = "WORSENING"
        else:
            continue  # known, stable warning — stay quiet
        alerts.append(f"[{tag}] {key}: {r.get('message')}")
        alerted_at[key] = now.isoformat()
    return alerts, alerted_at


def _parse_dt(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _worsened(prev_actual, curr_actual) -> bool:
    try:
        return float(curr_actual) != float(prev_actual)
    except (TypeError, ValueError):
        return str(curr_actual) != str(prev_actual)


def save_state(report: dict, alerted_at: dict):
    issues = {
        issue_key(r): {
            "status": r["status"],
            "actual": r.get("actual"),
            "alerted_at": alerted_at.get(issue_key(r)),
        }
        for r in report.get("results", [])
        if r.get("status") in ("warn", "critical")
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"updated": datetime.now().isoformat(), "issues": issues}, indent=2)
    )


def emit_alerts(lines: list[str]):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_LOG, "a") as f:
        for line in lines:
            f.write(f"{stamp} {line}\n")
    for line in lines:
        print(f"ALERT: {line}")
    # Best-effort desktop notification; cron usually has no display
    try:
        subprocess.run(
            ["notify-send", "NBA_AI health alert", "\n".join(lines[:5])],
            timeout=5,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="Current")
    args = parser.parse_args()

    report = run_health_check(args.season)
    if report is None:
        emit_alerts(["[CRITICAL] health_check failed to run or produced no JSON"])
        sys.exit(2)

    prev_state = {}
    if STATE_PATH.exists():
        try:
            prev_state = json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            pass

    alerts, alerted_at = find_new_issues(report, prev_state)
    save_state(report, alerted_at)

    summary = report.get("summary", {})
    print(
        f"Health check: {summary.get('passed', '?')} passed, "
        f"{summary.get('warnings', '?')} warnings, "
        f"{summary.get('critical', '?')} critical — "
        f"{len(alerts)} new/worsening"
    )
    if alerts:
        emit_alerts(alerts)
        sys.exit(1)


if __name__ == "__main__":
    main()
