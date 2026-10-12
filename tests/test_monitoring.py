"""Daily alerting (scripts/daily_health_alert.py) and the weekly backup of
irreplaceable tables (scripts/backup_precious_tables.py)."""

import gzip
import importlib.util
import os
import sqlite3
import time
from datetime import date, timedelta

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


alert = load_script("daily_health_alert")
backup = load_script("backup_precious_tables")


# ---------------------------------------------------------------------------
# Health-check warnings: alert when new or when the shortfall changes
# ---------------------------------------------------------------------------


def coverage_warning(expected, actual):
    return {
        "stage": "Predictions",
        "check_name": "predictions_Phase3",
        "status": "warn",
        "message": "m",
        "expected": expected,
        "actual": actual,
    }


KEY = "Predictions/predictions_Phase3"


@pytest.mark.parametrize(
    "prev, result, alerts",
    [
        ({}, coverage_warning(200, 150), ["NEW WARNING"]),
        # Known gap, more games played: shortfall unchanged, stay quiet
        ({KEY: {"status": "warn", "value": 45.0}}, coverage_warning(300, 255), []),
        # Gap grew: a predictor is failing
        (
            {KEY: {"status": "warn", "value": 45.0}},
            coverage_warning(300, 250),
            ["CHANGED"],
        ),
        # State saved before shortfalls were tracked: no false alarm
        ({KEY: {"status": "warn", "actual": 100}}, coverage_warning(300, 250), []),
    ],
)
def test_warnings_alert_on_new_or_changed_shortfall(prev, result, alerts):
    lines, _ = alert.find_new_issues({"results": [result]}, {"issues": prev})
    assert [line.split("]")[0].lstrip("[") for line in lines] == alerts


# ---------------------------------------------------------------------------
# Pipeline runs and backups
# ---------------------------------------------------------------------------


@pytest.fixture
def runs_db(tmp_path, monkeypatch):
    path = tmp_path / "runs.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE PipelineRuns (run_id TEXT, run_type TEXT,
            started_at TEXT, status TEXT, errors TEXT)""")
    monkeypatch.setattr(alert, "DB_PATH", str(path))
    return path


def add_run(path, started_at, status, errors="[]"):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO PipelineRuns VALUES (?, 'post_game', ?, ?, ?)",
            (started_at, started_at, status, errors),
        )


def test_failed_and_stuck_pipeline_runs_alert_once(runs_db):
    add_run(runs_db, "2026-10-08 14:00:00", "success")
    add_run(runs_db, "2026-10-09 14:00:00", "partial", '["database_update: boom"]')
    add_run(runs_db, "2026-10-10 03:00:00", "running")  # long dead

    lines, last = alert.pipeline_run_alerts("2026-10-08 00:00:00")
    assert len(lines) == 2
    assert "database_update: boom" in lines[0]
    assert "never finished" in lines[1]
    # Already reported runs aren't reported again
    assert alert.pipeline_run_alerts(last) == ([], last)


def test_run_still_in_progress_is_checked_again_later(runs_db):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    add_run(runs_db, now, "running")
    assert alert.pipeline_run_alerts("2026-01-01 00:00:00") == (
        [],
        "2026-01-01 00:00:00",
    )


def test_stale_or_missing_backups_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(alert, "BACKUP_DIR", tmp_path)
    assert "no precious-table backups" in alert.backup_alerts()[0]

    latest = tmp_path / "precious_20261001.sqlite.gz"
    latest.write_bytes(b"")
    assert alert.backup_alerts() == []
    nine_days_ago = time.time() - 9 * 86400
    os.utime(latest, (nine_days_ago, nine_days_ago))
    assert "days old" in alert.backup_alerts()[0]


@pytest.fixture
def source_db(tmp_path):
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as conn:
        for table in backup.PRECIOUS_TABLES:
            conn.execute(f"CREATE TABLE {table} (x INTEGER)")
            conn.execute(f"INSERT INTO {table} VALUES (1), (2)")
    return str(path)


def test_backup_survives_a_crashed_run(source_db, tmp_path):
    outdir = tmp_path / "daily"
    outdir.mkdir()
    stamp = time.strftime("%Y%m%d")
    # What a run that died midway leaves behind
    with sqlite3.connect(outdir / f"precious_{stamp}.sqlite") as conn:
        conn.execute("CREATE TABLE Predictions (x INTEGER)")

    gz = backup.backup(source_db, outdir)

    restored = tmp_path / "restored.sqlite"
    restored.write_bytes(gzip.decompress(gz.read_bytes()))
    with sqlite3.connect(restored) as conn:
        for table in backup.PRECIOUS_TABLES:
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 2


def test_rotation_keeps_recent_and_month_start_backups(tmp_path):
    # Weekly (Sunday) backups, Aug 2 .. Oct 11, after older daily ones
    days = ["20260731"] + [
        (date(2026, 8, 2) + timedelta(weeks=w)).strftime("%Y%m%d") for w in range(11)
    ]
    for day in days:
        (tmp_path / f"precious_{day}.sqlite.gz").write_bytes(b"")

    backup.rotate(tmp_path)

    kept = sorted(p.name[9:17] for p in tmp_path.glob("precious_*.sqlite.gz"))
    assert "20260731" in kept and "20260802" in kept  # each month's first kept
    assert "20260809" not in kept and "20260816" not in kept  # rotated out
    assert len(kept) == backup.KEEP + 2
