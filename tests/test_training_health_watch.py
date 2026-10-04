"""Unit tests for train/training_health_watch.py (no real YAML / TB required)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from train.training_health_watch import (
    DEFAULT_LLM_MODEL,
    build_digest,
    build_llm_prompt,
    detect_active_log_level,
    evaluate_findings,
    extract_alert_lines,
    render_digest_markdown,
    reset_watch_session,
    write_digest,
)
from train.training_process_logging import _is_alert_record
from datetime import datetime


def test_detect_active_log_level_picks_newest(tmp_path: Path) -> None:
    older = tmp_path / "level_a"
    newer = tmp_path / "level_b"
    older.mkdir()
    newer.mkdir()
    (older / "progress.csv").write_text("x\n1\n", encoding="utf-8")
    time.sleep(0.05)
    (newer / "progress.csv").write_text("x\n2\n", encoding="utf-8")
    assert detect_active_log_level(str(tmp_path)) == "level_b"


def test_evaluate_findings_log_prob_crit() -> None:
    findings = evaluate_findings(
        training_alive=True,
        active_log_dir=None,
        tb_latest={"train/rollout_log_prob_abs_err": (10, 1.5)},
        alert_lines=[],
        stale_minutes=45,
    )
    by_code = {f.code: f for f in findings}
    assert by_code["log_prob_mismatch"].severity == "CRIT"


def test_evaluate_findings_graft_warn() -> None:
    findings = evaluate_findings(
        training_alive=True,
        active_log_dir=None,
        tb_latest={},
        alert_lines=[
            "process_alerts.log: 2026-01-01 | WARNING  | Grafting policy weights "
            "(3 shape mismatches, e.g. query_linear.weight)"
        ],
        stale_minutes=45,
    )
    by_code = {f.code: f for f in findings}
    assert by_code["alert_logs"].severity == "WARN"


def test_evaluate_findings_policy_load_and_kl() -> None:
    findings = evaluate_findings(
        training_alive=True,
        active_log_dir=None,
        tb_latest={
            "train/rollout_policy_load_ok": (3, 0.0),
            "train/approx_kl": (3, 2.0),
        },
        alert_lines=["worker_1.log: Traceback (most recent call last):"],
        stale_minutes=45,
    )
    severities = {f.code: f.severity for f in findings}
    assert severities["policy_load"] == "CRIT"
    assert severities["approx_kl"] == "CRIT"
    assert severities["alert_logs"] == "CRIT"


def test_extract_alert_lines_filters_noise(tmp_path: Path) -> None:
    alerts = tmp_path / "process_alerts.log"
    alerts.write_text(
        "\n".join(
            [
                "2026-01-01 | INFO     | boring heartbeat",
                "2026-01-01 | WARNING  | Grafting policy weights (2 shape mismatches)",
                "2026-01-01 | ERROR    | Collector subprocess CRASHED",
                "2026-01-01 | INFO     | still boring",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    worker = tmp_path / "worker_1.log"
    worker.write_text("ok\nTraceback (most recent call last):\n  File x\n", encoding="utf-8")
    hits = extract_alert_lines([str(alerts), str(worker)], max_scan_lines=50, max_hits=20)
    joined = "\n".join(hits)
    assert "Grafting" in joined
    assert "CRASHED" in joined
    assert "Traceback" in joined
    assert "boring heartbeat" not in joined


def test_benign_replay_and_tensorboard_lines_are_not_critical(tmp_path: Path) -> None:
    benign = [
        "process_alerts.log: 2026-10-03 20:22:13.043 | ERROR    | "
        "TensorBoard process started but port 6006 is not listening.",
        "process_alerts.log: 2026-10-03 21:46:03.873 | ERROR    | "
        "PPO train: skipping weight update — flat replay abs err 0.217 > 0.05 "
        "(saved decisions do not match this policy)",
        "process_alerts.log: 2026-10-03 21:46:04.173 | ERROR    | "
        "Learner: replay mismatch — weights were not stepped and will not be republished",
        "process_alerts.log: 2026-10-03 21:46:04.177 | WARNING  | "
        "Learner: Update skipped after replay mismatch; collector keeps the previous checkpoint.",
        "process_alerts.log: 2026-10-03 21:46:04.180 | WARNING  | "
        "Learner: skipped periodic checkpoint save (replay mismatch, weights unchanged)",
    ]
    findings = evaluate_findings(
        training_alive=True,
        active_log_dir=None,
        tb_latest={"train/rollout_log_prob_abs_err": (10, 1.6e-4)},
        alert_lines=benign,
        stale_minutes=45,
    )
    by_code = {f.code: f for f in findings}
    assert by_code["alert_logs"].severity == "OK"
    assert by_code["log_prob_mismatch"].severity == "OK"

    alerts = tmp_path / "process_alerts.log"
    alerts.write_text(
        "\n".join(line.split(": ", 1)[1] for line in benign)
        + "\n2026-10-03 21:50:00.000 | ERROR    | Collector subprocess CRASHED\n",
        encoding="utf-8",
    )
    hits = extract_alert_lines([str(alerts)], max_scan_lines=50, max_hits=20)
    joined = "\n".join(hits)
    assert "TensorBoard process started" not in joined
    assert "flat replay abs err" not in joined
    assert "Update skipped after replay mismatch" not in joined
    assert "CRASHED" in joined


def test_alert_record_filter_keeps_graft_info() -> None:
    assert _is_alert_record({"level": type("L", (), {"no": 20})(), "message": "Grafting head weights"})
    assert _is_alert_record({"level": type("L", (), {"no": 30})(), "message": "something"})
    assert not _is_alert_record({"level": type("L", (), {"no": 20})(), "message": "Collector started"})


def test_build_digest_and_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    logs_u = repo / "logs_universal" / "ctf_level_01"
    workers = repo / "logs"
    logs_u.mkdir(parents=True)
    workers.mkdir()
    (logs_u / "progress.csv").write_text(
        "time/total_timesteps,train/approx_kl\n1000,0.01\n2000,0.02\n",
        encoding="utf-8",
    )
    (workers / "worker_1.log").write_text("all good\n", encoding="utf-8")
    (workers / "process_alerts.log").write_text(
        "2026-01-01 | WARNING  | Grafting policy weights\n",
        encoding="utf-8",
    )
    (workers / "vram_monitor.log").write_text("[VRAM] ok\n", encoding="utf-8")

    digest = build_digest(
        str(repo),
        log_dir_base="logs_universal",
        worker_logs_dir="logs",
        stale_minutes=9999,
        alert_since=None,  # no session filter
    )
    assert digest.active_level == "ctf_level_01"
    assert any("Grafting" in line for line in digest.alert_lines)
    assert digest.severity in ("WARN", "CRIT")
    md = render_digest_markdown(digest)
    assert "Warn / Error / Graft excerpt" in md

    health = repo / "logs_universal" / "_health"
    health.mkdir(parents=True, exist_ok=True)
    path = write_digest(digest, str(health))
    assert os.path.isfile(path)
    assert (health / "alerts_excerpt.md").is_file()
    # No timestamped digest spam.
    stamped = list(health.glob("digest_20*.md"))
    assert stamped == []


def test_reset_watch_session_clears_reports_and_alerts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    health = repo / "logs_universal" / "_health"
    logs = repo / "logs"
    health.mkdir(parents=True)
    logs.mkdir(parents=True)
    (health / "digest_20260101_120000.md").write_text("old\n", encoding="utf-8")
    (health / "digest_latest.md").write_text("old\n", encoding="utf-8")
    (health / "latest.md").write_text("old\n", encoding="utf-8")
    (health / "watch.lock").write_text("1\n", encoding="utf-8")
    (logs / "process_alerts.log").write_text(
        "2026-01-01 12:00:00 | ERROR    | old crash\n",
        encoding="utf-8",
    )
    (logs / "process_alerts.log.1").write_text("rotated\n", encoding="utf-8")

    stats = reset_watch_session(
        str(repo), log_dir_base="logs_universal", worker_logs_dir="logs"
    )
    assert stats["removed_reports"] >= 3
    assert not (health / "digest_20260101_120000.md").exists()
    assert (health / "watch.lock").exists()  # lock preserved
    assert (health / "session_start.txt").exists()
    assert (logs / "process_alerts.log").read_text(encoding="utf-8") == ""
    assert not (logs / "process_alerts.log.1").exists()


def test_extract_alert_lines_respects_session_since(tmp_path: Path) -> None:
    alerts = tmp_path / "process_alerts.log"
    alerts.write_text(
        "\n".join(
            [
                "2026-01-01 10:00:00 | ERROR    | old fixed bug",
                "2026-01-01 12:00:00 | WARNING  | Grafting after restart",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    since = datetime(2026, 1, 1, 11, 0, 0)
    hits = extract_alert_lines([str(alerts)], since=since, max_hits=20)
    joined = "\n".join(hits)
    assert "old fixed bug" not in joined
    assert "Grafting after restart" in joined


def test_llm_prompt_points_at_excerpt_only() -> None:
    prompt = build_llm_prompt("d.md", "a.md", "r.md")
    assert "a.md" in prompt
    assert "do not open full worker" in prompt.lower() or "Do not open full" in prompt
    assert DEFAULT_LLM_MODEL  # smoke: constant exists
