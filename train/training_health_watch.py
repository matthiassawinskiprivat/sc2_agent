"""
Long-running training health watcher (OS-agnostic).

Every hour (default): build a deterministic digest from worker logs,
progress.csv, and TensorBoard scalars; optionally ask the Cursor agent CLI
to analyze it and write a short status report.

Usage:
  python -u train/training_health_watch.py            # watch forever
  python -u train/training_health_watch.py --once     # single check
  python -u train/training_health_watch.py --no-llm   # digest only
"""

from __future__ import annotations

import argparse
import atexit
import csv
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

DEFAULT_INTERVAL_SEC = 3600
DEFAULT_STALE_MINUTES = 45
DEFAULT_LOG_TAIL_LINES = 200
DEFAULT_PROGRESS_TAIL_ROWS = 8
DEFAULT_ALERT_MAX_LINES = 120
# Cheap, fast model for triage — override with --model / SC2_HEALTH_MODEL.
DEFAULT_LLM_MODEL = os.environ.get("SC2_HEALTH_MODEL", "composer-2.5-fast").strip() or "composer-2.5-fast"
HEALTH_SUBDIR = "_health"
LOCK_NAME = "watch.lock"
DIGEST_NAME = "digest_latest.md"
ALERTS_EXCERPT_NAME = "alerts_excerpt.md"
REPORT_NAME = "latest.md"
SESSION_START_NAME = "session_start.txt"
PROCESS_ALERTS_NAME = "process_alerts.log"

# Prefix optional "filename: " then loguru timestamp.
ALERT_TS_RE = re.compile(
    r"^(?:[^:]+:\s*)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?"
)

KEY_TB_TAGS = (
    "rollout/ep_rew_mean",
    "train/explained_variance",
    "train/approx_kl",
    "train/clip_fraction",
    "train/rollout_log_prob_abs_err",
    "train/rollout_log_prob_abs_err_flat",
    "train/rollout_policy_load_ok",
    "train/value_loss",
    "train/entropy_loss",
    "time/fps",
    "metrics/collector/instant_fps",
)

# Loguru level tokens written by training_process_logging / stderr.
LOGURU_ALERT_LEVEL_RE = re.compile(r"\|\s*(WARNING|ERROR|CRITICAL)\s*\|")
# Free-form worker / print noise that still matters.
ERROR_LINE_RE = re.compile(
    r"(?i)\b("
    r"traceback|exception|error|cuda\s*error|out of memory|oom|"
    r"nan|inf\b|killed|segfault|access violation|failed to|"
    r"shared.?memory|FileMapping|rollout_log_prob|"
    r"graft|shape mismatch|missing key|unexpected key|incompatible"
    r")\b"
)

TRAINING_PROCESS_NEEDLES = (
    "async_train_universal.py",
    "train_universal.py",
    "train_pipeline.py",
)

# Operational noise. A refused PPO step is the replay guard dropping one bad
# buffer; the TensorBoard HTTP port is optional (the watcher reads event files
# directly). A real collector/learner disconnect still raises CRIT from
# train/rollout_log_prob_abs_err.
BENIGN_ALERT_RE = re.compile(
    r"TensorBoard process started but port"
    r"|flat replay abs err"
    r"|weights were not stepped and will not be republished"
    r"|Update skipped after replay mismatch"
    r"|skipped periodic checkpoint save \(replay mismatch"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _repo_root_from_env_or_file() -> str:
    env = os.environ.get("SC2_REPO_ROOT", "").strip()
    if env:
        return os.path.abspath(env)
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def _health_dir(repo_root: str, log_dir_base: str = "logs_universal") -> str:
    path = os.path.join(repo_root, log_dir_base, HEALTH_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _parse_alert_timestamp(line: str) -> Optional[datetime]:
    """Parse leading loguru wall time from an alert line, if present."""
    # Allow "process_alerts.log: 2026-09-27 15:54:17.276 | ..."
    text = line
    if ": " in line and not line[:1].isdigit():
        # Drop source prefix once.
        text = line.split(": ", 1)[1]
    match = ALERT_TS_RE.match(text) or ALERT_TS_RE.match(line)
    if not match:
        return None
    try:
        # Treat naive timestamps as local wall time from the training machine.
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def read_session_start(health_dir: str) -> Optional[datetime]:
    path = os.path.join(health_dir, SESSION_START_NAME)
    if not os.path.isfile(path):
        return None
    try:
        raw = open(path, "r", encoding="utf-8").read().strip()
        # Stored as ISO local-naive or with timezone.
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        return datetime.fromisoformat(raw).replace(tzinfo=None)
    except (OSError, ValueError):
        return None


def write_session_start(health_dir: str, when: Optional[datetime] = None) -> datetime:
    when = when or datetime.now()
    path = os.path.join(health_dir, SESSION_START_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(when.isoformat(timespec="seconds") + "\n")
    return when


def reset_watch_session(
    repo_root: str,
    *,
    log_dir_base: str = "logs_universal",
    worker_logs_dir: str = "logs",
) -> dict[str, int]:
    """
    Fresh watcher session: drop old health reports and clear process alert history.

    Call when the watch process starts so previously fixed issues cannot linger.
    """
    health = _health_dir(repo_root, log_dir_base)
    removed_reports = 0
    for name in os.listdir(health):
        if name == LOCK_NAME:
            continue
        path = os.path.join(health, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed_reports += 1
        except OSError:
            pass

    alerts_cleared = 0
    abs_worker = os.path.join(repo_root, worker_logs_dir)
    os.makedirs(abs_worker, exist_ok=True)
    alerts_path = os.path.join(abs_worker, PROCESS_ALERTS_NAME)
    for path in glob.glob(alerts_path + "*"):
        try:
            if os.path.abspath(path) == os.path.abspath(alerts_path):
                # Truncate in place so an already-open loguru sink keeps working.
                with open(path, "w", encoding="utf-8"):
                    pass
                alerts_cleared += 1
            elif os.path.isfile(path):
                os.remove(path)
                alerts_cleared += 1
        except OSError:
            pass

    write_session_start(health)
    return {
        "removed_reports": removed_reports,
        "alerts_cleared": alerts_cleared,
    }


def _tail_text_file(path: str, max_lines: int) -> list[str]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def _file_age_minutes(path: str, now: Optional[float] = None) -> Optional[float]:
    if not os.path.exists(path):
        return None
    now = time.time() if now is None else now
    try:
        return max(0.0, (now - os.path.getmtime(path)) / 60.0)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Digest building
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    severity: str  # OK | WARN | CRIT
    code: str
    message: str


@dataclass
class Digest:
    repo_root: str
    active_level: Optional[str]
    active_log_dir: Optional[str]
    training_alive: bool
    training_pids: list[int] = field(default_factory=list)
    progress_rows: list[dict[str, str]] = field(default_factory=list)
    tb_latest: dict[str, tuple[int, float]] = field(default_factory=dict)
    alert_lines: list[str] = field(default_factory=list)
    vram_tail: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    generated_at: str = ""

    @property
    def severity(self) -> str:
        rank = {"OK": 0, "WARN": 1, "CRIT": 2}
        best = "OK"
        for finding in self.findings:
            if rank.get(finding.severity, 0) > rank.get(best, 0):
                best = finding.severity
        return best


def detect_active_log_level(log_dir_base: str) -> Optional[str]:
    """Most recently touched curriculum folder under logs_universal."""
    if not os.path.isdir(log_dir_base):
        return None
    newest_name: Optional[str] = None
    newest_mtime = -1.0
    for name in os.listdir(log_dir_base):
        if name.startswith("_") or name.startswith("."):
            continue
        path = os.path.join(log_dir_base, name)
        if not os.path.isdir(path):
            continue
        try:
            mtime = os.path.getmtime(path)
            for child in os.listdir(path):
                child_path = os.path.join(path, child)
                try:
                    mtime = max(mtime, os.path.getmtime(child_path))
                except OSError:
                    pass
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest_name = name
    return newest_name


def find_training_pids() -> list[int]:
    """Return PIDs whose command line looks like our training entrypoints."""
    pids: list[int] = []
    try:
        import psutil  # type: ignore

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
            except (psutil.Error, TypeError):
                continue
            if any(needle in cmdline for needle in TRAINING_PROCESS_NEEDLES):
                pids.append(int(proc.info["pid"]))
        return sorted(set(pids))
    except Exception:
        pass

    # Fallback: Windows tasklist / Linux ps (best-effort, no cmdline on tasklist).
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
                text=True,
                errors="replace",
                timeout=20,
            )
            for line in out.splitlines():
                if any(needle in line for needle in TRAINING_PROCESS_NEEDLES):
                    parts = [p.strip() for p in line.split(",")]
                    if parts and parts[-1].isdigit():
                        pids.append(int(parts[-1]))
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, errors="replace")
            for line in out.splitlines():
                if any(needle in line for needle in TRAINING_PROCESS_NEEDLES):
                    parts = line.strip().split(None, 1)
                    if parts and parts[0].isdigit():
                        pids.append(int(parts[0]))
        except (OSError, subprocess.SubprocessError):
            pass
    return sorted(set(pids))


def read_progress_tail(csv_path: str, n_rows: int) -> list[dict[str, str]]:
    if not os.path.isfile(csv_path):
        return []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError:
        return []
    return rows[-n_rows:]


def read_tb_latest_scalars(log_dir: str, tags: tuple[str, ...]) -> dict[str, tuple[int, float]]:
    """Latest (step, value) for selected TensorBoard scalar tags."""
    if not os.path.isdir(log_dir):
        return {}
    event_files = [name for name in os.listdir(log_dir) if "tfevents" in name]
    if not event_files:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    size_guidance = {
        "scalars": 0,
        "tensors": 0,
        "histograms": 0,
        "images": 0,
        "audio": 0,
        "compressedHistograms": 0,
    }
    try:
        accumulator = EventAccumulator(log_dir, size_guidance=size_guidance)
        accumulator.Reload()
    except Exception:
        return {}

    available = set(accumulator.Tags().get("scalars", []))
    out: dict[str, tuple[int, float]] = {}
    for tag in tags:
        if tag not in available:
            continue
        try:
            events = accumulator.Scalars(tag)
        except Exception:
            continue
        if not events:
            continue
        last = events[-1]
        out[tag] = (int(last.step), float(last.value))
    return out


def is_benign_alert_line(line: str) -> bool:
    """True for expected log lines that must not raise digest severity."""
    return bool(BENIGN_ALERT_RE.search(line))


def _line_is_alert(line: str) -> bool:
    if not line.strip() or is_benign_alert_line(line):
        return False
    if LOGURU_ALERT_LEVEL_RE.search(line):
        return True
    return bool(ERROR_LINE_RE.search(line))


def extract_alert_lines(
    paths: list[str],
    *,
    max_scan_lines: int = DEFAULT_LOG_TAIL_LINES,
    max_hits: int = DEFAULT_ALERT_MAX_LINES,
    since: Optional[datetime] = None,
) -> list[str]:
    """
    Collect recent WARN/ERROR/graft-like lines from process + worker logs.

    When ``since`` is set, drop alert lines with an older parseable timestamp
    (and drop untimestamped lines from ``process_alerts.log`` only if ``since``
    is set — those are treated as pre-session noise after a restart truncate).
    """
    hits: list[str] = []
    existing = [p for p in paths if p and os.path.isfile(p)]
    existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for path in existing:
        label = os.path.basename(path)
        is_process_alerts = label.startswith(PROCESS_ALERTS_NAME)
        for line in _tail_text_file(path, max_scan_lines):
            if not _line_is_alert(line):
                continue
            if since is not None:
                ts = _parse_alert_timestamp(line)
                if ts is not None and ts < since:
                    continue
                if ts is None and is_process_alerts:
                    # After session reset, ignore undated leftover noise.
                    continue
            hits.append(f"{label}: {line[:400]}")
        if len(hits) >= max_hits * 2:
            break
    return hits[-max_hits:]


def collect_alert_log_paths(repo_root: str, worker_logs_dir: str = "logs") -> list[str]:
    abs_worker = os.path.join(repo_root, worker_logs_dir)
    paths = [os.path.join(abs_worker, PROCESS_ALERTS_NAME)]
    paths.extend(sorted(glob.glob(os.path.join(abs_worker, "worker_*.log"))))
    # Rotated process_alerts.log.* from loguru retention, if any.
    paths.extend(sorted(glob.glob(os.path.join(abs_worker, f"{PROCESS_ALERTS_NAME}*"))))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def scan_worker_logs(logs_dir: str, max_lines: int, max_hits: int = 40) -> list[str]:
    """Backward-compatible wrapper; prefer ``extract_alert_lines``."""
    paths = [os.path.join(logs_dir, PROCESS_ALERTS_NAME)]
    paths.extend(sorted(glob.glob(os.path.join(logs_dir, "worker_*.log"))))
    return extract_alert_lines(paths, max_scan_lines=max_lines, max_hits=max_hits)


def evaluate_findings(
    *,
    training_alive: bool,
    active_log_dir: Optional[str],
    tb_latest: dict[str, tuple[int, float]],
    alert_lines: list[str],
    stale_minutes: float,
    now: Optional[float] = None,
) -> list[Finding]:
    findings: list[Finding] = []
    now = time.time() if now is None else now

    if not training_alive:
        findings.append(
            Finding(
                "CRIT" if active_log_dir else "WARN",
                "training_process",
                "No async/universal training process found "
                f"({', '.join(TRAINING_PROCESS_NEEDLES)}).",
            )
        )
    else:
        findings.append(
            Finding("OK", "training_process", "Training process is running.")
        )

    if active_log_dir is None:
        findings.append(
            Finding("WARN", "active_level", "No active level folder under logs_universal.")
        )
    else:
        age = _file_age_minutes(active_log_dir, now=now)
        # Prefer newest artifact inside the level dir.
        newest_age = age
        try:
            for name in os.listdir(active_log_dir):
                child_age = _file_age_minutes(os.path.join(active_log_dir, name), now=now)
                if child_age is not None and (newest_age is None or child_age < newest_age):
                    newest_age = child_age
        except OSError:
            pass
        if training_alive and newest_age is not None and newest_age > stale_minutes:
            findings.append(
                Finding(
                    "WARN",
                    "stale_logs",
                    f"Active log dir looks stale ({newest_age:.0f} min since last write; "
                    f"threshold {stale_minutes:.0f} min).",
                )
            )
        elif training_alive:
            findings.append(
                Finding("OK", "stale_logs", "Active level logs were updated recently.")
            )

    def _tb(tag: str) -> Optional[float]:
        item = tb_latest.get(tag)
        return None if item is None else item[1]

    abs_err = _tb("train/rollout_log_prob_abs_err")
    if abs_err is None:
        abs_err = _tb("train/rollout_log_prob_abs_err_flat")
    if abs_err is not None:
        if abs_err > 1e-2:
            findings.append(
                Finding(
                    "CRIT",
                    "log_prob_mismatch",
                    f"rollout_log_prob_abs_err={abs_err:.4g} ≫ 1e-2 "
                    "(collector/learner obs disconnect).",
                )
            )
        elif abs_err > 1e-3:
            findings.append(
                Finding(
                    "WARN",
                    "log_prob_mismatch",
                    f"rollout_log_prob_abs_err={abs_err:.4g} elevated (healthy ~1e-4).",
                )
            )
        else:
            findings.append(
                Finding("OK", "log_prob_mismatch", f"rollout_log_prob_abs_err={abs_err:.4g}.")
            )

    load_ok = _tb("train/rollout_policy_load_ok")
    if load_ok is not None:
        if load_ok < 0.5:
            findings.append(
                Finding(
                    "CRIT",
                    "policy_load",
                    f"rollout_policy_load_ok={load_ok:.3g} (learner did not load rollout zip).",
                )
            )
        else:
            findings.append(
                Finding("OK", "policy_load", f"rollout_policy_load_ok={load_ok:.3g}.")
            )

    approx_kl = _tb("train/approx_kl")
    if approx_kl is not None:
        if approx_kl > 0.1:
            findings.append(
                Finding(
                    "CRIT",
                    "approx_kl",
                    f"approx_kl={approx_kl:.4g} ≫ 0.1 (policy update exploded).",
                )
            )
        elif approx_kl > 0.05:
            findings.append(
                Finding("WARN", "approx_kl", f"approx_kl={approx_kl:.4g} high (healthy ~0.001–0.02).")
            )
        else:
            findings.append(Finding("OK", "approx_kl", f"approx_kl={approx_kl:.4g}."))

    expl_var = _tb("train/explained_variance")
    if expl_var is not None:
        if expl_var < -0.5:
            findings.append(
                Finding(
                    "WARN",
                    "explained_variance",
                    f"explained_variance={expl_var:.3g} strongly negative "
                    "(alarm if stuck across many rollouts).",
                )
            )
        else:
            findings.append(
                Finding("OK", "explained_variance", f"explained_variance={expl_var:.3g}.")
            )

    value_loss = _tb("train/value_loss")
    if value_loss is not None and (value_loss != value_loss or abs(value_loss) == float("inf")):
        findings.append(
            Finding("CRIT", "value_loss", f"value_loss is non-finite ({value_loss}).")
        )

    scored_alerts = [line for line in alert_lines if not is_benign_alert_line(line)]
    if scored_alerts:
        err_hits = [
            line
            for line in scored_alerts
            if re.search(r"\|\s*(ERROR|CRITICAL)\s*\|", line)
            or re.search(r"(?i)\b(traceback|exception)\b", line)
        ]
        graft_hits = [
            line
            for line in scored_alerts
            if re.search(r"(?i)graft|shape mismatch", line)
        ]
        if err_hits:
            findings.append(
                Finding(
                    "CRIT",
                    "alert_logs",
                    f"{len(err_hits)} ERROR/Traceback-like line(s) in alert excerpt "
                    f"(of {len(scored_alerts)} total alerts).",
                )
            )
        elif graft_hits:
            findings.append(
                Finding(
                    "WARN",
                    "alert_logs",
                    f"{len(graft_hits)} graft/shape-mismatch line(s) in alert excerpt — "
                    "verify channel/head sizes after curriculum load.",
                )
            )
        else:
            findings.append(
                Finding(
                    "WARN",
                    "alert_logs",
                    f"{len(scored_alerts)} warning-like line(s) in alert excerpt.",
                )
            )
    else:
        findings.append(Finding("OK", "alert_logs", "No WARN/ERROR/graft lines in recent alert logs."))

    if not findings:
        findings.append(Finding("OK", "baseline", "No issues detected."))
    return findings


def build_digest(
    repo_root: str,
    *,
    log_dir_base: str = "logs_universal",
    worker_logs_dir: str = "logs",
    stale_minutes: float = DEFAULT_STALE_MINUTES,
    progress_tail_rows: int = DEFAULT_PROGRESS_TAIL_ROWS,
    log_tail_lines: int = DEFAULT_LOG_TAIL_LINES,
    alert_max_lines: int = DEFAULT_ALERT_MAX_LINES,
    alert_since: Optional[datetime] = None,
) -> Digest:
    abs_log_base = os.path.join(repo_root, log_dir_base)
    abs_worker_dir = os.path.join(repo_root, worker_logs_dir)
    health = _health_dir(repo_root, log_dir_base)
    since = alert_since if alert_since is not None else read_session_start(health)
    active = detect_active_log_level(abs_log_base)
    active_dir = os.path.join(abs_log_base, active) if active else None
    pids = find_training_pids()
    progress_rows: list[dict[str, str]] = []
    tb_latest: dict[str, tuple[int, float]] = {}
    if active_dir:
        progress_rows = read_progress_tail(
            os.path.join(active_dir, "progress.csv"), progress_tail_rows
        )
        tb_latest = read_tb_latest_scalars(active_dir, KEY_TB_TAGS)
    alert_lines = extract_alert_lines(
        collect_alert_log_paths(repo_root, worker_logs_dir),
        max_scan_lines=log_tail_lines,
        max_hits=alert_max_lines,
        since=since,
    )
    vram_tail = _tail_text_file(os.path.join(abs_worker_dir, "vram_monitor.log"), 15)
    findings = evaluate_findings(
        training_alive=bool(pids),
        active_log_dir=active_dir,
        tb_latest=tb_latest,
        alert_lines=alert_lines,
        stale_minutes=stale_minutes,
    )
    return Digest(
        repo_root=repo_root,
        active_level=active,
        active_log_dir=active_dir,
        training_alive=bool(pids),
        training_pids=pids,
        progress_rows=progress_rows,
        tb_latest=tb_latest,
        alert_lines=alert_lines,
        vram_tail=vram_tail,
        findings=findings,
        generated_at=_utc_now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def render_digest_markdown(digest: Digest) -> str:
    lines: list[str] = [
        f"# Training health digest",
        "",
        f"- Generated: {digest.generated_at}",
        f"- Severity: **{digest.severity}**",
        f"- Repo: `{digest.repo_root}`",
        f"- Active level: `{digest.active_level or 'none'}`",
        f"- Training alive: {digest.training_alive} (pids={digest.training_pids or 'none'})",
        "",
        "## Findings",
        "",
    ]
    for finding in digest.findings:
        lines.append(f"- **{finding.severity}** `{finding.code}`: {finding.message}")

    lines.extend(["", "## TensorBoard latest scalars", ""])
    if not digest.tb_latest:
        lines.append("_No scalar tags found (missing/empty tfevents or tensorboard unavailable)._")
    else:
        for tag, (step, value) in sorted(digest.tb_latest.items()):
            lines.append(f"- `{tag}` @ step {step}: {value:.6g}")

    lines.extend(["", "## progress.csv tail", ""])
    if not digest.progress_rows:
        lines.append("_No progress.csv rows._")
    else:
        keys = list(digest.progress_rows[-1].keys())
        interesting = [
            k
            for k in keys
            if any(
                part in k
                for part in (
                    "ep_rew",
                    "explained",
                    "approx_kl",
                    "clip_fraction",
                    "log_prob",
                    "policy_load",
                    "value_loss",
                    "entropy",
                    "fps",
                    "total_timesteps",
                    "learner_ready",
                )
            )
        ] or keys[:12]
        lines.append("| " + " | ".join(interesting) + " |")
        lines.append("| " + " | ".join("---" for _ in interesting) + " |")
        for row in digest.progress_rows:
            cells = [str(row.get(k, ""))[:24] for k in interesting]
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Warn / Error / Graft excerpt", ""])
    lines.append(
        f"_At most {DEFAULT_ALERT_MAX_LINES} recent alert lines from "
        f"`logs/{PROCESS_ALERTS_NAME}` and worker logs. Full INFO logs are not included._"
    )
    lines.append("")
    if not digest.alert_lines:
        lines.append("_None._")
    else:
        lines.append("```")
        lines.extend(digest.alert_lines)
        lines.append("```")

    lines.extend(["", "## VRAM monitor tail", ""])
    if not digest.vram_tail:
        lines.append("_Empty / missing `logs/vram_monitor.log`._")
    else:
        lines.append("```")
        lines.extend(digest.vram_tail[-8:])
        lines.append("```")

    lines.append("")
    return "\n".join(lines)


def write_alerts_excerpt(digest: Digest, health_dir: str) -> str:
    path = os.path.join(health_dir, ALERTS_EXCERPT_NAME)
    lines = [
        "# Training alert excerpt (WARN / ERROR / graft)",
        "",
        f"- Generated: {digest.generated_at}",
        f"- Lines: {len(digest.alert_lines)}",
        "",
        "Do not open full worker or process logs unless this excerpt is empty "
        "and metrics already look broken.",
        "",
    ]
    if not digest.alert_lines:
        lines.append("_No alert lines in the scanned window._")
    else:
        lines.append("```")
        lines.extend(digest.alert_lines)
        lines.append("```")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def write_digest(digest: Digest, health_dir: str) -> str:
    """Overwrite the single rolling digest (no timestamped copies)."""
    path = os.path.join(health_dir, DIGEST_NAME)
    text = render_digest_markdown(digest)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    write_alerts_excerpt(digest, health_dir)
    return path


def write_deterministic_report(digest: Digest, health_dir: str, note: str = "") -> str:
    path = os.path.join(health_dir, REPORT_NAME)
    lines = [
        f"# Training health report",
        "",
        f"- Generated: {digest.generated_at}",
        f"- Severity: **{digest.severity}**",
        f"- Active level: `{digest.active_level or 'none'}`",
        "",
        "## Summary",
        "",
    ]
    crit = [f for f in digest.findings if f.severity == "CRIT"]
    warn = [f for f in digest.findings if f.severity == "WARN"]
    if crit:
        lines.append("Critical:")
        for finding in crit:
            lines.append(f"- {finding.message}")
    if warn:
        lines.append("Warnings:")
        for finding in warn:
            lines.append(f"- {finding.message}")
    if not crit and not warn:
        lines.append("No deterministic alarms. Training signals look within expected ranges.")
    if note:
        lines.extend(["", f"_Note: {note}_", ""])
    lines.extend(
        [
            "",
            f"Rolling digest: `{DIGEST_NAME}` (overwritten each check)",
            f"Alert excerpt: `{ALERTS_EXCERPT_NAME}`",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Cursor agent CLI
# ---------------------------------------------------------------------------

def resolve_agent_bin(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit if os.path.isfile(explicit) else shutil.which(explicit)
    env = os.environ.get("CURSOR_AGENT_BIN", "").strip()
    if env:
        return env if os.path.isfile(env) else shutil.which(env)
    for name in ("agent", "cursor-agent"):
        found = shutil.which(name)
        if found:
            return found
    # Common Windows install locations (best-effort; do not scan the whole profile).
    candidates = [
        os.path.join(os.path.expanduser("~"), ".local", "bin", "agent.exe"),
        os.path.join(os.path.expanduser("~"), ".local", "bin", "agent.cmd"),
        os.path.join(os.path.expanduser("~"), ".local", "bin", "agent"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "cursor-agent", "agent.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "cursor-agent", "agent.cmd"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def build_llm_prompt(digest_path: str, alerts_path: str, report_path: str) -> str:
    return f"""You are monitoring SC2 unit-agent training on this local machine.

Read ONLY these files (do not invent metrics; do not open full worker/process logs):
1. AGENTS.md — section "PPO training metrics" / TensorBoard alarm table (skim)
2. Compact digest: `{digest_path}`
3. Warn/Error/Graft excerpt only: `{alerts_path}`

Write a concise status report to `{report_path}` with:
- Severity line: OK | WARN | CRIT
- 3–8 bullet findings with concrete evidence (metric values / alert lines)
- One "Likely cause" bullet if severity is WARN/CRIT
- One "Suggested next check" bullet

Rules:
- Stay cheap: do not explore the repo beyond the files above unless a cited path is missing.
- Do not modify training code or configs.
- Do not restart training.
- Prefer false negatives over noisy alarms for explained_variance sag alone (normal in CTF self-play).
- Graft / shape-mismatch lines in the alert excerpt are high priority even if TensorBoard looks "ok".
- Treat rollout_log_prob_abs_err ≫ 1e-2, rollout_policy_load_ok=0, approx_kl ≫ 0.1 with log-prob mismatch, NaNs, and Tracebacks as high priority.
"""


def run_llm_analysis(
    *,
    repo_root: str,
    digest_path: str,
    alerts_path: str,
    report_path: str,
    agent_bin: Optional[str],
    model: Optional[str],
    timeout_sec: int,
) -> tuple[bool, str]:
    binary = resolve_agent_bin(agent_bin)
    if not binary:
        return False, (
            "Cursor agent CLI not found. Install it, put `agent` on PATH, "
            "or set CURSOR_AGENT_BIN. Wrote deterministic report only."
        )

    prompt = build_llm_prompt(digest_path, alerts_path, report_path)
    model_id = (model or DEFAULT_LLM_MODEL).strip()
    cmd = [
        binary,
        "-p",
        "--trust",
        "--force",
        "--workspace",
        repo_root,
        "--model",
        model_id,
        prompt,
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return False, f"agent timed out after {timeout_sec}s"
    except OSError as exc:
        return False, f"failed to launch agent: {exc}"

    if completed.returncode != 0 and not os.path.isfile(report_path):
        err = (completed.stderr or completed.stdout or "").strip()[-800:]
        return False, f"agent exited {completed.returncode}: {err}"

    if not os.path.isfile(report_path):
        # Agent printed analysis but did not write the file — capture stdout.
        body = (completed.stdout or "").strip()
        if body:
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write("# Training health report (from agent stdout)\n\n")
                handle.write(body)
                handle.write("\n")
            return True, f"agent ({model_id}) finished; saved stdout into latest.md"
        return False, "agent finished but did not create latest.md"

    return True, f"agent ({model_id}) wrote {report_path}"


# ---------------------------------------------------------------------------
# Lock / watch loop
# ---------------------------------------------------------------------------

class WatchLock:
    def __init__(self, path: str):
        self.path = path
        self._held = False

    def acquire(self) -> bool:
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    old_pid = int(handle.read().strip() or "0")
            except (OSError, ValueError):
                old_pid = 0
            if old_pid and _pid_alive(old_pid):
                print(f"Another watcher is running (pid {old_pid}); exiting.", flush=True)
                return False
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self._held = True
        atexit.register(self.release)
        return True

    def release(self) -> None:
        if not self._held:
            return
        try:
            if os.path.isfile(self.path):
                with open(self.path, "r", encoding="utf-8") as handle:
                    if handle.read().strip() == str(os.getpid()):
                        os.remove(self.path)
        except OSError:
            pass
        self._held = False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    except Exception:
        pass
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"],
                text=True,
                errors="replace",
                timeout=10,
            )
            return str(pid) in out and "No tasks" not in out
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_once(args: argparse.Namespace) -> int:
    repo_root = os.path.abspath(args.repo_root)
    health = _health_dir(repo_root, args.log_dir_base)
    digest = build_digest(
        repo_root,
        log_dir_base=args.log_dir_base,
        worker_logs_dir=args.worker_logs_dir,
        stale_minutes=args.stale_minutes,
        progress_tail_rows=args.progress_tail_rows,
        log_tail_lines=args.log_tail_lines,
        alert_max_lines=args.alert_max_lines,
    )
    digest_path = write_digest(digest, health)
    report_path = os.path.join(health, REPORT_NAME)

    print(
        f"[{digest.generated_at}] severity={digest.severity} "
        f"level={digest.active_level} alive={digest.training_alive}",
        flush=True,
    )
    print(f"Digest: {digest_path}", flush=True)

    use_llm = not args.no_llm
    # Default: only spend LLM tokens when deterministic severity is not OK.
    llm_on_warn_only = not getattr(args, "llm_always", False)
    if use_llm and llm_on_warn_only and digest.severity == "OK":
        use_llm = False
        write_deterministic_report(
            digest,
            health,
            note="LLM skipped (severity OK; default --llm-on-warn-only). Use --llm-always to force.",
        )
        print(f"Report: {report_path} (deterministic, LLM skipped)", flush=True)
        return 0

    if use_llm:
        alerts_path = os.path.join(health, ALERTS_EXCERPT_NAME)
        ok, msg = run_llm_analysis(
            repo_root=repo_root,
            digest_path=digest_path,
            alerts_path=alerts_path,
            report_path=report_path,
            agent_bin=args.agent_bin,
            model=args.model,
            timeout_sec=args.llm_timeout,
        )
        print(msg, flush=True)
        if not ok:
            write_deterministic_report(digest, health, note=msg)
    else:
        write_deterministic_report(digest, health, note="LLM disabled (--no-llm).")
        print(f"Report: {report_path}", flush=True)

    return 0 if digest.severity != "CRIT" else 2


def watch_loop(args: argparse.Namespace) -> int:
    repo_root = os.path.abspath(args.repo_root)
    health = _health_dir(repo_root, args.log_dir_base)
    lock = WatchLock(os.path.join(health, LOCK_NAME))
    if not lock.acquire():
        return 1

    stats = reset_watch_session(
        repo_root,
        log_dir_base=args.log_dir_base,
        worker_logs_dir=args.worker_logs_dir,
    )
    # Re-acquire marker after wipe (reset keeps lock file if present; ensure pid file exists).
    if not os.path.isfile(os.path.join(health, LOCK_NAME)):
        with open(os.path.join(health, LOCK_NAME), "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))

    stop = {"flag": False}

    def _stop(*_args: Any) -> None:
        stop["flag"] = True
        print("Stop requested; exiting after current cycle...", flush=True)

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    print(
        f"Training health watcher started. interval={args.interval}s "
        f"repo={args.repo_root}",
        flush=True,
    )
    print(
        f"Fresh session: removed {stats['removed_reports']} old report file(s), "
        f"cleared {stats['alerts_cleared']} process alert file(s).",
        flush=True,
    )
    while not stop["flag"]:
        started = time.time()
        try:
            run_once(args)
        except Exception as exc:
            print(f"Watch cycle failed: {exc}", flush=True)
        elapsed = time.time() - started
        sleep_for = max(1.0, float(args.interval) - elapsed)
        end = time.time() + sleep_for
        while not stop["flag"] and time.time() < end:
            time.sleep(min(1.0, end - time.time()))
    lock.release()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SC2 training health watcher")
    parser.add_argument(
        "--repo-root",
        default=_repo_root_from_env_or_file(),
        help="Repository root (default: SC2_REPO_ROOT or parent of train/)",
    )
    parser.add_argument("--once", action="store_true", help="Run a single check and exit")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Reset health reports + process_alerts before this run "
        "(always done automatically in watch mode)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SEC,
        help="Seconds between checks in watch mode (default: 3600)",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip Cursor agent analysis")
    parser.add_argument(
        "--llm-always",
        action="store_true",
        help="Call the LLM even when deterministic severity is OK (default: only WARN/CRIT)",
    )
    parser.add_argument(
        "--llm-on-warn-only",
        action="store_true",
        help=argparse.SUPPRESS,  # kept for older bats; now the default
    )
    parser.add_argument("--agent-bin", default=None, help="Path to Cursor agent CLI")
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"Cursor agent model id (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument("--llm-timeout", type=int, default=600, help="Agent timeout seconds")
    parser.add_argument("--log-dir-base", default="logs_universal")
    parser.add_argument("--worker-logs-dir", default="logs")
    parser.add_argument("--stale-minutes", type=float, default=DEFAULT_STALE_MINUTES)
    parser.add_argument("--progress-tail-rows", type=int, default=DEFAULT_PROGRESS_TAIL_ROWS)
    parser.add_argument("--log-tail-lines", type=int, default=DEFAULT_LOG_TAIL_LINES)
    parser.add_argument("--alert-max-lines", type=int, default=DEFAULT_ALERT_MAX_LINES)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.repo_root = os.path.abspath(args.repo_root)
    if not os.path.isdir(args.repo_root):
        print(f"ERROR: repo root not found: {args.repo_root}", file=sys.stderr)
        return 1
    if args.once:
        if args.fresh:
            stats = reset_watch_session(
                args.repo_root,
                log_dir_base=args.log_dir_base,
                worker_logs_dir=args.worker_logs_dir,
            )
            print(
                f"Fresh session: removed {stats['removed_reports']} old report file(s), "
                f"cleared {stats['alerts_cleared']} process alert file(s).",
                flush=True,
            )
        return run_once(args)
    return watch_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
