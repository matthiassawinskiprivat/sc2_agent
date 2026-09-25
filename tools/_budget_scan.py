import yaml
from pathlib import Path

root = Path(r"D:/sc2_unit_agent/config_universal")
pipe = yaml.safe_load((root / "pipeline.yaml").read_text(encoding="utf-8"))
levels = pipe["levels"]
start = next(
    i
    for i, l in enumerate(levels)
    if l.startswith("ctf_") or l.startswith("scenario_") or l.startswith("aoe_")
)

print(f"{'level':55} {'maxM':>6} {'maxV':>6} {'sumM':>6} {'sumV':>6}")
prev = None
for name in levels[start:]:
    p = root / f"{name}.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    pools = (cfg.get("env") or {}).get("unit_pools") or []
    budgets = [
        (
            pool.get("race"),
            pool.get("high_level_command"),
            int(pool.get("mineral_budget") or 0),
            int(pool.get("vespene_budget") or 0),
        )
        for pool in pools
    ]
    max_m = max((b[2] for b in budgets), default=0)
    max_v = max((b[3] for b in budgets), default=0)
    sum_m = sum(b[2] for b in budgets)
    sum_v = sum(b[3] for b in budgets)
    jump = ""
    if prev is not None:
        dm, dv = max_m - prev[0], max_v - prev[1]
        if abs(dm) > 300 or abs(dv) > 200:
            jump = f"  << JUMP dm={dm:+d} dv={dv:+d}"
    print(f"{name:55} {max_m:6d} {max_v:6d} {sum_m:6d} {sum_v:6d}{jump}")
    for race, role, m, v in budgets:
        print(f"  - {race:8} {role or '?':7} m={m:5d} v={v:5d}")
    prev = (max_m, max_v)
