"""Reject truncated / non-MPQ SC2 map stubs."""

from pathlib import Path

from src.utils.sc2_map_files import is_usable_sc2map, list_usable_sc2map_stems


def test_is_usable_sc2map_rejects_truncated_1mib(tmp_path):
    stub = tmp_path / "AcropolisAIE.SC2Map"
    stub.write_bytes(b"MPQ" + b"\x00" * (1_048_576 - 3))
    assert not is_usable_sc2map(stub)


def test_is_usable_sc2map_accepts_large_mpq(tmp_path):
    good = tmp_path / "AbyssalReefAIE.SC2Map"
    good.write_bytes(b"MPQ" + b"\x00" * 2_500_000)
    assert is_usable_sc2map(good)


def test_is_usable_sc2map_rejects_non_mpq(tmp_path):
    bad = tmp_path / "Foo.SC2Map"
    bad.write_bytes(b"NOT" + b"\x00" * 3_000_000)
    assert not is_usable_sc2map(bad)


def test_list_usable_sc2map_stems_skips_stubs(tmp_path):
    (tmp_path / "Good.SC2Map").write_bytes(b"MPQ" + b"\x00" * 2_500_000)
    (tmp_path / "Bad.SC2Map").write_bytes(b"MPQ" + b"\x00" * 1_000_000)
    assert list_usable_sc2map_stems(tmp_path) == ["Good"]


def test_build_agent_acropolis_is_usable_after_repair():
    path = Path("train/maps/ladder/build_agent/AcropolisAIE.SC2Map")
    assert path.is_file()
    assert is_usable_sc2map(path)
    assert path.stat().st_size > 2_000_000
