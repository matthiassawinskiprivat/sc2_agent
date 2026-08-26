"""view_tensorboard launcher builds a non-empty combined logdir_spec."""

from train.view_tensorboard import main


def test_view_tensorboard_print_spec_only(tmp_path, capsys):
    (tmp_path / "logs_universal" / "lvl").mkdir(parents=True)
    (tmp_path / "logs_build_zerg" / "build0").mkdir(parents=True)
    rc = main(["--repo-root", str(tmp_path), "--print-spec-only"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert "control:" in out
    assert "build:" in out
    assert "logs_universal" in out
    assert "logs_build_zerg" in out
