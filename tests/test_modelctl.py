# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

import modelctl  # noqa: E402


def test_list_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    rc = modelctl.main(["list", "--models-dir", str(tmp_path)])
    assert rc == 0


def test_profile_error_exit_code(tmp_path, capsys):
    rc = modelctl.main(["start", "ghost", "--models-dir", str(tmp_path)])
    assert rc == 2
    assert "不存在" in capsys.readouterr().out


def test_status_output(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.yaml").write_text("name: a\nengine: vllm\nport: 8000\n", encoding="utf-8")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    rc = modelctl.main(["status", "--models-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "a" in out and "vllm" in out and "8000" in out
