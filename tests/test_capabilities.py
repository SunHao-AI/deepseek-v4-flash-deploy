# -*- coding: utf-8 -*-
"""core/capabilities.py 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.capabilities import cc_at_least, free_vram_total_mb, probe  # noqa: E402

SMI_5880 = "\n".join(["RTX 5880 Ada Generation, 49140, 48000, 580.65.05, 8.9"] * 8)


def test_probe_5880():
    caps = probe(nvidia_smi_output=SMI_5880)
    assert caps.gpu_count == 8
    assert caps.gpu_name == "RTX 5880 Ada Generation"
    assert caps.vram_total_mb == 49140
    assert caps.vram_free_mb == [48000] * 8
    assert caps.compute_capability == "8.9"
    assert caps.cuda_driver == "580.65.05"


def test_probe_failure_returns_empty():
    caps = probe(nvidia_smi_output="")
    assert caps.gpu_count == 0
    assert caps.compute_capability == ""


def test_cc_at_least():
    assert cc_at_least("8.9", 8, 9)
    assert cc_at_least("8.9", 8, 0)
    assert not cc_at_least("8.9", 9, 0)
    assert not cc_at_least("", 8, 0)


def test_free_vram_total():
    caps = probe(nvidia_smi_output=SMI_5880)
    assert free_vram_total_mb(caps) == 48000 * 8
