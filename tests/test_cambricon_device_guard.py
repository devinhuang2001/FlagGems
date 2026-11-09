# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""CPU-only tests for Cambricon device aliases, without importing MLU kernels."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


_HELPER = (
    Path(__file__).resolve().parents[1]
    / "src/flag_gems/runtime/backend/_cambricon/ops/_device_guard.py"
)


def load_guard(monkeypatch):
    fake_runtime = types.ModuleType("flag_gems.runtime")
    fake_runtime.device = types.SimpleNamespace(name="mlu")
    fake_package = types.ModuleType("flag_gems")
    fake_package.runtime = fake_runtime
    monkeypatch.setitem(sys.modules, "flag_gems", fake_package)
    monkeypatch.setitem(sys.modules, "flag_gems.runtime", fake_runtime)
    spec = importlib.util.spec_from_file_location("_cambricon_device_guard_test", _HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tensor_type,registered_name,expected",
    [
        ("mlu", "flagos", True),
        ("flagos", "flagos", True),
        ("cpu", "flagos", False),
        ("cuda", "flagos", False),
        ("flagos", "mlu", False),
    ],
)
def test_cambricon_backend_device_alias(
    monkeypatch, tensor_type, registered_name, expected
):
    guard = load_guard(monkeypatch)
    monkeypatch.setattr(
        torch._C, "_get_privateuse1_backend_name", lambda: registered_name
    )
    assert guard.is_backend_device_type(tensor_type) is expected


def test_cambricon_alias_registration_is_not_cached(monkeypatch):
    guard = load_guard(monkeypatch)
    registered = {"name": "mlu"}
    monkeypatch.setattr(
        torch._C, "_get_privateuse1_backend_name", lambda: registered["name"]
    )
    assert not guard.is_backend_device_type("flagos")
    registered["name"] = "flagos"
    assert guard.is_backend_device_type("flagos")
