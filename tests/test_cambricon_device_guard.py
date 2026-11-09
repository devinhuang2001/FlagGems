# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""CPU-only regression coverage for renamed Cambricon PrivateUse1 devices."""

import pytest
import torch

from flag_gems.runtime.backend._cambricon.ops._device_guard import (
    is_backend_device_type,
)


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
    monkeypatch.setattr(
        torch._C,
        "_get_privateuse1_backend_name",
        lambda: registered_name,
    )
    assert is_backend_device_type(tensor_type) is expected


def test_cambricon_alias_registration_is_not_cached(monkeypatch):
    registered = {"name": "mlu"}
    monkeypatch.setattr(
        torch._C,
        "_get_privateuse1_backend_name",
        lambda: registered["name"],
    )
    assert not is_backend_device_type("flagos")
    registered["name"] = "flagos"
    assert is_backend_device_type("flagos")
