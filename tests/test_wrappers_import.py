# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import importlib
import importlib.util
import pkgutil

import pytest


def _wrapper_module_names() -> list[str]:
    spec = importlib.util.find_spec("moe_store.wrappers")
    if spec is None or spec.submodule_search_locations is None:
        return ["moe_store.wrappers"]

    names = ["moe_store.wrappers"]
    names.extend(
        module.name
        for module in pkgutil.walk_packages(
            spec.submodule_search_locations,
            prefix="moe_store.wrappers.",
        )
    )
    return names


@pytest.mark.parametrize("module_name", _wrapper_module_names())
def test_wrapper_module_import(module_name: str) -> None:
    importlib.import_module(module_name)
