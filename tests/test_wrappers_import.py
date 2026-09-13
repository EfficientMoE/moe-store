# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import importlib
import importlib.util
import pkgutil

import pytest

ENGINE_COUPLED = {
    "moe_store.wrappers.deepseek",
    "moe_store.wrappers.deepseek_v4.fp8_expert",
    "moe_store.wrappers.deepseek_v4.official_offload_adapter",
    "moe_store.wrappers.gpt_oss",
}


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


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param(
            module_name,
            marks=(
                pytest.mark.xfail(reason="requires MoE-Infinity engine runtime")
                if module_name in ENGINE_COUPLED
                else ()
            ),
        )
        for module_name in _wrapper_module_names()
    ],
)
def test_wrapper_module_import(module_name: str) -> None:
    importlib.import_module(module_name)
