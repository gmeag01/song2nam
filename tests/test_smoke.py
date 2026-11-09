"""패키지 구조가 깨지지 않았는지 확인하는 기본 테스트."""

import importlib

import pytest

MODULES = [
    "ampcap",
    "ampcap.separate",
    "ampcap.research_llm",
    "ampcap.session",
    "ampcap.graybox",
    "ampcap.graybox.filters",
    "ampcap.graybox.blocks",
    "ampcap.graybox.amp",
    "ampcap.synth_data",
    "ampcap.encoder",
    "ampcap.stats",
    "ampcap.optimize",
    "ampcap.distill",
    "ampcap.evaluate",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
