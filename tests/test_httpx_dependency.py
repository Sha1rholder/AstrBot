from pathlib import Path

import pytest
from packaging.requirements import Requirement

from astrbot.core.utils.toml_parser import read_pyproject_project_dependencies

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("package", "extras"),
    [("httpx", {"socks"}), ("httpx2", {"http2", "socks"})],
)
def test_http_client_dependencies_match_and_include_required_extras(package, extras):
    requirements = [
        line.split("#", 1)[0].strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    pyproject = read_pyproject_project_dependencies(PROJECT_ROOT / "pyproject.toml")
    dependencies = []
    for entries in (requirements, pyproject):
        matches = [
            dependency
            for entry in entries
            if (dependency := Requirement(entry)).name == package
        ]
        assert len(matches) == 1
        assert matches[0].extras == extras
        dependencies.append(matches[0])
    assert dependencies[0] == dependencies[1]
