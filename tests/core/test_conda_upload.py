"""Exercise publication orchestration without building or uploading packages."""

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("scenario", ["success", "build-failure", "empty", "no-token"])
def test_conda_upload_uses_active_tools_and_only_fresh_artifacts(tmp_path, scenario):
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "commands.log"
    stale = tmp_path / "old-build" / "noarch"
    stale.mkdir(parents=True)
    (stale / "tomobar-stale.conda").touch()
    scripts = {
        "conda": """#!/bin/bash
set -euo pipefail
echo build >> "$CI_TEST_LOG"
if [[ "$CI_TEST_SCENARIO" == build-failure ]]; then exit 7; fi
mkdir -p "$CONDA_BLD_PATH/noarch"
touch "$CONDA_BLD_PATH/noarch/unrelated.conda"
if [[ "$CI_TEST_SCENARIO" == success ]]; then
    touch "$CONDA_BLD_PATH/noarch/tomobar-new.conda"
    touch "$CONDA_BLD_PATH/noarch/tomobar-new.tar.bz2"
fi
""",
        "anaconda": """#!/bin/bash
set -euo pipefail
echo "$*" >> "$CI_TEST_LOG"
""",
    }
    for name, content in scripts.items():
        tool = tools / name
        tool.write_text(content)
        tool.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
        RUNNER_TEMP=str(tmp_path),
        CONDA_BLD_PATH=str(stale.parent),
        CI_TEST_LOG=str(log),
        CI_TEST_SCENARIO=scenario,
        ANACONDA_API_TOKEN="test-token",
    )
    env.pop("CONDA", None)  # Executables come from the activated environment.
    if scenario == "no-token":
        env.pop("ANACONDA_API_TOKEN")
    script = Path(__file__).resolve().parents[2] / ".scripts" / "conda_upload.sh"
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    calls = log.read_text().splitlines() if log.exists() else []
    if scenario == "success":
        assert result.returncode == 0, result.stderr
        assert calls[0] == "build"
        assert len(calls) == 3
        uploaded = [Path(call.split()[-2]) for call in calls[1:]]
        assert {path.name for path in uploaded} == {
            "tomobar-new.conda",
            "tomobar-new.tar.bz2",
        }
        assert all(path.parent != stale and path.is_file() for path in uploaded)
    else:
        assert result.returncode != 0
        assert calls == ([] if scenario == "no-token" else ["build"])
        if scenario == "empty":
            assert "No ToMoBAR packages were produced" in result.stderr
