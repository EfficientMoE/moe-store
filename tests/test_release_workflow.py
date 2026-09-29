from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_tag_release_is_github_primary_and_pypi_is_non_blocking():
    workflow = (WORKFLOWS / "publish.yml").read_text()

    assert "workflow_dispatch:" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "name: Publish GitHub Release" in workflow
    assert "gh release create" in workflow
    assert "needs: build" in workflow
    assert "name: Mirror optional release to PyPI" in workflow
    assert "needs: release" in workflow
    assert "continue-on-error: true" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow


def test_no_separate_pypi_workflow_can_gate_or_drift_from_release():
    assert not (WORKFLOWS / "publish-pypi.yml").exists()
