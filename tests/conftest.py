"""Pytest configuration and global fixtures for comics_panel_extraction tests."""

from pathlib import Path
import pytest
from comics_panel_extraction.io.fixture_loader import GoldenFixtureStore, get_default_fixture_store


@pytest.fixture
def repo_root() -> Path:
    """Returns absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def fixture_store(repo_root: Path) -> GoldenFixtureStore:
    """Returns initialized GoldenFixtureStore."""
    manifest_p = repo_root / "tests" / "fixtures" / "golden_fixture_manifest.json"
    bundle_root = repo_root / ".local_fixtures"
    return GoldenFixtureStore(bundle_root=bundle_root, manifest_path=manifest_p)
