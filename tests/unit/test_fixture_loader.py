"""Unit tests for golden fixture store and loader."""

from pathlib import Path
import pytest
from comics_panel_extraction.io.fixture_loader import GoldenFixtureStore


@pytest.mark.unit
def test_manifest_parsing(fixture_store: GoldenFixtureStore):
    fixtures = fixture_store.list_fixtures()
    assert len(fixtures) == 16
    
    # Check comic and manga breakdown
    comic = [f for f in fixtures if f.dataset == "Comic"]
    manga = [f for f in fixtures if f.dataset == "Manga"]
    assert len(comic) == 8
    assert len(manga) == 8


@pytest.mark.unit
def test_fixture_verification(fixture_store: GoldenFixtureStore):
    rec = fixture_store.get_record("FIX_COMIC_01")
    assert rec is not None
    assert rec.dimensions == [448, 448]
    assert rec.evaluation_ready is True
    
    # Check artifact-level availability and SHA verification
    if fixture_store.is_available("FIX_COMIC_01", "final_mask"):
        assert fixture_store.verify_artifact_sha("FIX_COMIC_01", "final_mask") is True
        mask = fixture_store.load_mask("FIX_COMIC_01", "final_mask")
        assert mask.shape == (448, 448)
    else:
        pytest.skip("Local fixture bundle not populated in test environment.")
