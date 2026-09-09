"""IO subsystem for reading/writing images, masks, and resolving fixtures."""

from comics_panel_extraction.io.image_io import (
    prepare_output_path,
    load_model_ready_tensor,
    load_binary_mask,
    load_probability_mask,
    save_probability_mask,
    save_binary_mask,
)
from comics_panel_extraction.io.fixture_loader import (
    GoldenFixtureRecord,
    GoldenFixtureStore,
    get_default_fixture_store,
)

__all__ = [
    "prepare_output_path",
    "load_model_ready_tensor",
    "load_binary_mask",
    "load_probability_mask",
    "save_probability_mask",
    "save_binary_mask",
    "GoldenFixtureRecord",
    "GoldenFixtureStore",
    "get_default_fixture_store",
]
