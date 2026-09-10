"""
Golden fixture loader and local bundle management utilities.

Allows locating, verifying, and reading local fixture bundles.
Gracefully skips tests when copyrighted local fixtures are absent.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


@dataclass(frozen=True)
class ArtifactEntry:
    """Artifact metadata within a golden fixture."""
    artifact_role: str
    legacy_source_path: Optional[str]
    local_fixture_rel_path: Optional[str]
    sha256: Optional[str]
    size_bytes: Optional[int]
    availability: str
    provenance_status: str


@dataclass(frozen=True)
class GoldenFixtureRecord:
    """Metadata record for a single golden fixture."""
    fixture_id: str
    page_id: str
    dataset: str
    dimensions: List[int]
    evaluation_ready: bool
    artifacts: Dict[str, ArtifactEntry]
    behavioral_purpose: str
    notes: str


class GoldenFixtureStore:
    """Manages access to local fixture bundle."""
    
    def __init__(self, bundle_root: Path, manifest_path: Path) -> None:
        self.bundle_root = Path(bundle_root)
        self.manifest_path = Path(manifest_path)
        self._fixtures: Dict[str, GoldenFixtureRecord] = {}
        self._load_manifest()
        
    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fixtures_list = data.get("fixtures", [])
        for item in fixtures_list:
            art_dict = {}
            for k, v in item.get("artifacts", {}).items():
                art_dict[k] = ArtifactEntry(
                    artifact_role=k,
                    legacy_source_path=v.get("legacy_source_path"),
                    local_fixture_rel_path=v.get("local_fixture_rel_path"),
                    sha256=v.get("sha256"),
                    size_bytes=v.get("size_bytes"),
                    availability=v.get("availability", "UNAVAILABLE"),
                    provenance_status=v.get("provenance_status", "UNRESOLVED"),
                )
            
            rec = GoldenFixtureRecord(
                fixture_id=item["fixture_id"],
                page_id=item["page_id"],
                dataset=item["dataset"],
                dimensions=item.get("dimensions", [448, 448]),
                evaluation_ready=item.get("evaluation_ready", False),
                artifacts=art_dict,
                behavioral_purpose=item.get("behavioral_purpose", ""),
                notes=item.get("notes", ""),
            )
            self._fixtures[rec.fixture_id] = rec
            
    def get_record(self, fixture_id: str) -> Optional[GoldenFixtureRecord]:
        return self._fixtures.get(fixture_id)
    
    def list_fixtures(self) -> List[GoldenFixtureRecord]:
        return list(self._fixtures.values())
    
    def resolve_artifact_path(self, fixture_id: str, artifact_key: str = "final_mask") -> Optional[Path]:
        rec = self.get_record(fixture_id)
        if not rec or artifact_key not in rec.artifacts:
            return None
        art = rec.artifacts[artifact_key]
        if not art.local_fixture_rel_path:
            return None
        full_path = self.bundle_root / art.local_fixture_rel_path
        return full_path if full_path.exists() else None
    
    def is_available(self, fixture_id: str, artifact_key: str = "final_mask") -> bool:
        path = self.resolve_artifact_path(fixture_id, artifact_key)
        return path is not None and path.exists()
    
    def verify_artifact_sha(self, fixture_id: str, artifact_key: str = "final_mask") -> bool:
        rec = self.get_record(fixture_id)
        if not rec or artifact_key not in rec.artifacts:
            return False
        art = rec.artifacts[artifact_key]
        if not art.local_fixture_rel_path or not art.sha256:
            return False
        path = self.bundle_root / art.local_fixture_rel_path
        if not path.exists():
            return False
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return actual_sha == art.sha256
    
    def load_mask(self, fixture_id: str, artifact_key: str = "final_mask") -> np.ndarray:
        rec = self.get_record(fixture_id)
        if not rec:
            raise KeyError(f"Fixture ID not found: {fixture_id}")
        if artifact_key not in rec.artifacts:
            raise KeyError(f"Artifact {artifact_key} not in fixture {fixture_id}")
        art = rec.artifacts[artifact_key]
        if not art.local_fixture_rel_path:
            raise FileNotFoundError(f"Artifact {artifact_key} unavailable for {fixture_id}")
        path = self.bundle_root / art.local_fixture_rel_path
        if not path.exists():
            raise FileNotFoundError(f"Local fixture file missing: {path}")
        if not self.verify_artifact_sha(fixture_id, artifact_key):
            raise ValueError(f"Fixture SHA-256 mismatch for {fixture_id} ({artifact_key})")
        
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask image from {path}")
        return mask


def get_default_fixture_store(root_override: Optional[Path] = None) -> GoldenFixtureStore:
    """Resolves the default fixture store based on environment or repo layout."""
    env_root = os.environ.get("COMICS_FIXTURE_ROOT")
    env_manifest = os.environ.get("COMICS_FIXTURE_MANIFEST")
    
    if root_override:
        bundle_root = root_override
    elif env_root:
        bundle_root = Path(env_root)
    else:
        bundle_root = Path(__file__).resolve().parent.parent.parent.parent / ".local_fixtures"
        
    if env_manifest:
        manifest_p = Path(env_manifest)
    else:
        manifest_p = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures" / "golden_fixture_manifest.json"
        
    return GoldenFixtureStore(bundle_root=bundle_root, manifest_path=manifest_p)
