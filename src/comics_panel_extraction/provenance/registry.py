"""
Provenance registration and query utilities.

Provides lightweight metadata lookup for migrated components
without embedding heavy comments in runtime algorithm code.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ProvenanceRecord:
    """Provenance tracking record for a repository symbol."""
    symbol_name: str
    target_module: str
    legacy_file: str
    notebook_cell: Optional[int]
    legacy_symbol: str
    behavior_contract_id: str
    transformation_classification: str
    migration_status: str


class ProvenanceRegistry:
    """In-memory queryable registry of symbol provenance."""
    
    def __init__(self, records: List[ProvenanceRecord]) -> None:
        self._by_symbol = {r.symbol_name: r for r in records}
    
    def get(self, symbol_name: str) -> Optional[ProvenanceRecord]:
        return self._by_symbol.get(symbol_name)
    
    def list_all(self) -> List[ProvenanceRecord]:
        return list(self._by_symbol.values())
    
    @classmethod
    def load_from_json(cls, json_path: Path) -> "ProvenanceRegistry":
        if not json_path.exists():
            return cls([])
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [
            ProvenanceRecord(
                symbol_name=d["symbol_name"],
                target_module=d["target_module"],
                legacy_file=d["legacy_file"],
                notebook_cell=d.get("notebook_cell"),
                legacy_symbol=d["legacy_symbol"],
                behavior_contract_id=d["behavior_contract_id"],
                transformation_classification=d["transformation_classification"],
                migration_status=d["migration_status"],
            )
            for d in data
        ]
        return cls(records)
