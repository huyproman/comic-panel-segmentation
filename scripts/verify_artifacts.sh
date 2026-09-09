#!/usr/bin/env bash
# ==============================================================================
# verify_artifacts.sh
# Verifies presence and SHA-256 of external model artifacts.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${REPO_ROOT}/artifacts/artifact_manifest.json"

python3 -c "
import json, hashlib, sys
from pathlib import Path

repo_root = Path('${REPO_ROOT}').resolve()
manifest_path = Path('${MANIFEST}').resolve()

with open(manifest_path) as f:
    data = json.load(f)

for art in data.get('external_artifacts', []):
    rel_path = art['expected_local_storage_rel_path']
    full_path = repo_root / rel_path
    exp_sha = art['sha256']
    print(f'Checking artifact: {art[\"artifact_logical_id\"]} ({rel_path})')
    if not full_path.exists():
        print(f'  [MISSING] Artifact file not found at: {full_path}')
        print(f'  [ACTION] {art[\"acquisition_instructions\"]}')
        continue
    act_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
    if act_sha == exp_sha:
        print(f'  [OK] SHA-256 verified exact: {act_sha}')
    else:
        print(f'  [FAIL] SHA mismatch! Expected {exp_sha}, got {act_sha}')
        sys.exit(1)
"
