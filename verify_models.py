"""Fail a deployment build if the final model files are missing or altered."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "models/manifest.json").read_text())
for name, meta in manifest.items():
    path = root / "models" / f"{name}.onnx"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == meta["sha256"], f"Checksum mismatch: {name}"
    assert path.stat().st_size == meta["bytes"], f"Size mismatch: {name}"
    print(f"Verified {name}: {meta['bytes']} bytes")
