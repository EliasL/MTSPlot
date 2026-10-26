"""Reuse managed configuration identities and strict current CSV reading."""

from dataclasses import dataclass
import json
from pathlib import Path

from MTS_Management.config import Config
from MTS_Management.data import read_macros


@dataclass(frozen=True)
class Run:
    folder: Path
    config: Config
    manifest: dict
    termination: dict | None

    @classmethod
    def read(cls, folder):
        folder = Path(folder).resolve()
        config = Config.read(folder / "config.conf", expected_name=folder.name)
        manifest = json.loads((folder / "manifest.json").read_text())
        if (manifest["config_hash"] != config.config_hash or
                manifest["settings"] != config.as_dict()):
            raise ValueError(f"Manifest/configuration mismatch: {folder}")
        path = folder / "termination.json"
        termination = json.loads(path.read_text())["termination"] if path.exists() else None
        return cls(folder, config, manifest, termination)

    def macros(self):
        return read_macros(self.folder / "macroData.csv")


def discover(root):
    """Read a run or immediate managed children; never recurse into large logs."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    folders = [root] if (root / "manifest.json").exists() else sorted(
        p.parent for p in root.glob("*/manifest.json"))
    runs = [Run.read(folder) for folder in folders]
    hashes = [r.config.config_hash for r in runs]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate configurations in data root; choose one run per configuration")
    return runs
