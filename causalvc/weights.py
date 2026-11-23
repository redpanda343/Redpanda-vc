"""Pretrained files the code needs, fetched on first use into weights/.

Each entry has a download address and, where known, the file's SHA-256. A file
already in weights/ is used as it is. CAUSALVC_WEIGHTS points somewhere else.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

from filelock import FileLock

ROOT = Path(os.environ.get("REDPANDA_DATA_ROOT") or Path(__file__).resolve().parent.parent).resolve()
FOLDER = Path(os.environ.get("CAUSALVC_WEIGHTS", (ROOT / "models" / "causalvc") if os.environ.get("REDPANDA_DATA_ROOT") else (ROOT / "weights")))

FILES = {
    # The pitch model used to label training audio, from the same place Applio fetches it.
    "rmvpe.pt": (
        "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/rmvpe.pt",
        "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
    ),
}


# The base a voice is built from: one zip holding these four files (checksums are of the files, not the zip).
BASE_URL = os.environ.get("CAUSALVC_BASE_URL", "https://huggingface.co/buckets/Mojobigbones/cvc/resolve/causalvc-base-v4-100hz.zip?download=true")
BASE_NAME = "causalvc-base"
BASE_FILES = {
    "last.pt": "e8f2fd8cf7fb61b68a454f73d1faea479d8a923aef7dd5242119a216d9ff6f61",
    "encoder.pt": "4b7d7fe78fe01a94648eaf3216e2a14004da8bd09caf7ae0c6661a3251c26693",
    "pitch.pt": "a4c3f1098cd8592233203129059c93857d8575407edc05faee5b47af0b999636",
    "args.json": "e39434d48669e6d63a659214df69be7a3f2378c17163e8e3b85dadc6d5f655b2",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, label: str) -> None:
    """Fetch url to dest, through a .part file so an interrupted download is never mistaken for a whole one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {label} from {url}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "causalvc"})) as r, open(part, "wb") as f:
        total, got, shown = int(r.headers.get("Content-Length") or 0), 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            pct = 100 * got // total if total else 0
            if pct // 10 != shown and total:
                shown = pct // 10
                print(f"  {pct}% of {total / 1e6:.0f} MB", flush=True)
    part.replace(dest)


def ensure(name: str) -> Path:
    """Path of a pretrained file, downloading it first if it is not there yet."""
    path = FOLDER / name
    if path.is_file():
        return path
    url, digest = FILES[name]
    rvc_pitch = ROOT / "models" / "predictors" / "rmvpe_rvc.pt"
    if name == "rmvpe.pt" and rvc_pitch.is_file() and sha256(rvc_pitch) == digest:
        return rvc_pitch
    download(url, path, name)
    if digest and sha256(path) != digest:
        path.unlink()
        sys.exit(f"{name}: the downloaded file does not match its expected checksum; nothing was kept")
    return path


def base_folder() -> Path:
    return FOLDER / BASE_NAME


def ensure_base() -> Path:
    """Folder holding the base (last.pt, args.json, encoder.pt, pitch.pt), downloaded and unpacked on first use."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    with FileLock(FOLDER / ".base.lock"):
        return _ensure_base()


def _ensure_base() -> Path:
    folder = base_folder()
    if all((folder / name).is_file() for name in BASE_FILES) and (
        not BASE_URL or sha256(folder / "args.json") == BASE_FILES["args.json"]
    ):
        return folder
    if not BASE_URL:
        sys.exit(f"the base is not in {folder} and no download address is set (CAUSALVC_BASE_URL)")
    archive = FOLDER / f"{BASE_NAME}.zip"
    download(BASE_URL, archive, f"{BASE_NAME}.zip")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Take only the expected files, by name, wherever they sit in the archive; nothing else is written.
        members = {Path(m.filename).name: m for m in z.infolist() if not m.is_dir()}
        missing = [name for name in BASE_FILES if name not in members]
        if missing:
            sys.exit(f"{archive.name} does not contain {', '.join(missing)}")
        for name, digest in BASE_FILES.items():
            print(f"unpacking {name}", flush=True)
            with z.open(members[name]) as src, open(folder / name, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            if sha256(folder / name) != digest:
                (folder / name).unlink()
                sys.exit(f"{name} in {archive.name} does not match its expected checksum; it was not kept")
    archive.unlink()
    return folder


if __name__ == "__main__":  # python -m causalvc.weights : fetch everything now (setup)
    for name in FILES:
        print(ensure(name))
    print(ensure_base())
