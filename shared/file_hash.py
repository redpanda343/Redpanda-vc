import hashlib
import json
import os
import threading

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sha256_cache.json")
_lock = threading.Lock()
_cache = None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entries():
    global _cache
    if _cache is None:
        try:
            with open(CACHE_PATH, encoding="utf-8") as file:
                _cache = json.load(file)
        except (OSError, ValueError):
            _cache = {}
        if not isinstance(_cache, dict):
            _cache = {}
    return _cache


def _save(entries):
    temporary = f"{CACHE_PATH}.{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(entries, file)
        os.replace(temporary, CACHE_PATH)
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass


def cached_sha256(path):
    path = os.path.abspath(path)
    status = os.stat(path)
    key = os.path.normcase(path)
    signature = [status.st_size, status.st_mtime_ns]
    with _lock:
        entry = _entries().get(key)
    if isinstance(entry, list) and entry[:2] == signature and len(entry) == 3:
        return entry[2]
    digest = sha256(path)
    with _lock:
        entries = _entries()
        entries[key] = [*signature, digest]
        _save(entries)
    return digest
