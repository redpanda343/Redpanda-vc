"""Pure-NumPy NWC dataset loader, bucket sampler, and threaded prefetcher for JAX training.

This module intentionally avoids importing `jax` or `torch` so worker threads/subprocesses
never touch TPU/PJRT or GPU contexts.
"""

from __future__ import annotations

import ctypes
import io
import json
import math
import os
import queue
import re
import shutil
import struct
import sys
import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator, Mapping, Sequence

try:
    import resource
except ImportError:
    resource = None

import numpy as np
import soundfile as sf

from rvc.train.backend import effective_cpu_count


def _available_ram_bytes() -> int:
    """Return available host physical memory in bytes for bounded dataset sample caching."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass
    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
    return 4 * (1024 ** 3)


def _sample_nbytes(item: tuple[Any, ...]) -> int:
    return sum(x.nbytes for x in item if isinstance(x, np.ndarray))


_NPY_DESCR_RE = re.compile(r"['\"]descr['\"]\s*:\s*(['\"])([^'\"]+)\1")
_NPY_FORTRAN_RE = re.compile(r"['\"]fortran_order['\"]\s*:\s*(True|False)")
_NPY_SHAPE_RE = re.compile(r"['\"]shape['\"]\s*:\s*\(([^)]*)\)")


def _parse_npy_fileobj(f, max_rows: int | None = None, *, raw_buf: bytes | None = None) -> np.ndarray:
    """Parse C-contiguous .npy header from binary stream and read up to `max_rows`."""
    prefix = f.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
        raise ValueError("Not an NPY file")
    major = prefix[6]
    if major == 1:
        hlen = int.from_bytes(f.read(2), "little")
    elif major == 2:
        hlen = int.from_bytes(f.read(4), "little")
    else:
        raise ValueError(f"Unsupported NPY version {major}")
    hdr = f.read(hlen).decode("latin1")
    m_descr = _NPY_DESCR_RE.search(hdr)
    m_fort = _NPY_FORTRAN_RE.search(hdr)
    m_shape = _NPY_SHAPE_RE.search(hdr)
    if not (m_descr and m_fort and m_shape) or m_fort.group(1) != "False":
        raise ValueError("Non-standard NPY header")
    dtype = np.dtype(m_descr.group(2))
    if dtype.hasobject:
        raise ValueError("Object arrays require np.load validation")
    dims = [int(tok.strip()) for tok in m_shape.group(1).split(",") if tok.strip()]
    if not dims:
        raise ValueError("Empty scalar shape")
    rows = min(dims[0], int(max_rows)) if max_rows is not None else dims[0]
    tail_shape = tuple(dims[1:])
    row_stride = math.prod(tail_shape) if tail_shape else 1
    count = rows * row_stride
    if raw_buf is not None:
        arr = np.frombuffer(raw_buf, dtype=dtype, count=count, offset=f.tell()).copy()
    else:
        arr = np.fromfile(f, dtype=dtype, count=count)
    if arr.size != count:
        raise ValueError("Short NPY read")
    return arr.reshape((rows,) + tail_shape)


def _fast_load_npy(path: str, max_rows: int | None = None) -> np.ndarray:
    """Parse C-contiguous .npy header directly and read up to `max_rows` without ast.literal_eval."""
    try:
        with open(path, "rb") as f:
            return _parse_npy_fileobj(f, max_rows=max_rows)
    except Exception:
        return np.load(path, allow_pickle=False)


def _fast_load_npy_bytes(raw: bytes, max_rows: int | None = None) -> np.ndarray:
    """Parse C-contiguous .npy payload directly from an in-memory buffer."""
    try:
        return _parse_npy_fileobj(io.BytesIO(raw), max_rows=max_rows, raw_buf=raw)
    except Exception:
        return np.load(io.BytesIO(raw), allow_pickle=False)


_DEFAULT_SF_READ = sf.read
_PCM16_INV = np.float32(1.0 / 32768.0)


def _parse_wav_fileobj(
    f,
    max_samples: int | None = None,
    *,
    allow_int16: bool = False,
) -> tuple[np.ndarray, int]:
    """Parse uncompressed mono PCM-16/float32 WAV header and data from a binary stream."""
    head = f.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("Not RIFF/WAVE")
    fmt_code = ch = sr = bits = 0
    for _ in range(8):
        chdr = f.read(8)
        if len(chdr) < 8:
            break
        cid = chdr[:4]
        csize = int.from_bytes(chdr[4:8], "little")
        if cid == b"fmt ":
            if csize < 16 or csize > 4096:
                raise ValueError("Invalid fmt chunk")
            fbody = f.read(csize)
            if len(fbody) < csize:
                raise ValueError("Short fmt chunk")
            if csize & 1:
                f.read(1)
            fmt_code, ch, sr, _, _, bits = struct.unpack("<HHIIHH", fbody[:16])
            if ch != 1 or not ((fmt_code == 1 and bits == 16) or (fmt_code == 3 and bits == 32)):
                raise ValueError("Unsupported WAV encoding for fast path")
        elif cid == b"data":
            if sr <= 0 or bits <= 0:
                raise ValueError("Missing fmt before data")
            bw = bits // 8
            n_avail = csize // bw
            n_read = min(n_avail, int(max_samples)) if max_samples is not None else n_avail
            n_bytes = n_read * bw
            raw = f.read(n_bytes)
            if len(raw) != n_bytes:
                raise ValueError("Short WAV data")
            if fmt_code == 1:
                pcm = np.frombuffer(raw, dtype="<i2")
                if allow_int16:
                    return np.array(pcm, dtype=np.int16, copy=True), int(sr)
                audio = pcm.astype(np.float32)
                audio *= _PCM16_INV
            else:
                audio = np.array(np.frombuffer(raw, dtype="<f4"), dtype=np.float32, copy=True)
            return audio, int(sr)
        else:
            if csize < 0:
                raise ValueError("Invalid chunk size")
            f.seek(csize + (csize & 1), os.SEEK_CUR)
    raise ValueError("Missing WAV data chunk")


def _fast_load_wav(
    path: str,
    max_samples: int | None = None,
    *,
    allow_int16: bool = False,
) -> tuple[np.ndarray, int]:
    """Read uncompressed mono PCM-16/float32 WAV directly without libsndfile seek overhead."""
    if sf.read is not _DEFAULT_SF_READ or not path.lower().endswith(".wav"):
        return sf.read(path, dtype="float32")
    try:
        with open(path, "rb") as f:
            return _parse_wav_fileobj(f, max_samples=max_samples, allow_int16=allow_int16)
    except Exception:
        pass
    return sf.read(path, dtype="float32")


def _fast_load_wav_bytes(
    raw: bytes,
    path_hint: str,
    max_samples: int | None = None,
    *,
    allow_int16: bool = False,
) -> tuple[np.ndarray, int]:
    """Decode WAV/audio payload directly from in-memory bytes."""
    if sf.read is _DEFAULT_SF_READ and path_hint.lower().endswith(".wav"):
        try:
            return _parse_wav_fileobj(io.BytesIO(raw), max_samples=max_samples, allow_int16=allow_int16)
        except Exception:
            pass
    if sf.read is _DEFAULT_SF_READ:
        try:
            return _DEFAULT_SF_READ(io.BytesIO(raw), dtype="float32")
        except Exception:
            pass
    return sf.read(path_hint, dtype="float32")


_SHARD_DIR_SUFFIX = ".shards"
_SHARD_MANIFEST_NAME = "manifest.json"
_SHARD_PLAN_NAME = ".plan.json"
_SHARD_INDEX_NAME = "index.npy"
_RECORD_MAGIC = b"RVC2"
_RECORD_HEADER_STRUCT = struct.Struct("<4sIIIIIIi")
_FLAG_WAV_INT16 = 1
_DEFAULT_CLIPS_PER_SHARD = 512


def _max_open_shard_fds() -> int:
    if resource is not None and os.name == "posix":
        try:
            soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft and soft > 0:
                return min(128, max(16, int(soft) // 4))
        except Exception:
            pass
    return 64 if os.name == "nt" else 128


def _align_clip_arrays(
    audio: np.ndarray,
    sr: int,
    phone_raw: np.ndarray,
    pitch: np.ndarray,
    pitchf: np.ndarray,
    dv_str: str,
    *,
    expected_sr: int,
    hop_length: int,
    max_sample_frames: int,
    audiopath: str,
) -> tuple[int, np.ndarray, bool, np.ndarray, np.ndarray, np.ndarray, int]:
    """Align and trim loaded waveform, phoneme, and F0 arrays using canonical dataset rules."""
    n_num = min(int(phone_raw.shape[0]) * 2, int(pitch.shape[0]), int(pitchf.shape[0]), max_sample_frames)
    n_phone_half = (n_num + 1) // 2
    phone_half = np.asarray(phone_raw[:n_phone_half, :], dtype=np.float32)
    pitch_arr = np.asarray(pitch[:n_num], dtype=np.int32)
    pitchf_arr = np.asarray(pitchf[:n_num], dtype=np.float32)
    try:
        sid = int(dv_str)
    except ValueError:
        sid = 0

    if audio.ndim > 1:
        audio = audio[:, 0]
    if int(sr) != expected_sr:
        raise ValueError(f"{sr} SR doesn't match target {expected_sr} SR in '{audiopath}'.")

    n_frames_audio = min(len(audio) // hop_length, max_sample_frames)
    len_min = min(n_num, n_frames_audio)
    len_wav = len_min * hop_length
    if len_min < n_num:
        phone_half = phone_half[: (len_min + 1) // 2, :]
        pitch_arr = pitch_arr[:len_min]
        pitchf_arr = pitchf_arr[:len_min]

    wav_is_int16 = audio.dtype == np.int16
    wav_1d = np.ascontiguousarray(audio[:len_wav], dtype=np.int16 if wav_is_int16 else np.float32)
    phone_out = np.ascontiguousarray(phone_half, dtype="<f4")
    pitch_out = np.ascontiguousarray(pitch_arr, dtype="<i4")
    pitchf_out = np.ascontiguousarray(pitchf_arr, dtype="<f4")
    return len_min, wav_1d, wav_is_int16, phone_out, pitch_out, pitchf_out, sid


def _encode_clip_record(
    len_min: int,
    wav_1d: np.ndarray,
    wav_is_int16: bool,
    phone_half: np.ndarray,
    pitch: np.ndarray,
    pitchf: np.ndarray,
    sid: int,
    sr: int,
) -> bytes:
    """Serialize one trimmed clip into a self-describing binary record."""
    phone_frames = int(phone_half.shape[0])
    phone_dim = int(phone_half.shape[1]) if phone_half.ndim == 2 else 0
    wav_le = np.ascontiguousarray(wav_1d, dtype="<i2" if wav_is_int16 else "<f4")
    header = _RECORD_HEADER_STRUCT.pack(
        _RECORD_MAGIC,
        int(len_min),
        int(wav_le.shape[0]),
        phone_frames,
        phone_dim,
        _FLAG_WAV_INT16 if wav_is_int16 else 0,
        int(sr),
        int(sid),
    )
    return b"".join((header, wav_le.tobytes(), phone_half.tobytes(), pitch.tobytes(), pitchf.tobytes()))


def _decode_clip_record(
    raw: bytes,
    *,
    expected_sr: int,
    repeat_phone: bool,
    phone_dtype: np.dtype,
    compute_spec: bool,
    allow_pcm16: bool,
    filter_length: int,
    hop_length: int,
    win_length: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
    """Deserialize one clip record into owned NumPy arrays matching TextAudioDataset.__getitem__."""
    if len(raw) < _RECORD_HEADER_STRUCT.size:
        raise ValueError("Truncated shard record header")
    magic, len_min, wav_len, phone_frames, phone_dim, flags, sr, sid = _RECORD_HEADER_STRUCT.unpack_from(raw, 0)
    if magic != _RECORD_MAGIC:
        raise ValueError("Invalid shard record magic")
    if int(sr) != int(expected_sr):
        raise ValueError(f"{sr} SR doesn't match target {expected_sr} SR in shard record.")

    wav_is_int16 = bool(flags & _FLAG_WAV_INT16)
    wav_dt = np.dtype("<i2") if wav_is_int16 else np.dtype("<f4")
    pos = _RECORD_HEADER_STRUCT.size
    wav_end = pos + wav_len * wav_dt.itemsize
    phone_end = wav_end + phone_frames * phone_dim * 4
    pitch_end = phone_end + len_min * 4
    pitchf_end = pitch_end + len_min * 4
    if pitchf_end != len(raw):
        raise ValueError(f"Shard record size mismatch: expected {pitchf_end}, got {len(raw)}")

    mv = memoryview(raw)
    wav_1d = np.frombuffer(mv[pos:wav_end], dtype=wav_dt)
    if wav_is_int16:
        if allow_pcm16 and not compute_spec:
            wav_out = np.array(wav_1d, dtype=np.int16, copy=True)[:, None]
        else:
            wav_out = (np.array(wav_1d, dtype=np.float32, copy=True) * _PCM16_INV)[:, None]
    else:
        wav_out = np.array(wav_1d, dtype=np.float32, copy=True)[:, None]

    phone_use = np.array(
        np.frombuffer(mv[wav_end:phone_end], dtype="<f4").reshape(phone_frames, phone_dim),
        dtype=phone_dtype,
        copy=True,
    )
    phone = np.repeat(phone_use, 2, axis=0)[:len_min, :] if repeat_phone else phone_use
    pitch = np.array(np.frombuffer(mv[phone_end:pitch_end], dtype="<i4"), dtype=np.int32, copy=True)
    pitchf = np.array(np.frombuffer(mv[pitch_end:pitchf_end], dtype="<f4"), dtype=np.float32, copy=True)
    spec = (
        spectrogram_np(np.ascontiguousarray(wav_out[:, 0], dtype=np.float32), filter_length, hop_length, win_length)
        if compute_spec
        else None
    )
    return spec, wav_out, phone, pitch, pitchf, np.int32(sid)


def _posix_pread_exact(fd: int, offset: int, nbytes: int) -> bytes:
    out = bytearray()
    while len(out) < nbytes:
        chunk = os.pread(fd, nbytes - len(out), offset + len(out))
        if not chunk:
            raise EOFError("Unexpected EOF during shard pread")
        out.extend(chunk)
    return bytes(out)


class _DatasetShardReader:
    """Thread-safe LRU file-descriptor reader for contiguous `.rpack` dataset shards."""

    def __init__(self, shards_dir: str, index_arr: np.ndarray, expected_sr: int) -> None:
        self.shards_dir = shards_dir
        self.index = index_arr
        self.expected_sr = int(expected_sr)
        self._max_fds = _max_open_shard_fds()
        self._fd_lock = threading.Lock()
        self._fds: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._has_pread = hasattr(os, "pread")

    def _open_flags(self) -> int:
        return os.O_RDONLY | getattr(os, "O_BINARY", 0)

    def _acquire_entry(self, shard_id: int) -> dict[str, Any]:
        with self._fd_lock:
            entry = self._fds.get(shard_id)
            if entry is not None and not entry["stale"]:
                entry["ref"] += 1
                self._fds.move_to_end(shard_id)
                return entry

        shard_path = os.path.join(self.shards_dir, f"shard_{shard_id:05d}.rpack")
        new_fd = os.open(shard_path, self._open_flags())
        to_close: list[int] = []
        with self._fd_lock:
            existing = self._fds.get(shard_id)
            if existing is not None and not existing["stale"]:
                existing["ref"] += 1
                self._fds.move_to_end(shard_id)
                to_close.append(new_fd)
                entry = existing
            else:
                entry = {"fd": new_fd, "ref": 1, "stale": False, "lock": threading.Lock()}
                self._fds[shard_id] = entry
                if len(self._fds) > self._max_fds:
                    for old_sid in list(self._fds.keys()):
                        if len(self._fds) <= self._max_fds:
                            break
                        if old_sid == shard_id:
                            continue
                        old_entry = self._fds[old_sid]
                        if old_entry["ref"] == 0:
                            self._fds.pop(old_sid)
                            old_entry["stale"] = True
                            to_close.append(old_entry["fd"])
        for fd_c in to_close:
            try:
                os.close(fd_c)
            except OSError:
                pass
        return entry

    def _release_entry(self, entry: dict[str, Any]) -> None:
        close_fd: int | None = None
        with self._fd_lock:
            entry["ref"] -= 1
            if entry["stale"] and entry["ref"] <= 0:
                close_fd = entry["fd"]
                entry["fd"] = -1
        if close_fd is not None and close_fd >= 0:
            try:
                os.close(close_fd)
            except OSError:
                pass

    def pread_clip(self, index: int) -> bytes:
        shard_id, offset, nbytes = (int(v) for v in self.index[index])
        entry = self._acquire_entry(shard_id)
        try:
            if self._has_pread:
                return _posix_pread_exact(entry["fd"], offset, nbytes)
            with entry["lock"]:
                os.lseek(entry["fd"], offset, os.SEEK_SET)
                out = bytearray()
                while len(out) < nbytes:
                    chunk = os.read(entry["fd"], nbytes - len(out))
                    if not chunk:
                        raise EOFError("Unexpected EOF during shard read")
                    out.extend(chunk)
                return bytes(out)
        finally:
            self._release_entry(entry)

    def close(self) -> None:
        to_close: list[int] = []
        with self._fd_lock:
            for entry in self._fds.values():
                entry["stale"] = True
                if entry["ref"] <= 0 and entry["fd"] >= 0:
                    to_close.append(entry["fd"])
                    entry["fd"] = -1
            self._fds.clear()
        for fd_c in to_close:
            try:
                os.close(fd_c)
            except OSError:
                pass


def _shard_dir_for_filelist(training_files: str) -> str:
    return f"{training_files}{_SHARD_DIR_SUFFIX}"


def _build_dataset_fingerprint(dataset: "TextAudioDataset") -> dict[str, Any] | None:
    tf = getattr(dataset, "training_files", "")
    if not tf or not os.path.isfile(tf):
        return None
    try:
        st = os.stat(tf)
    except OSError:
        return None
    return {
        "version": 1,
        "filelist_mtime_ns": int(st.st_mtime_ns),
        "filelist_size": int(st.st_size),
        "count": len(dataset.audiopaths_and_text),
        "sample_rate": int(dataset.sample_rate),
        "hop_length": int(dataset.hop_length),
        "max_sample_frames": int(dataset.max_sample_frames),
    }


def _meta_matches_fingerprint(meta: Mapping[str, Any], fp: Mapping[str, Any]) -> bool:
    return all(meta.get(k) == v for k, v in fp.items())


def _open_validated_shard_reader(dataset: "TextAudioDataset") -> _DatasetShardReader | None:
    fp = _build_dataset_fingerprint(dataset)
    if fp is None or fp["count"] <= 0:
        return None
    shards_dir = _shard_dir_for_filelist(dataset.training_files)
    manifest_path = os.path.join(shards_dir, _SHARD_MANIFEST_NAME)
    index_path = os.path.join(shards_dir, _SHARD_INDEX_NAME)
    if not (os.path.isfile(manifest_path) and os.path.isfile(index_path)):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not _meta_matches_fingerprint(manifest, fp):
            return None
        num_shards = int(manifest.get("num_shards", 0))
        index_arr = np.load(index_path, allow_pickle=False)
        if index_arr.ndim != 2 or index_arr.shape != (fp["count"], 3):
            return None
        if num_shards <= 0:
            return None
        for sid in range(num_shards):
            sp = os.path.join(shards_dir, f"shard_{sid:05d}.rpack")
            if not os.path.isfile(sp):
                return None
        # Spot-check shard sizes cleanly without stat loops if consistent
        shard_ids = index_arr[:, 0]
        ends = index_arr[:, 1] + index_arr[:, 2]
        for sid in range(num_shards):
            mask = shard_ids == sid
            if np.any(mask):
                need_size = int(ends[mask].max())
                actual_size = os.path.getsize(os.path.join(shards_dir, f"shard_{sid:05d}.rpack"))
                if actual_size < need_size:
                    return None
        return _DatasetShardReader(shards_dir, index_arr, expected_sr=fp["sample_rate"])
    except Exception:
        return None


def _atomic_write_json(path: str, payload: Any) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


_SYNC_FILE_RANGE_WRITE = 2
_SYNC_FILE_RANGE_FN: Any = False  # Lazy sentinel: False=uninitialized, None=unavailable


def _get_sync_file_range():
    """Return Linux 64-bit libc sync_file_range(fd, offset, nbytes, flags) or None."""
    global _SYNC_FILE_RANGE_FN
    if _SYNC_FILE_RANGE_FN is not False:
        return _SYNC_FILE_RANGE_FN
    if not sys.platform.startswith("linux") or ctypes.sizeof(ctypes.c_void_p) != 8:
        _SYNC_FILE_RANGE_FN = None
        return None
    try:
        fn = ctypes.CDLL(None).sync_file_range
        fn.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint]
        fn.restype = ctypes.c_int
        _SYNC_FILE_RANGE_FN = fn
    except Exception:
        _SYNC_FILE_RANGE_FN = None
    return _SYNC_FILE_RANGE_FN


def _fadvise_dontneed(fd: int) -> None:
    """Hint Linux to drop one-shot file pages from cache cleanly to avoid memory bloat."""
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


class _DirFdCache:
    """Thread-safe cache of open directory file descriptors for fast dir_fd openat()."""

    def __init__(self) -> None:
        self._supported = os.open in getattr(os, "supports_dir_fd", set())
        self._lock = threading.Lock()
        self._fds: dict[str, int] = {}

    def open_read(self, path: str) -> tuple[int, bool]:
        """Return (fd, is_regular_open); falls back to full path when dir_fd is unsupported."""
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if not self._supported:
            return os.open(path, flags), False
        d, fn = os.path.split(path)
        if not fn:
            return os.open(path, flags), False
        d_key = d or "."
        dfd = self._fds.get(d_key)
        if dfd is None:
            new_dfd = os.open(d_key, os.O_RDONLY)
            with self._lock:
                dfd = self._fds.get(d_key)
                if dfd is None:
                    self._fds[d_key] = new_dfd
                    dfd = new_dfd
                else:
                    try:
                        os.close(new_dfd)
                    except OSError:
                        pass
        return os.open(fn, flags, dir_fd=dfd), True

    def close(self) -> None:
        with self._lock:
            for dfd in self._fds.values():
                try:
                    os.close(dfd)
                except OSError:
                    pass
            self._fds.clear()


def _read_file_nocache(path: str, dir_fds: _DirFdCache | None) -> bytes:
    """Read file payload with optional dir_fd openat() and drop OS page cache immediately."""
    if dir_fds is not None:
        fd, _ = dir_fds.open_read(path)
    else:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        with os.fdopen(fd, "rb", buffering=0, closefd=False) as rf:
            data = rf.read()
        _fadvise_dontneed(fd)
        return data
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _encode_clip_to_record_and_item(
    row: Sequence[str],
    *,
    dataset: "TextAudioDataset",
    want_seed_item: bool,
    dir_fds: _DirFdCache | None = None,
) -> tuple[bytes, tuple[Any, ...] | None]:
    audiopath, phone_path, pitch_path, pitchf_path, dv_str = row[:5]
    max_frames = int(dataset.max_sample_frames)
    max_phone_rows = (max_frames + 1) // 2 + 1
    if dir_fds is not None and sf.read is _DEFAULT_SF_READ:
        phone_raw = _fast_load_npy_bytes(_read_file_nocache(phone_path, dir_fds), max_rows=max_phone_rows)
        pitch = _fast_load_npy_bytes(_read_file_nocache(pitch_path, dir_fds), max_rows=max_frames)
        pitchf = _fast_load_npy_bytes(_read_file_nocache(pitchf_path, dir_fds), max_rows=max_frames)
        n_est = min(int(phone_raw.shape[0]) * 2, int(pitch.shape[0]), int(pitchf.shape[0]), max_frames)
        audio, sr = _fast_load_wav_bytes(
            _read_file_nocache(audiopath, dir_fds),
            audiopath,
            max_samples=n_est * int(dataset.hop_length),
            allow_int16=True,
        )
    else:
        phone_raw = _fast_load_npy(phone_path, max_rows=max_phone_rows)
        pitch = _fast_load_npy(pitch_path, max_rows=max_frames)
        pitchf = _fast_load_npy(pitchf_path, max_rows=max_frames)
        n_est = min(int(phone_raw.shape[0]) * 2, int(pitch.shape[0]), int(pitchf.shape[0]), max_frames)
        audio, sr = _fast_load_wav(
            audiopath,
            max_samples=n_est * int(dataset.hop_length),
            allow_int16=True,
        )

    len_min, wav_1d, wav_is_int16, phone_out, pitch_out, pitchf_out, sid = _align_clip_arrays(
        audio,
        sr,
        phone_raw,
        pitch,
        pitchf,
        dv_str,
        expected_sr=int(dataset.sample_rate),
        hop_length=int(dataset.hop_length),
        max_sample_frames=max_frames,
        audiopath=audiopath,
    )
    rec = _encode_clip_record(len_min, wav_1d, wav_is_int16, phone_out, pitch_out, pitchf_out, sid, int(sr))
    seed_item: tuple[Any, ...] | None = None
    if want_seed_item:
        if wav_is_int16 and not dataset.compute_spec:
            wav_col = np.ascontiguousarray(wav_1d, dtype=np.int16)[:, None]
        elif wav_is_int16:
            wav_col = np.ascontiguousarray(wav_1d.astype(np.float32) * _PCM16_INV)[:, None]
        else:
            wav_col = np.ascontiguousarray(wav_1d, dtype=np.float32)[:, None]
        ph_typed = np.asarray(phone_out, dtype=dataset.phone_dtype)
        ph_item = np.repeat(ph_typed, 2, axis=0)[:len_min, :] if dataset.repeat_phone else np.ascontiguousarray(ph_typed)
        pitch_item = np.ascontiguousarray(pitch_out, dtype=np.int32)
        pitchf_item = np.ascontiguousarray(pitchf_out, dtype=np.float32)
        spec_item = (
            spectrogram_np(
                np.ascontiguousarray(wav_col[:, 0], dtype=np.float32),
                dataset.filter_length,
                dataset.hop_length,
                dataset.win_length,
            )
            if dataset.compute_spec
            else None
        )
        seed_item = (spec_item, wav_col, ph_item, pitch_item, pitchf_item, np.int32(sid))
    return rec, seed_item


def _read_shard_records(
    shard_id: int,
    clip_indices: Sequence[int],
    *,
    dataset: "TextAudioDataset",
    seed_state: dict[str, Any] | None,
    max_seed_bytes: int,
    dir_fds: _DirFdCache | None = None,
    on_clip_done: Any = None,
) -> tuple[int, list[int], list[bytes], list[tuple[int, tuple[Any, ...]]]]:
    """Read and encode one sorted clip chunk via dir_fd openat() without OS cache pollution."""
    records: list[bytes] = []
    seeded: list[tuple[int, tuple[Any, ...]]] = []

    for gidx in clip_indices:
        gidx_i = int(gidx)
        want_seed = False
        if seed_state is not None and max_seed_bytes > 0:
            with seed_state["lock"]:
                want_seed = seed_state["bytes"] < max_seed_bytes
        rec, s_item = _encode_clip_to_record_and_item(
            dataset.audiopaths_and_text[gidx_i],
            dataset=dataset,
            want_seed_item=want_seed,
            dir_fds=dir_fds,
        )
        records.append(rec)
        if s_item is not None and seed_state is not None and max_seed_bytes > 0:
            n_b = _sample_nbytes(s_item)
            with seed_state["lock"]:
                if seed_state["bytes"] + n_b <= max_seed_bytes:
                    seed_state["bytes"] += n_b
                    seeded.append((gidx_i, s_item))
        if on_clip_done is not None:
            on_clip_done(1)

    return shard_id, [int(i) for i in clip_indices], records, seeded


def ensure_dataset_shards(
    dataset: "TextAudioDataset",
    *,
    workers: int = 4,
    progress: bool = True,
    max_seed_bytes: int = 0,
    clips_per_shard: int = _DEFAULT_CLIPS_PER_SHARD,
) -> tuple[_DatasetShardReader | None, dict[int, tuple[Any, ...]]]:
    """Ensure contiguous `.rpack` dataset shards exist for fast Epoch 1 I/O without concurrent training contention."""
    fp = _build_dataset_fingerprint(dataset)
    if fp is None or fp["count"] <= 0:
        return None, {}

    existing = _open_validated_shard_reader(dataset)
    if existing is not None:
        dataset._shard_reader = existing
        return existing, {}

    shards_dir = _shard_dir_for_filelist(dataset.training_files)
    os.makedirs(shards_dir, exist_ok=True)
    # Remove orphaned temp files from any interrupted run
    for fname in os.listdir(shards_dir):
        if fname.endswith(".tmp"):
            try:
                os.remove(os.path.join(shards_dir, fname))
            except OSError:
                pass

    n_clips = fp["count"]
    per_shard = max(1, min(int(clips_per_shard), n_clips))
    plan_fp = dict(fp)
    plan_fp["clips_per_shard"] = per_shard
    plan_path = os.path.join(shards_dir, _SHARD_PLAN_NAME)
    plan_valid = False
    if os.path.isfile(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as pf:
                old_plan = json.load(pf)
            plan_valid = _meta_matches_fingerprint(old_plan, plan_fp)
        except Exception:
            plan_valid = False
    if not plan_valid:
        for fname in os.listdir(shards_dir):
            if fname.endswith((".rpack", ".idx.npy", ".npy", ".json")):
                try:
                    os.remove(os.path.join(shards_dir, fname))
                except OSError:
                    pass
        _atomic_write_json(plan_path, plan_fp)

    # Check free disk space cleanly before writing (~1.05x estimate + 4 GiB headroom for large datasets)
    try:
        sample_k = min(32, n_clips)
        sample_bytes = 0
        for si in range(sample_k):
            for p in dataset.audiopaths_and_text[si][:4]:
                if os.path.isfile(p):
                    sample_bytes += os.path.getsize(p)
        est_bytes = int((sample_bytes / max(1, sample_k)) * n_clips * 1.05)
        free_bytes = shutil.disk_usage(shards_dir).free
        headroom = 4 * (1024 ** 3) if n_clips >= 512 else 0
        if free_bytes < est_bytes + headroom:
            print(
                f"Skipping dataset shard pack: need ~{(est_bytes + headroom)/(1024**3):.1f} GiB free, only {free_bytes/(1024**3):.1f} GiB free.",
                flush=True,
            )
            return None, {}
    except OSError:
        pass

    sorted_indices = sorted(
        range(n_clips),
        key=lambda i: (
            os.path.dirname(os.path.abspath(dataset.audiopaths_and_text[i][0])),
            os.path.basename(dataset.audiopaths_and_text[i][0]),
            dataset.audiopaths_and_text[i][0],
        ),
    )
    shard_groups = [sorted_indices[s : s + per_shard] for s in range(0, n_clips, per_shard)]
    num_shards = len(shard_groups)

    missing_shards: list[int] = []
    done_clips = 0
    for sid, idxs in enumerate(shard_groups):
        rpack_p = os.path.join(shards_dir, f"shard_{sid:05d}.rpack")
        idx_p = os.path.join(shards_dir, f"shard_{sid:05d}.idx.npy")
        ok = False
        if os.path.isfile(rpack_p) and os.path.isfile(idx_p):
            try:
                s_idx = np.load(idx_p, allow_pickle=False)
                if s_idx.shape == (len(idxs), 3) and np.array_equal(s_idx[:, 0], np.asarray(idxs, dtype=np.int64)):
                    need_sz = int((s_idx[:, 1] + s_idx[:, 2]).max()) if len(idxs) > 0 else 0
                    if os.path.getsize(rpack_p) == need_sz:
                        ok = True
            except Exception:
                ok = False
        if ok:
            done_clips += len(idxs)
        else:
            missing_shards.append(sid)

    seed_items: dict[int, tuple[Any, ...]] = {}
    if missing_shards:
        avail = _available_ram_bytes()
        # Keep CPU, RAM, and GCE disk gentle: 4-6 dir_fd openat() threads + paced background writeback.
        sub_w = min(6, max(4, int(workers))) if avail >= 8 * (1024 ** 3) else min(2, max(1, int(workers)))
        dir_fds = _DirFdCache()
        sync_range_fn = _get_sync_file_range()
        seed_state = {"bytes": 0, "lock": threading.Lock()}
        pbar = None
        pbar_lock = threading.Lock()
        if progress:
            print(
                f"Packing {n_clips} dataset clips into {num_shards} contiguous shard(s) ({sub_w} low-overhead thread(s); one-time per dataset)...",
                flush=True,
            )
            try:
                from tqdm import tqdm

                pbar = tqdm(total=n_clips, initial=done_clips, unit="clip", desc="Shard Pack", leave=True)
            except Exception:
                pbar = None

        def _tick(n: int) -> None:
            if pbar is not None:
                with pbar_lock:
                    pbar.update(n)

        write_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        write_stop = threading.Event()
        writer_exc: list[BaseException] = []
        writer_sentinel = object()
        _SYNC_CHUNK = 16 * 1024 * 1024

        def _writer_loop() -> None:
            try:
                while True:
                    item = write_queue.get()
                    try:
                        if item is writer_sentinel:
                            return
                        w_sid, w_clip_idxs, w_rec_list = item
                        rpack_p = os.path.join(shards_dir, f"shard_{w_sid:05d}.rpack")
                        idx_p = os.path.join(shards_dir, f"shard_{w_sid:05d}.idx.npy")
                        rpack_tmp = f"{rpack_p}.{os.getpid()}.tmp"
                        idx_tmp = f"{idx_p}.{os.getpid()}.tmp"
                        s_idx = np.empty((len(w_clip_idxs), 3), dtype=np.int64)
                        off = 0
                        flushed_off = 0
                        with open(rpack_tmp, "wb", buffering=4 * 1024 * 1024) as wf:
                            w_fd = wf.fileno()
                            for li, (gidx, rec_b) in enumerate(zip(w_clip_idxs, w_rec_list)):
                                s_idx[li, 0] = gidx
                                s_idx[li, 1] = off
                                s_idx[li, 2] = len(rec_b)
                                wf.write(rec_b)
                                off += len(rec_b)
                                if sync_range_fn is not None and (off - flushed_off) >= _SYNC_CHUNK:
                                    wf.flush()
                                    sync_range_fn(w_fd, flushed_off, off - flushed_off, _SYNC_FILE_RANGE_WRITE)
                                    flushed_off = off
                            wf.flush()
                            if hasattr(os, "fdatasync"):
                                os.fdatasync(w_fd)
                            else:
                                os.fsync(w_fd)
                            _fadvise_dontneed(w_fd)
                        w_rec_list.clear()
                        with open(idx_tmp, "wb") as ix_f:
                            np.save(ix_f, s_idx)
                        os.replace(idx_tmp, idx_p)
                        os.replace(rpack_tmp, rpack_p)
                    finally:
                        write_queue.task_done()
            except BaseException as exc:
                writer_exc.append(exc)
                write_stop.set()

        writer_thread = threading.Thread(target=_writer_loop, daemon=True)
        writer_thread.start()
        pool_ctx = ThreadPoolExecutor(max_workers=sub_w) if sub_w > 1 else None

        def _enqueue_shard_read(sid_val: int) -> tuple[int, list[int], list[Any]]:
            c_idxs = shard_groups[sid_val]
            n_sub = min(sub_w, len(c_idxs)) if (pool_ctx is not None and len(c_idxs) >= 4) else 1
            if n_sub <= 1:
                return sid_val, c_idxs, [
                    (
                        pool_ctx.submit(
                            _read_shard_records,
                            sid_val,
                            c_idxs,
                            dataset=dataset,
                            seed_state=seed_state,
                            max_seed_bytes=max_seed_bytes,
                            dir_fds=dir_fds,
                            on_clip_done=_tick,
                        )
                        if pool_ctx is not None
                        else _read_shard_records(
                            sid_val,
                            c_idxs,
                            dataset=dataset,
                            seed_state=seed_state,
                            max_seed_bytes=max_seed_bytes,
                            dir_fds=dir_fds,
                            on_clip_done=_tick,
                        )
                    )
                ]
            step = (len(c_idxs) + n_sub - 1) // n_sub
            sub_slices = [s for s in (c_idxs[k * step : (k + 1) * step] for k in range(n_sub)) if s]
            assert pool_ctx is not None
            futs = [
                pool_ctx.submit(
                    _read_shard_records,
                    sid_val,
                    sub_s,
                    dataset=dataset,
                    seed_state=seed_state,
                    max_seed_bytes=max_seed_bytes,
                    dir_fds=dir_fds,
                    on_clip_done=_tick,
                )
                for sub_s in sub_slices
            ]
            return sid_val, c_idxs, futs

        try:
            # 1-shard lookahead keeps heap small (< ~450 MiB) while removing barrier gaps
            pending_shards: deque[tuple[int, list[int], list[Any]]] = deque()
            next_pos = 0
            while next_pos < len(missing_shards) and len(pending_shards) < 2:
                pending_shards.append(_enqueue_shard_read(missing_shards[next_pos]))
                next_pos += 1

            while pending_shards:
                if writer_exc:
                    raise writer_exc[0]
                sid, clip_idxs, handles = pending_shards.popleft()
                if next_pos < len(missing_shards) and not write_stop.is_set():
                    pending_shards.append(_enqueue_shard_read(missing_shards[next_pos]))
                    next_pos += 1

                rec_list: list[bytes] = []
                for h in handles:
                    _, _, sub_recs, sub_seeds = h.result() if isinstance(h, Future) else h
                    rec_list.extend(sub_recs)
                    for gidx, s_item in sub_seeds:
                        seed_items[gidx] = s_item

                while not write_stop.is_set():
                    try:
                        write_queue.put((sid, clip_idxs, rec_list), timeout=0.1)
                        break
                    except queue.Full:
                        if writer_exc:
                            raise writer_exc[0]
                if writer_exc:
                    raise writer_exc[0]

            while not write_stop.is_set():
                try:
                    write_queue.put(writer_sentinel, timeout=0.1)
                    break
                except queue.Full:
                    if writer_exc:
                        raise writer_exc[0]
            writer_thread.join()
            if writer_exc:
                raise writer_exc[0]
        finally:
            write_stop.set()
            if pool_ctx is not None:
                pool_ctx.shutdown(wait=True, cancel_futures=True)
            if writer_thread.is_alive():
                try:
                    while not write_queue.empty():
                        write_queue.get_nowait()
                        write_queue.task_done()
                except Exception:
                    pass
                try:
                    write_queue.put_nowait(writer_sentinel)
                except queue.Full:
                    pass
                writer_thread.join(timeout=5.0)
            dir_fds.close()
            if pbar is not None:
                pbar.close()

    # Assemble global index.npy and commit manifest.json
    global_idx = np.empty((n_clips, 3), dtype=np.int64)
    for sid in range(num_shards):
        idx_p = os.path.join(shards_dir, f"shard_{sid:05d}.idx.npy")
        s_idx = np.load(idx_p, allow_pickle=False)
        g_ids = s_idx[:, 0]
        global_idx[g_ids, 0] = sid
        global_idx[g_ids, 1] = s_idx[:, 1]
        global_idx[g_ids, 2] = s_idx[:, 2]

    index_path = os.path.join(shards_dir, _SHARD_INDEX_NAME)
    index_tmp = f"{index_path}.{os.getpid()}.tmp"
    with open(index_tmp, "wb") as gf:
        np.save(gf, global_idx)
    os.replace(index_tmp, index_path)

    manifest = dict(fp)
    manifest["clips_per_shard"] = per_shard
    manifest["num_shards"] = num_shards
    _atomic_write_json(os.path.join(shards_dir, _SHARD_MANIFEST_NAME), manifest)

    reader = _DatasetShardReader(shards_dir, global_idx, expected_sr=fp["sample_rate"])
    dataset._shard_reader = reader
    return reader, seed_items


def preload_shards_into_cache(
    dataset: "TextAudioDataset",
    sample_cache: dict[int, tuple[Any, ...]],
    *,
    max_bytes: int,
    workers: int = 4,
    progress: bool = True,
    repeat_phone: bool | None = None,
) -> int:
    """Sequentially stream whole `.rpack` shards into RAM `_sample_cache`, avoiding slow random 640K seeks."""
    use_repeat = bool(getattr(dataset, "repeat_phone", True)) if repeat_phone is None else bool(repeat_phone)
    cur_bytes = sum(_sample_nbytes(v) for v in sample_cache.values())
    reader = getattr(dataset, "_shard_reader", None)
    if reader is None or max_bytes <= 0 or cur_bytes >= max_bytes:
        return cur_bytes

    n_items = len(dataset)
    idx_arr = reader.index
    per_shard: dict[int, list[tuple[int, int, int]]] = {}
    missing_total = 0
    for gidx in range(n_items):
        if gidx in sample_cache:
            continue
        sid = int(idx_arr[gidx, 0])
        per_shard.setdefault(sid, []).append((gidx, int(idx_arr[gidx, 1]), int(idx_arr[gidx, 2])))
        missing_total += 1
    if not per_shard:
        return cur_bytes

    for entries in per_shard.values():
        entries.sort(key=lambda t: t[1])

    total_bytes = 0
    shard_sizes: dict[int, int] = {}
    for sid in sorted(per_shard.keys()):
        sp = os.path.join(reader.shards_dir, f"shard_{sid:05d}.rpack")
        try:
            sz = os.path.getsize(sp)
        except OSError:
            sz = 0
        shard_sizes[sid] = sz
        total_bytes += sz

    decode_w = min(8, max(2, int(workers)))
    pbar = None
    if progress:
        print(
            f"Loading {missing_total} clip(s) ({total_bytes / (1024 ** 3):.2f} GiB) sequentially from {len(per_shard)} shard(s) into host RAM...",
            flush=True,
        )
        try:
            from tqdm import tqdm

            pbar = tqdm(
                total=max(1, total_bytes),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Shard -> RAM",
                file=sys.stdout,
                mininterval=0.25,
                leave=True,
            )
        except Exception:
            pbar = None

    def _decode_slice(mv_blob: memoryview, item: tuple[int, int, int]):
        gidx, off, nb = item
        rec = _decode_clip_record(
            mv_blob[off : off + nb],
            expected_sr=dataset.sample_rate,
            repeat_phone=use_repeat,
            phone_dtype=dataset.phone_dtype,
            compute_spec=dataset.compute_spec,
            allow_pcm16=not dataset.compute_spec,
            filter_length=dataset.filter_length,
            hop_length=dataset.hop_length,
            win_length=dataset.win_length,
        )
        return gidx, rec

    open_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    _READ_CHUNK = 8 * 1024 * 1024
    # Reuse one buffer across shards so 65 shards do not churn 20 GiB of heap allocations
    reusable_buf = bytearray(max((shard_sizes.values()), default=0))
    try:
        with ThreadPoolExecutor(max_workers=decode_w) as pool:
            for sid in sorted(per_shard.keys()):
                if cur_bytes >= max_bytes:
                    break
                entries = per_shard[sid]
                shard_path = os.path.join(reader.shards_dir, f"shard_{sid:05d}.rpack")
                fd = os.open(shard_path, open_flags)
                fsize = shard_sizes.get(sid, 0)
                if fsize <= 0:
                    fsize = os.fstat(fd).st_size
                if len(reusable_buf) < fsize:
                    reusable_buf = bytearray(fsize)
                mv_full = memoryview(reusable_buf)[:fsize]
                got = 0
                try:
                    if hasattr(os, "posix_fadvise"):
                        try:
                            os.posix_fadvise(fd, 0, fsize, os.POSIX_FADV_SEQUENTIAL)
                        except OSError:
                            pass
                    with os.fdopen(fd, "rb", buffering=0, closefd=False) as rf:
                        while got < fsize:
                            end = min(fsize, got + _READ_CHUNK)
                            n_read = rf.readinto(mv_full[got:end])
                            if not n_read:
                                break
                            got += n_read
                            if pbar is not None:
                                pbar.update(n_read)
                    _fadvise_dontneed(fd)
                finally:
                    os.close(fd)

                mv_slice = mv_full[:got]
                try:
                    if len(entries) <= 4:
                        decoded_list = [_decode_slice(mv_slice, it) for it in entries]
                    else:
                        decoded_list = list(pool.map(lambda it: _decode_slice(mv_slice, it), entries))
                finally:
                    try:
                        mv_slice.release()
                    except BufferError:
                        pass
                    try:
                        mv_full.release()
                    except BufferError:
                        pass

                for gidx, rec in decoded_list:
                    n_b = _sample_nbytes(rec)
                    if cur_bytes + n_b > max_bytes:
                        break
                    sample_cache[gidx] = rec
                    cur_bytes += n_b
                if cur_bytes >= max_bytes:
                    break
    finally:
        if pbar is not None:
            pbar.close()
        del reusable_buf

    if len(sample_cache) >= n_items:
        reader.close()
    return cur_bytes


def _hann_window_np(win_length: int) -> np.ndarray:
    if win_length <= 1:
        return np.ones((max(1, win_length),), dtype=np.float32)
    idx = np.arange(win_length, dtype=np.float64)
    return (0.5 * (1.0 - np.cos(2.0 * math.pi * idx / float(win_length)))).astype(np.float32)


def spectrogram_np(y: np.ndarray, n_fft: int, hop_size: int, win_size: int) -> np.ndarray:
    """Pure-NumPy linear STFT magnitude matching spectrogram_torch(center=False) -> (TT, F)."""
    y_1d = np.asarray(y, dtype=np.float32).reshape(-1)
    pad = int((n_fft - hop_size) / 2)
    y_pad = np.pad(y_1d, (pad, pad), mode="reflect")

    win = _hann_window_np(win_size)
    if win_size < n_fft:
        w_left = (n_fft - win_size) // 2
        w_right = n_fft - win_size - w_left
        win = np.pad(win, (w_left, w_right))

    frames = np.lib.stride_tricks.sliding_window_view(y_pad, window_shape=n_fft)[::hop_size]
    windowed = frames * win[None, :]
    spec_c = np.fft.rfft(windowed, n=n_fft, axis=-1)
    mag = np.sqrt(spec_c.real * spec_c.real + spec_c.imag * spec_c.imag + 2e-7)
    return np.ascontiguousarray(mag.astype(np.float32))


def load_filepaths_and_text(filename: str, split: str = "|") -> list[list[str]]:
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip().split(split) for line in f if line.strip()]


def _probe_audio_frames(audiopath: str, hop_length: int, max_sample_frames: int) -> int:
    try:
        info = sf.info(audiopath)
        if getattr(info, "frames", 0) > 0:
            return min(int(info.frames) // hop_length, max_sample_frames)
    except Exception:
        pass
    return min(os.path.getsize(audiopath) // (3 * hop_length), max_sample_frames)


class TextAudioDataset:
    """Loads WAV audio, optional linear STFT, phoneme features, and pitch for NWC JAX training."""

    def __init__(
        self,
        data_cfg,
        *,
        compute_spec: bool = True,
        repeat_phone: bool = True,
        phone_dtype: Any = np.float32,
    ) -> None:
        self.training_files = str(data_cfg.training_files)
        self.audiopaths_and_text = load_filepaths_and_text(self.training_files)
        self.max_wav_value = float(data_cfg.max_wav_value)
        self.sample_rate = int(data_cfg.sample_rate)
        self.filter_length = int(data_cfg.filter_length)
        self.hop_length = int(data_cfg.hop_length)
        self.win_length = int(data_cfg.win_length)
        self.max_sample_frames = 896
        self.compute_spec = bool(compute_spec)
        self.repeat_phone = bool(repeat_phone)
        self.phone_dtype = np.dtype(phone_dtype)
        self.min_text_len = int(getattr(data_cfg, "min_text_len", 1))
        self.max_text_len = int(getattr(data_cfg, "max_text_len", 5000))
        self._filter()
        self._shard_reader: _DatasetShardReader | None = _open_validated_shard_reader(self)

    @classmethod
    def from_cached_entries(cls, cfg_dict: Mapping[str, Any]) -> "TextAudioDataset":
        """Reconstruct dataset in worker processes using pre-filtered file lists and lengths."""
        obj = cls.__new__(cls)
        obj.training_files = str(cfg_dict.get("training_files", ""))
        obj.audiopaths_and_text = [list(row) for row in cfg_dict["audiopaths_and_text"]]
        obj.lengths = [int(x) for x in cfg_dict.get("lengths", ())]
        obj.max_wav_value = float(cfg_dict.get("max_wav_value", 32768.0))
        obj.sample_rate = int(cfg_dict["sample_rate"])
        obj.filter_length = int(cfg_dict["filter_length"])
        obj.hop_length = int(cfg_dict["hop_length"])
        obj.win_length = int(cfg_dict["win_length"])
        obj.max_sample_frames = int(cfg_dict.get("max_sample_frames", 896))
        obj.compute_spec = bool(cfg_dict.get("compute_spec", False))
        obj.repeat_phone = bool(cfg_dict.get("repeat_phone", True))
        obj.phone_dtype = np.dtype(cfg_dict.get("phone_dtype", "float32"))
        obj.min_text_len = int(cfg_dict.get("min_text_len", 1))
        obj.max_text_len = int(cfg_dict.get("max_text_len", 5000))
        obj._shard_reader = _open_validated_shard_reader(obj)
        return obj

    def worker_config(self, hop_length: int, allowed_frames: Sequence[int] = ()) -> dict[str, Any]:
        """Serialize picklable dataset metadata for multi-process batch workers."""
        return {
            "training_files": getattr(self, "training_files", ""),
            "audiopaths_and_text": self.audiopaths_and_text,
            "lengths": self.lengths,
            "max_wav_value": self.max_wav_value,
            "sample_rate": self.sample_rate,
            "filter_length": self.filter_length,
            "hop_length": int(hop_length),
            "win_length": self.win_length,
            "max_sample_frames": self.max_sample_frames,
            "compute_spec": self.compute_spec,
            "repeat_phone": self.repeat_phone,
            "phone_dtype": self.phone_dtype.name,
            "min_text_len": self.min_text_len,
            "max_text_len": self.max_text_len,
            "allowed_frames": tuple(int(x) for x in allowed_frames),
        }

    def _filter(self) -> None:
        kept: list[list[str]] = []
        for entry in self.audiopaths_and_text:
            audiopath, text, pitch, pitchf, dv = entry[:5]
            if self.min_text_len <= len(text) <= self.max_text_len:
                kept.append([audiopath, text, pitch, pitchf, dv])

        training_files = getattr(self, "training_files", "")
        cache_path = f"{training_files}.lengths.json" if training_files else ""
        src_stat = None
        if cache_path and os.path.isfile(training_files):
            try:
                src_stat = os.stat(training_files)
                if os.path.isfile(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as cf:
                        meta = json.load(cf)
                    cached = meta.get("lengths")
                    if (
                        meta.get("mtime_ns") == src_stat.st_mtime_ns
                        and meta.get("size") == src_stat.st_size
                        and meta.get("count") == len(kept)
                        and meta.get("hop_length") == self.hop_length
                        and meta.get("max_sample_frames") == self.max_sample_frames
                        and isinstance(cached, list)
                        and len(cached) == len(kept)
                    ):
                        self.audiopaths_and_text = kept
                        self.lengths = [int(x) for x in cached]
                        return
            except (OSError, ValueError, TypeError):
                pass

        if len(kept) > 32:
            workers = min(32, max(4, effective_cpu_count() * 4))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                lengths = list(
                    pool.map(
                        lambda row: _probe_audio_frames(row[0], self.hop_length, self.max_sample_frames),
                        kept,
                    )
                )
        else:
            lengths = [_probe_audio_frames(row[0], self.hop_length, self.max_sample_frames) for row in kept]
        self.audiopaths_and_text = kept
        self.lengths = lengths

        if cache_path and src_stat is not None:
            try:
                tmp_path = f"{cache_path}.tmp.{os.getpid()}"
                with open(tmp_path, "w", encoding="utf-8") as cf:
                    json.dump(
                        {
                            "mtime_ns": src_stat.st_mtime_ns,
                            "size": src_stat.st_size,
                            "count": len(kept),
                            "hop_length": self.hop_length,
                            "max_sample_frames": self.max_sample_frames,
                            "lengths": lengths,
                        },
                        cf,
                    )
                os.replace(tmp_path, cache_path)
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self.audiopaths_and_text)

    def load_labels(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int32, int]:
        """Load trimmed 50Hz/100Hz phoneme and F0 labels for sample `index`."""
        _, phone_path, pitch_path, pitchf_path, dv_str = self.audiopaths_and_text[index]
        phone_raw = _fast_load_npy(phone_path, max_rows=(self.max_sample_frames + 1) // 2 + 1)
        pitch = _fast_load_npy(pitch_path, max_rows=self.max_sample_frames)
        pitchf = _fast_load_npy(pitchf_path, max_rows=self.max_sample_frames)

        n_num = min(int(phone_raw.shape[0]) * 2, int(pitch.shape[0]), int(pitchf.shape[0]), self.max_sample_frames)
        n_phone_half = (n_num + 1) // 2
        phone_half = np.asarray(phone_raw[:n_phone_half, :], dtype=self.phone_dtype)
        pitch_arr = np.asarray(pitch[:n_num], dtype=np.int32)
        pitchf_arr = np.asarray(pitchf[:n_num], dtype=np.float32)
        try:
            sid = np.int32(int(dv_str))
        except ValueError:
            sid = np.int32(0)
        return phone_half, pitch_arr, pitchf_arr, sid, n_num

    def load_item_with_cached_labels(
        self,
        index: int,
        labels: tuple[np.ndarray, np.ndarray, np.ndarray, np.int32, int],
        *,
        allow_pcm16: bool = False,
        repeat_phone: bool | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
        """Load WAV waveform and align with pre-loaded labels."""
        audiopath = self.audiopaths_and_text[index][0]
        phone_half, pitch, pitchf, sid, n_num = labels

        keep_i16 = bool(allow_pcm16 and not self.compute_spec)
        use_repeat = self.repeat_phone if repeat_phone is None else bool(repeat_phone)
        audio, sr = _fast_load_wav(audiopath, max_samples=n_num * self.hop_length, allow_int16=keep_i16)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if int(sr) != self.sample_rate:
            raise ValueError(f"{sr} SR doesn't match target {self.sample_rate} SR in '{audiopath}'.")

        n_frames_audio = min(len(audio) // self.hop_length, self.max_sample_frames)
        len_min = min(n_num, n_frames_audio)
        len_wav = len_min * self.hop_length
        if keep_i16 and audio.dtype == np.int16:
            wav = np.ascontiguousarray(audio[:len_wav], dtype=np.int16)[:, None]
        else:
            audio_f32 = np.asarray(audio[:len_wav], dtype=np.float32)
            wav = audio_f32[:, None]  # (T_wav, 1) NWC

        if len_min < n_num:
            phone_use = phone_half[: (len_min + 1) // 2, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]
        else:
            phone_use = phone_half

        phone = np.repeat(phone_use, 2, axis=0)[:len_min, :] if use_repeat else phone_use
        spec = (
            spectrogram_np(np.ascontiguousarray(wav[:, 0], dtype=np.float32), self.filter_length, self.hop_length, self.win_length)
            if self.compute_spec
            else None
        )
        return spec, wav, phone, pitch, pitchf, sid

    @property
    def has_shards(self) -> bool:
        return getattr(self, "_shard_reader", None) is not None

    def load_item_fast(
        self,
        index: int,
        *,
        allow_pcm16: bool = False,
        repeat_phone: bool | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
        """Read clip directly from contiguous `.rpack` shard when available, else loose files."""
        use_repeat = self.repeat_phone if repeat_phone is None else bool(repeat_phone)
        reader = getattr(self, "_shard_reader", None)
        if reader is not None:
            raw = reader.pread_clip(index)
            return _decode_clip_record(
                raw,
                expected_sr=self.sample_rate,
                repeat_phone=use_repeat,
                phone_dtype=self.phone_dtype,
                compute_spec=self.compute_spec,
                allow_pcm16=allow_pcm16,
                filter_length=self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )
        return self.load_item_with_cached_labels(
            index,
            self.load_labels(index),
            allow_pcm16=allow_pcm16,
            repeat_phone=use_repeat,
        )

    def __getitem__(self, index: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
        if getattr(self, "_shard_reader", None) is not None:
            return self.load_item_fast(index, allow_pcm16=False)
        return self.load_item_with_cached_labels(index, self.load_labels(index))


class BucketBatchSampler:
    """Length-bucketed batch sampler emitting (bucket_target_frames, [sample_indices]) pairs."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Sequence[int],
        *,
        shuffle: bool = True,
        max_frames: int | None = None,
        batch_multiple: int = 1,
        static_bucket_padding: bool = True,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.boundaries = list(boundaries)
        self.shuffle = bool(shuffle)
        self.max_frames = max_frames
        self.batch_multiple = max(1, int(batch_multiple))
        self.static_bucket_padding = bool(static_bucket_padding)
        self.epoch = 0
        self.start_batch = 0

        self.buckets, self.bucket_upper_bounds, self.bucket_batch_sizes, self.num_samples_per_bucket = (
            self._create_buckets()
        )
        self.allowed_frames: tuple[int, ...] = (
            tuple(sorted(set(self.bucket_upper_bounds))) if self.static_bucket_padding else ()
        )

    def unique_batch_shapes(self) -> list[tuple[int, int]]:
        """Return sorted unique (batch_size, target_frames) static bucket shapes."""
        if not self.static_bucket_padding:
            return []
        return sorted(set(zip(self.bucket_batch_sizes, self.bucket_upper_bounds)))

    def _bisect(self, x: int) -> int:
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] < x <= self.boundaries[i + 1]:
                return i
        return -1

    def _align_batch_size(self, raw_bs: int) -> int:
        if raw_bs <= self.batch_multiple:
            return self.batch_multiple
        return (raw_bs // self.batch_multiple) * self.batch_multiple

    def _create_buckets(self) -> tuple[list[list[int]], list[int], list[int], list[int]]:
        raw_buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) - 1)]
        for idx, length in enumerate(self.lengths):
            b_idx = self._bisect(length)
            if b_idx != -1:
                raw_buckets[b_idx].append(idx)

        buckets: list[list[int]] = []
        upper_bounds: list[int] = []
        for i, b in enumerate(raw_buckets):
            if b:
                buckets.append(b)
                upper_bounds.append(int(self.boundaries[i + 1]))

        batch_sizes: list[int] = []
        num_samples_per_bucket: list[int] = []
        for b in buckets:
            if self.max_frames:
                longest = max(self.lengths[idx] for idx in b)
                bs = max(self.batch_multiple, min(self.batch_size, self.max_frames // max(1, longest)))
                bs = self._align_batch_size(bs)
            else:
                bs = self._align_batch_size(self.batch_size)
            batch_sizes.append(bs)
            rem = (bs - (len(b) % bs)) % bs
            num_samples_per_bucket.append(len(b) + rem)
        return buckets, upper_bounds, batch_sizes, num_samples_per_bucket

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)

    def __len__(self) -> int:
        return sum(n // bs for n, bs in zip(self.num_samples_per_bucket, self.bucket_batch_sizes))

    def __iter__(self) -> Iterator[tuple[int | None, list[int]]]:
        rng = np.random.default_rng(self.epoch)
        batches: list[tuple[int | None, list[int]]] = []
        for i, bucket in enumerate(self.buckets):
            n_b = len(bucket)
            perm = rng.permutation(n_b).tolist() if self.shuffle else list(range(n_b))
            n_target = self.num_samples_per_bucket[i]
            rem = n_target - n_b
            perm = perm + perm * (rem // n_b) + perm[: (rem % n_b)]

            bs = self.bucket_batch_sizes[i]
            target_frames = self.bucket_upper_bounds[i] if self.static_bucket_padding else None
            for j in range(len(perm) // bs):
                batch_idx = [bucket[p] for p in perm[j * bs : (j + 1) * bs]]
                batches.append((target_frames, batch_idx))

        if self.shuffle and batches:
            order = rng.permutation(len(batches)).tolist()
            batches = [batches[k] for k in order]

        return iter(batches[self.start_batch :])


def _snap_frames_to_bucket(needed: int, target_frames: int | None, allowed_frames: Sequence[int] | None) -> int:
    if target_frames is None:
        return needed
    target = int(target_frames)
    if needed <= target:
        return target
    if allowed_frames:
        for bound in allowed_frames:
            if bound >= needed:
                return int(bound)
    return needed


def collate_nwc_batch(
    rows: Sequence[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]],
    hop_length: int,
    target_frames: int | None = None,
    allowed_frames: Sequence[int] | None = None,
    phone_half: bool = False,
) -> tuple[np.ndarray, ...]:
    """Collate sample rows into NWC NumPy arrays with optional static XLA bucket padding."""
    del phone_half
    has_spec = rows[0][0] is not None
    # Pitch (index 3) always tracks full 100Hz frame count even when RAM cache stores compact unrepeated phones.
    sorted_rows = sorted(rows, key=lambda r: r[3].shape[0], reverse=True)
    batch_size = len(sorted_rows)

    phone_lengths = np.asarray([r[3].shape[0] for r in sorted_rows], dtype=np.int32)
    wave_lengths = np.asarray([r[1].shape[0] for r in sorted_rows], dtype=np.int32)
    spec_lengths = (
        np.asarray([r[0].shape[0] for r in sorted_rows], dtype=np.int32)
        if has_spec
        else phone_lengths.copy()
    )

    max_spec = int(spec_lengths.max())
    max_phone = int(phone_lengths.max())
    frames = _snap_frames_to_bucket(max(max_spec, max_phone), target_frames, allowed_frames)
    wave_frames = max(frames * int(hop_length), int(wave_lengths.max()))

    phone_dim = sorted_rows[0][2].shape[1]

    phone_padded = np.empty((batch_size, frames, phone_dim), dtype=sorted_rows[0][2].dtype)
    pitch_padded = np.empty((batch_size, frames), dtype=np.int32)
    pitchf_padded = np.empty((batch_size, frames), dtype=np.float32)
    wave_padded = np.empty((batch_size, wave_frames, 1), dtype=np.float32)
    sid = np.asarray([int(r[5]) for r in sorted_rows], dtype=np.int32)
    spec_padded = (
        np.empty((batch_size, frames, sorted_rows[0][0].shape[1]), dtype=np.float32)
        if has_spec
        else None
    )

    for i, (spec, wav, phone, pitch, pitchf, _) in enumerate(sorted_rows):
        w_len = wav.shape[0]
        p_len = phone.shape[0]
        pt_len = pitch.shape[0]
        pf_len = pitchf.shape[0]
        if has_spec and spec_padded is not None and spec is not None:
            s_len = spec.shape[0]
            spec_padded[i, :s_len, :] = spec
            if s_len < frames:
                spec_padded[i, s_len:, :] = 0.0
        if wav.dtype == np.int16:
            np.multiply(wav.astype(np.float32), _PCM16_INV, out=wave_padded[i, :w_len, :])
        else:
            wave_padded[i, :w_len, :] = wav
        if w_len < wave_frames:
            wave_padded[i, w_len:, :] = 0.0
        if p_len == pt_len:
            phone_padded[i, :p_len, :] = phone
        elif p_len < pt_len and p_len == (pt_len + 1) // 2:
            # Expand compact unrepeated cache entry directly into 100Hz frame slots (identical to np.repeat(..., 2)[:pt_len]).
            even_n = (pt_len + 1) // 2
            odd_n = pt_len // 2
            phone_padded[i, 0:pt_len:2, :] = phone[:even_n]
            phone_padded[i, 1:pt_len:2, :] = phone[:odd_n]
        else:
            rep = np.repeat(phone, 2, axis=0) if p_len < pt_len else phone
            phone_padded[i, :pt_len, :] = rep[:pt_len]
        if pt_len < frames:
            phone_padded[i, pt_len:, :] = 0.0
        pitch_padded[i, :pt_len] = pitch
        if pt_len < frames:
            pitch_padded[i, pt_len:] = 0
        pitchf_padded[i, :pf_len] = pitchf
        if pf_len < frames:
            pitchf_padded[i, pf_len:] = 0.0

    if has_spec and spec_padded is not None:
        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )
    return (
        phone_padded,
        phone_lengths,
        pitch_padded,
        pitchf_padded,
        spec_lengths,
        wave_padded,
        wave_lengths,
        sid,
    )


class PrefetchBatchLoader:
    """Multi-process / threaded NumPy batch prefetcher + bounded RAM cache for high TPU throughput."""

    def __init__(
        self,
        dataset: TextAudioDataset,
        sampler: BucketBatchSampler,
        hop_length: int,
        num_workers: int = 4,
        prefetch_batches: int = 8,
        cache_in_ram: bool = False,
        use_multiprocessing: bool | None = None,
        enable_disk_pack: bool | None = None,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.hop_length = int(hop_length)
        self.num_workers = max(0, int(num_workers))
        self.prefetch_batches = max(2, int(prefetch_batches))
        self.cache_in_ram = bool(cache_in_ram)
        self.allowed_frames = getattr(sampler, "allowed_frames", ())
        self.use_multiprocessing = bool(use_multiprocessing) and (not self.cache_in_ram) and (self.num_workers > 0)
        self._cache: list[tuple[np.ndarray, ...]] | None = None
        avail = _available_ram_bytes()
        self._sample_cache: dict[int, tuple[Any, ...]] = {}
        self._sample_cache_bytes = 0
        self._sample_cache_lock = threading.Lock()
        self._sample_cache_limit = max(0, min(int(avail * 0.65), avail - 4 * (1024 ** 3)))
        self._label_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.int32, int]] = {}
        self._label_cache_bytes = 0
        self._label_cache_limit = max(0, min(int(avail * 0.72), avail - 3 * (1024 ** 3)))
        self._inflight: dict[int, Future] = {}
        self._warmup_stop = threading.Event()
        self._warmup_thread: threading.Thread | None = None
        self._closed = False
        self._mp_pool = None
        # Overlayfs random-read IOPS peaks at ~8 threads; higher thread counts degrade throughput.
        self._pool_workers = (
            min(8, max(4, self.num_workers))
            if self.num_workers > 0 and not self.use_multiprocessing
            else 0
        )
        self._pool = (
            ThreadPoolExecutor(max_workers=self._pool_workers)
            if self._pool_workers > 0
            else None
        )
        tf = getattr(dataset, "training_files", "")
        if tf:
            for ext in (".pack.bin", ".pack.toc", ".pack.meta.json"):
                stale_p = f"{tf}{ext}"
                if os.path.isfile(stale_p):
                    try:
                        os.remove(stale_p)
                    except OSError:
                        pass
        if bool(enable_disk_pack) and isinstance(dataset, TextAudioDataset):
            # Build/verify shards without 20+ GiB heap churn, then stream sequential whole shards cleanly into RAM.
            _reader, seeded = ensure_dataset_shards(
                dataset,
                workers=max(1, self._pool_workers or self.num_workers or 4),
                progress=True,
                max_seed_bytes=0,
            )
            if seeded:
                for gidx, s_item in seeded.items():
                    self._sample_cache[gidx] = s_item
                    self._sample_cache_bytes += _sample_nbytes(s_item)
                seeded.clear()
            if (
                dataset.has_shards
                and not self.cache_in_ram
                and not self.use_multiprocessing
                and self._sample_cache_limit > 0
            ):
                preload_budget = int(self._sample_cache_limit * 0.92)
                self._sample_cache_bytes = preload_shards_into_cache(
                    dataset,
                    self._sample_cache,
                    max_bytes=preload_budget,
                    workers=max(1, self._pool_workers or self.num_workers or 4),
                    progress=True,
                    repeat_phone=False,
                )

    def _try_cache_sample(self, idx: int, item: tuple[Any, ...]) -> None:
        if self._sample_cache_limit <= 0 or self._sample_cache_bytes >= self._sample_cache_limit:
            return
        item_bytes = _sample_nbytes(item)
        with self._sample_cache_lock:
            if idx not in self._sample_cache and self._sample_cache_bytes + item_bytes <= self._sample_cache_limit:
                self._sample_cache[idx] = item
                self._sample_cache_bytes += item_bytes

    def _get_item(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        cached = self._sample_cache.get(idx)
        if cached is not None:
            return cached
        if getattr(self.dataset, "has_shards", False):
            item = self.dataset.load_item_fast(idx, allow_pcm16=True, repeat_phone=False)
            self._try_cache_sample(idx, item)
            return item
        cached_lbl = self._label_cache.get(idx)
        if cached_lbl is not None:
            return self.dataset.load_item_with_cached_labels(
                idx, cached_lbl, allow_pcm16=True, repeat_phone=False
            )

        if hasattr(self.dataset, "load_labels"):
            labels = self.dataset.load_labels(idx)
            item = self.dataset.load_item_with_cached_labels(
                idx, labels, allow_pcm16=True, repeat_phone=False
            )
            if (self._sample_cache_bytes + self._label_cache_bytes) < self._label_cache_limit:
                item_bytes = _sample_nbytes(item)
                lbl_bytes = _sample_nbytes(labels[:3])
                with self._sample_cache_lock:
                    if idx not in self._sample_cache and (
                        self._sample_cache_bytes + item_bytes <= self._sample_cache_limit
                    ):
                        self._sample_cache[idx] = item
                        self._sample_cache_bytes += item_bytes
                    elif idx not in self._label_cache and (
                        self._sample_cache_bytes + self._label_cache_bytes + lbl_bytes <= self._label_cache_limit
                    ):
                        self._label_cache[idx] = labels
                        self._label_cache_bytes += lbl_bytes
            return item

        item = self.dataset[idx]
        self._try_cache_sample(idx, item)
        return item

    def _load_sample_tracked(self, idx: int):
        try:
            return self._get_item(idx)
        finally:
            with self._sample_cache_lock:
                self._inflight.pop(idx, None)

    def _submit_sample(self, idx: int) -> tuple[bool, Any]:
        if self._cache is not None:
            return True, self._cache[idx]
        cached = self._sample_cache.get(idx)
        if cached is not None:
            return True, cached
        with self._sample_cache_lock:
            cached = self._sample_cache.get(idx)
            if cached is not None:
                return True, cached
            if self._closed or self._pool is None:
                return True, self._get_item(idx)
            fut = self._inflight.get(idx)
            if fut is None:
                fut = self._pool.submit(self._load_sample_tracked, idx)
                self._inflight[idx] = fut
            return False, fut

    def start_warmup(
        self,
        epoch: int = 1,
        start_batch: int = 0,
        max_samples: int | None = None,
    ) -> None:
        """Pre-populate sample/label RAM caches during XLA compile in exact Epoch batch order."""
        if (
            self._sample_cache_limit <= 0
            or self.cache_in_ram
            or self.use_multiprocessing
            or self._pool is None
            or self._closed
        ):
            return
        if self._warmup_thread is not None and self._warmup_thread.is_alive():
            return
        self.sampler.set_epoch(epoch, start_batch)
        specs = list(iter(self.sampler))
        if not specs:
            return

        order: list[int] = []
        seen: set[int] = set()
        limit = int(max_samples) if max_samples is not None else len(self.dataset)
        for _, indices in specs:
            for raw_idx in indices:
                idx_i = int(raw_idx)
                if idx_i not in seen:
                    seen.add(idx_i)
                    order.append(idx_i)
                    if len(order) >= limit:
                        break
            if len(order) >= limit:
                break
        if not order:
            return

        self._warmup_stop.clear()

        def _warmup_loop() -> None:
            in_flight: deque[Future] = deque()
            max_pending = max(4, min(24, self._pool_workers * 2))
            for idx_i in order:
                if self._warmup_stop.is_set() or self._closed:
                    break
                if idx_i in self._sample_cache:
                    continue
                if self._sample_cache_bytes >= int(self._sample_cache_limit * 0.90):
                    break
                ready, item = self._submit_sample(idx_i)
                if not ready:
                    in_flight.append(item)
                while len(in_flight) >= max_pending:
                    fut = in_flight.popleft()
                    try:
                        fut.result()
                    except BaseException:
                        pass
            while in_flight and not self._warmup_stop.is_set() and not self._closed:
                fut = in_flight.popleft()
                try:
                    fut.result()
                except BaseException:
                    pass

        self._warmup_thread = threading.Thread(target=_warmup_loop, daemon=True)
        self._warmup_thread.start()

    def stop_warmup(self, wait: bool = False) -> None:
        self._warmup_stop.set()
        if wait and self._warmup_thread is not None:
            self._warmup_thread.join(timeout=2.0)
            self._warmup_thread = None

    def _prepare_batch(self, spec_pair: tuple[int | None, list[int]]):
        target_frames, indices = spec_pair
        rows = [self._get_item(i) for i in indices]
        return collate_nwc_batch(
            rows,
            self.hop_length,
            target_frames=target_frames,
            allowed_frames=self.allowed_frames,
        )

    def _load_for_cache(self, idx: int):
        if getattr(self.dataset, "has_shards", False):
            return self.dataset.load_item_fast(idx, allow_pcm16=True, repeat_phone=False)
        if hasattr(self.dataset, "load_labels"):
            return self.dataset.load_item_with_cached_labels(
                idx, self.dataset.load_labels(idx), allow_pcm16=True, repeat_phone=False
            )
        return self.dataset[idx]

    def _ensure_cache(self) -> None:
        if not self.cache_in_ram or self._cache is not None:
            return
        indices = list(range(len(self.dataset)))
        print(f"Caching {len(indices)} training samples into host RAM...", flush=True)
        if self._pool is not None:
            self._cache = list(self._pool.map(self._load_for_cache, indices))
        else:
            self._cache = [self._load_for_cache(i) for i in indices]

    def __iter__(self):
        self._ensure_cache()
        specs = list(iter(self.sampler))
        if not specs:
            return

        if self.use_multiprocessing and not self.cache_in_ram and self.num_workers > 0:
            if self._mp_pool is None:
                from rvc.train.jax_data_worker import SpawnBatchPool

                cfg_dict = self.dataset.worker_config(self.hop_length, self.allowed_frames)
                self._mp_pool = SpawnBatchPool(
                    cfg_dict,
                    num_workers=self.num_workers,
                    prefetch_batches=self.prefetch_batches,
                )
            yield from self._mp_pool.iter_batches(specs)
            return

        if self._pool is None or self.num_workers == 0:
            for sp in specs:
                yield self._prepare_batch(sp)
            return

        self.stop_warmup(wait=False)

        out_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        sentinel = object()
        max_sample_budget = max(192, self._pool_workers * 24)
        batch_window = max(self.prefetch_batches, 6)

        def _producer() -> None:
            pending_batches: deque[tuple[int | None, list[tuple[bool, Any]]]] = deque()
            spec_iter = iter(specs)
            in_flight_futs = 0
            try:
                while True:
                    while len(pending_batches) < batch_window and (
                        not pending_batches or in_flight_futs < max_sample_budget
                    ):
                        sp = next(spec_iter, None)
                        if sp is None:
                            break
                        target_frames, indices = sp
                        slots = [self._submit_sample(int(i)) for i in indices]
                        in_flight_futs += sum(1 for ready, _ in slots if not ready)
                        pending_batches.append((target_frames, slots))

                    if not pending_batches or stop_event.is_set():
                        break

                    target_frames, slots = pending_batches.popleft()
                    rows = []
                    for ready, item in slots:
                        if ready:
                            rows.append(item)
                        else:
                            rows.append(item.result())
                            in_flight_futs -= 1
                    if stop_event.is_set():
                        break
                    batch = collate_nwc_batch(
                        rows,
                        self.hop_length,
                        target_frames=target_frames,
                        allowed_frames=self.allowed_frames,
                    )
                    while not stop_event.is_set():
                        try:
                            out_queue.put(batch, timeout=0.05)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:
                while not stop_event.is_set():
                    try:
                        out_queue.put(exc, timeout=0.05)
                        break
                    except queue.Full:
                        continue
            finally:
                while not stop_event.is_set():
                    try:
                        out_queue.put(sentinel, timeout=0.05)
                        break
                    except queue.Full:
                        continue

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        try:
            while True:
                item = out_queue.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_event.set()
            while True:
                try:
                    if out_queue.get_nowait() is sentinel:
                        break
                except queue.Empty:
                    break
            thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop_warmup(wait=True)
        with self._sample_cache_lock:
            self._closed = True
            self._inflight.clear()
        self._sample_cache.clear()
        self._sample_cache_bytes = 0
        self._label_cache.clear()
        self._label_cache_bytes = 0
        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool = None
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        shard_reader = getattr(self.dataset, "_shard_reader", None)
        if shard_reader is not None:
            shard_reader.close()
