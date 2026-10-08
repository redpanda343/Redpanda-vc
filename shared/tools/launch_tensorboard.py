import logging
import os
import threading
import time
from urllib.parse import urlsplit

from tensorboard import program

log_path = os.path.abspath("logs")
_tb_url = None
_tb_thread = None
_tb_ready = threading.Event()
_tb_lock = threading.Lock()
PINNED_CARDS = (
    "?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss"
    "%2Fg%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A"
    "%22loss%2Fd%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag"
    "%22%3A%22loss%2Fg%2Fkl%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C"
    "%22tag%22%3A%22loss%2Fg%2Fmel%22%7D%5D"
)


def _new_tensorboard():
    tb = program.TensorBoard()
    tb.configure(
        argv=[
            None,
            "--logdir",
            log_path,
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--path_prefix",
            "/tensorboard",
        ]
    )
    return tb


def get_tb_url():
    if not _tb_url:
        return None
    try:
        parsed = urlsplit(_tb_url)
        if (
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.port
            and parsed.path.rstrip("/") == "/tensorboard"
        ):
            return _tb_url
    except ValueError:
        return None
    return None


def launch_tensorboard_pipeline():
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("tensorboard").setLevel(logging.WARNING)

    tb = _new_tensorboard()
    url = tb.launch()
    print(f"TensorBoard running at: {url}{PINNED_CARDS}")

    while True:
        time.sleep(600)


def launch_tensorboard():
    global _tb_url, _tb_thread
    with _tb_lock:
        if _tb_thread is not None and _tb_thread.is_alive():
            _tb_ready.wait(timeout=10)
            return get_tb_url()
        _tb_url = None
        _tb_ready.clear()
        _tb_thread = threading.Thread(target=_start_tb, daemon=True)
        _tb_thread.start()
    _tb_ready.wait(timeout=15)
    return get_tb_url()


def _start_tb():
    global _tb_url
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("tensorboard").setLevel(logging.WARNING)
    tb = _new_tensorboard()
    try:
        _tb_url = tb.launch()
    except Exception as error:
        _tb_url = None
        print(f"TensorBoard failed to start: {error}")
    finally:
        _tb_ready.set()
    if not _tb_url:
        return
    print(f"TensorBoard running at: {_tb_url}{PINNED_CARDS}")
    while True:
        time.sleep(600)
