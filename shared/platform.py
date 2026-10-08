import os
import sys
import platform


def platform_config():
    if sys.platform == "darwin" and platform.machine() == "arm64":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _merge_folder(source, target):
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(source):
        source_path = os.path.join(source, name)
        target_path = os.path.join(target, name)
        if os.path.isdir(source_path) and not os.path.islink(source_path) and os.path.isdir(target_path):
            _merge_folder(source_path, target_path)
        elif not os.path.exists(target_path):
            os.replace(source_path, target_path)
        elif name == ".gitkeep":
            os.remove(source_path)
    try:
        os.rmdir(source)
    except OSError:
        pass


def migrate_models_folder(root):
    legacy = os.path.join(root, "rvc", "models")
    if os.path.isdir(legacy):
        _merge_folder(legacy, os.path.join(root, "models"))
