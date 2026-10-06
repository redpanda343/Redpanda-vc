import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from tqdm import tqdm

from rvc.lib.tools.prerequisites_download import _sha256, download_file

ROOT = Path(__file__).resolve().parents[2]
VOCODER_DIRECTORY = 'pc_nsf_hifigan_44.1k_hop512_128bin_2025.02'
VOCODER_FILENAME = 'model.ckpt'
VOCODER_URL = ('https://github.com/openvpi/vocoders/releases/download/'
               'pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/' + VOCODER_DIRECTORY + '.zip')
VOCODER_ARCHIVE_SHA256 = '9d98ba73727f2abb75172cf8249d75182237e8472fc3b6ed09c721ae8b0e83c6'
VOCODER_FILES = {
    'model.ckpt': 'd6dd28909d2a1a2dcf74b3e3aa0b82b48695b87979fdf41561940aeecd85c67f',
    'config.json': '983bb78f45f6790e573033bfab46163743e0182d564fc5eca6938743cca6a2ea',
    'NOTICE.txt': 'fcfca845812eb462e82f4c19985fce2abc772af979226b9297287c97e9a97f65',
    'NOTICE.zh-CN.txt': '8983f1ac07a0b240b9d4c970c8081007b0f932e5e223d88f8f8aef0e6570ed81',
    'STATEMENTS.txt': 'fe208abd522fd9e77ee16063ca2b3d6466fe3a35db70578f615d28543dd62d85',
}
_download_lock = threading.Lock()


def _valid_bundle(directory):
    return all((directory / name).is_file() and _sha256(directory / name) == checksum
               for name, checksum in VOCODER_FILES.items())


def default_vocoder():
    destination = ROOT / 'rvc' / 'models' / 'pretraineds' / 'rectified' / VOCODER_DIRECTORY
    with _download_lock:
        if not _valid_bundle(destination):
            print('Downloading the official OpenVPI Rectified Flow NSF-HiFiGAN vocoder...', flush=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.nsf-hifigan-') as temporary:
                temporary = Path(temporary)
                archive_path = temporary / (VOCODER_DIRECTORY + '.zip')
                with tqdm(total=52675337, unit='B', unit_scale=True, desc='NSF-HiFiGAN vocoder') as progress:
                    download_file(VOCODER_URL, str(archive_path), progress, expected_sha256=VOCODER_ARCHIVE_SHA256)
                with zipfile.ZipFile(archive_path) as archive:
                    for name, checksum in VOCODER_FILES.items():
                        extracted = temporary / name
                        with archive.open(VOCODER_DIRECTORY + '/' + name) as source, extracted.open('wb') as target:
                            shutil.copyfileobj(source, target)
                        if _sha256(extracted) != checksum:
                            raise IOError(f'Checksum verification failed for vocoder file {name}.')
                destination.mkdir(parents=True, exist_ok=True)
                for name in (*[name for name in VOCODER_FILES if name != VOCODER_FILENAME], VOCODER_FILENAME):
                    os.replace(temporary / name, destination / name)
    return str(destination / VOCODER_FILENAME)
