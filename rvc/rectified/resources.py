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
HNSEP_URL = 'https://github.com/yxlllc/vocal-remover/releases/download/hnsep_240512/hnsep_240512.zip'
HNSEP_ARCHIVE_SHA256 = '3d353d6a14005690819210ad0cb8134326caa286e30f329da266049ed9cdff13'
HNSEP_FILES = {
    'config.yaml': '1b05438fec32da55e3d8408aa2b1108220912ea86d30c4ed152d4f56d53f4121',
    'model.pt': 'd4dd9f8259692f9ceb5b05d86fd53553ef582a1c65184f4cfff2e00cdfe33dd2',
}
_download_lock = threading.Lock()


def _valid_bundle(directory, files=VOCODER_FILES):
    return all((directory / name).is_file() and _sha256(directory / name) == checksum
               for name, checksum in files.items())


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


def hnsep_model():
    destination = ROOT / 'rvc' / 'models' / 'predictors' / 'hnsep' / 'vr'
    with _download_lock:
        if not _valid_bundle(destination, HNSEP_FILES):
            print('Downloading the DiffSinger harmonic-noise separation model...', flush=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.hnsep-') as temporary:
                temporary = Path(temporary)
                archive_path = temporary / 'hnsep_240512.zip'
                with tqdm(total=54974388, unit='B', unit_scale=True, desc='Harmonic-noise separator') as progress:
                    download_file(HNSEP_URL, str(archive_path), progress, expected_sha256=HNSEP_ARCHIVE_SHA256)
                with zipfile.ZipFile(archive_path) as archive:
                    for name, checksum in HNSEP_FILES.items():
                        extracted = temporary / name
                        with archive.open('vr/' + name) as source, extracted.open('wb') as target:
                            shutil.copyfileobj(source, target)
                        if _sha256(extracted) != checksum:
                            raise IOError(f'Checksum verification failed for separator file {name}.')
                destination.mkdir(parents=True, exist_ok=True)
                for name in ('config.yaml', 'model.pt'):
                    os.replace(temporary / name, destination / name)
    return str(destination / 'model.pt')
