import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from tqdm import tqdm

from shared.file_hash import cached_sha256
from shared.tools.prerequisites_download import _sha256, download_file

ROOT = Path(__file__).resolve().parents[1]
VOCODER_ROOT = ROOT / 'models' / 'pretraineds' / 'rectified'
CUSTOM_VOCODER_ROOT = ROOT / 'models' / 'rectified' / 'custom'
VOCODER_EXTENSIONS = ('.ckpt', '.pth', '.onnx', '.safetensors')
DEFAULT_VOCODER = 'pc_nsf_hifigan'
VOCODERS = {
    'pc_nsf_hifigan': dict(
        title='OpenVPI PC-NSF-HiFiGAN',
        kind='pc',
        mel='openvpi_44khz',
        directory='pc_nsf_hifigan_44.1k_hop512_128bin_2025.02',
        filename='model.ckpt',
        url=('https://github.com/openvpi/vocoders/releases/download/'
             'pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip'),
        archive_sha256='9d98ba73727f2abb75172cf8249d75182237e8472fc3b6ed09c721ae8b0e83c6',
        archive_size=52675337,
        archive_folder='pc_nsf_hifigan_44.1k_hop512_128bin_2025.02',
        files={
            'model.ckpt': 'd6dd28909d2a1a2dcf74b3e3aa0b82b48695b87979fdf41561940aeecd85c67f',
            'config.json': '983bb78f45f6790e573033bfab46163743e0182d564fc5eca6938743cca6a2ea',
            'NOTICE.txt': 'fcfca845812eb462e82f4c19985fce2abc772af979226b9297287c97e9a97f65',
            'NOTICE.zh-CN.txt': '8983f1ac07a0b240b9d4c970c8081007b0f932e5e223d88f8f8aef0e6570ed81',
            'STATEMENTS.txt': 'fe208abd522fd9e77ee16063ca2b3d6466fe3a35db70578f615d28543dd62d85',
        },
    ),
    'nsf_hifigan': dict(
        title='OpenVPI NSF-HiFiGAN (2022.12)',
        kind='nsf',
        mel='openvpi_44khz',
        directory='nsf_hifigan_20221211',
        filename='model.ckpt',
        url='https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip',
        archive_sha256='d86ea84b7e2c9169afb5ccbb720b5542704be519c643f698332f2014a8f2d6bd',
        archive_size=52778967,
        archive_folder='nsf_hifigan',
        members={'model.ckpt': 'model'},
        files={
            'model.ckpt': '2c576b63b7ed952161b70fad34e0562ace502ce689195520d8a2a6c051de29d6',
            'config.json': '9707614b59c299766a91ea25b5ec62cfd813a45a902766c454f75b6868118684',
            'NOTICE.txt': 'a393b44505ccb6d1da63c2c73ccbbdaeb9b877a5227bf41b1b1e4a8429a51dd6',
            'NOTICE.zh-CN.txt': 'ea5511e12932a33481c212c1c19f6225af90ff6dc6f3e34a41050f028823ebb5',
        },
    ),
    'tgm_hifigan': dict(
        title='tgm_hifigan (pc100) by tigermeat',
        kind='pc',
        mel='openvpi_44khz',
        directory='pc_tgm_hifigan_100',
        filename='pc_tgm_hifigan_100.onnx',
        url='https://github.com/mrtigermeat/tgm_hifigan/releases/download/pc100/dsvocoder.zip',
        archive_sha256='379c25c1642038abd32771f91dc4053cd6e8954e7cb09ae5d565c7809cdb8a09',
        archive_size=52726638,
        archive_folder='dsvocoder',
        files={
            'pc_tgm_hifigan_100.onnx': '02ef00608d504ac4136a64a72b653cc812419bc958ee3996fa5429b32eb6599c',
            'vocoder.yaml': '6ea1f299241165565e6cb311548457ff96e40dee755c7dbfd7ffab5b98231843',
            'readme.txt': '2558b6726ec666a24ff94b3b4cb729e93150111c963a87930fd14006cfa2810e',
            'documents/LICENSE.txt': 'eaede9100eb76a99273b6dda37b36ae5d2c3f2495b97dc4ebde12eb69f5e3fd5',
            'documents/readme.txt': '92a87a354e7d1c5de45513d1ec48098b77d6a045803010f02566a39d831041d1',
            'documents/tgm_hifigan Data Credits.xlsx': '626f3180f9be4d2ae666031a6b1e4f796741d36b7d11c06276a664e4dc2b82f5',
        },
    ),
    'vocos_44khz': dict(
        title='Vocos 44.1 kHz by patriotyk (no F0, speech)',
        kind='vocos',
        mel='vocos_44khz',
        directory='vocos_mel_44100khz',
        filename='model.pth',
        base_url=('https://huggingface.co/patriotyk/vocos-mel-hifigan-compat-44100khz/resolve/'
                  'e7d50512f731887429abfa9ba1e82d1a76f2360d/'),
        downloads={
            'model.pth': ('pytorch_model.bin', '310df58f31cd77e56c494df8389f7ebd57bf7b594c585d9221eb9fe888785572',
                          56324327),
            'config.yaml': ('config.yaml', '0d970f5cdc1913730c417b49e476bb09bb8b874583d113f71a7f10c8bb3a4b7d', 526),
        },
    ),
    'vocos_24khz': dict(
        title='Vocos 24 kHz by charactr (no F0, speech)',
        kind='vocos',
        mel='vocos_24khz',
        directory='vocos_mel_24khz',
        filename='model.pth',
        base_url='https://huggingface.co/charactr/vocos-mel-24khz/resolve/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/',
        downloads={
            'model.pth': ('pytorch_model.bin', '97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a',
                          54365991),
            'config.yaml': ('config.yaml', 'da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc', 461),
        },
    ),
    'pupu_vocoder': dict(
        title='Pupu-Vocoder Small by spellbrush (no F0)',
        kind='pupu',
        mel='pupu_44khz',
        directory='pupu_vocoder_small',
        filename='model.safetensors',
        base_url=('https://huggingface.co/spellbrush/AliasingFreeNeuralAudioSynthesis/resolve/'
                  '2a180a639566f45f0d005627ebce21f2b658264a/pupuvocoder/checkpoint/'
                  'epoch-0051_step-2553605_loss-62.135194/'),
        downloads={
            'model.safetensors': ('model.safetensors',
                                  '62320716a66f7312584a41b48e5e87bf7969cc88a44cdafed46eb2c6129f9dfe', 60989884),
        },
    ),
}
HNSEP_URL = 'https://github.com/yxlllc/vocal-remover/releases/download/hnsep_240512/hnsep_240512.zip'
HNSEP_ARCHIVE_SHA256 = '3d353d6a14005690819210ad0cb8134326caa286e30f329da266049ed9cdff13'
HNSEP_FILES = {
    'config.yaml': '1b05438fec32da55e3d8408aa2b1108220912ea86d30c4ed152d4f56d53f4121',
    'model.pt': 'd4dd9f8259692f9ceb5b05d86fd53553ef582a1c65184f4cfff2e00cdfe33dd2',
}
_download_lock = threading.Lock()


def _valid_bundle(directory, files):
    return all((directory / name).is_file() and cached_sha256(directory / name) == checksum
               for name, checksum in files.items())


def _install_bundle(destination, url, archive_sha256, archive_size, archive_folder, files, primary, description,
                    members=None):
    with _download_lock:
        if _valid_bundle(destination, files):
            return
        print(f'Downloading the {description}...', flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.download-') as temporary:
            temporary = Path(temporary)
            archive_path = temporary / 'bundle.zip'
            with tqdm(total=archive_size, unit='B', unit_scale=True, desc=description) as progress:
                download_file(url, str(archive_path), progress, expected_sha256=archive_sha256)
            with zipfile.ZipFile(archive_path) as archive:
                for name, checksum in files.items():
                    extracted = temporary / 'files' / name
                    extracted.parent.mkdir(parents=True, exist_ok=True)
                    member = (members or {}).get(name, name)
                    member = f'{archive_folder}/{member}' if archive_folder else member
                    with archive.open(member) as source, extracted.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    if _sha256(extracted) != checksum:
                        raise IOError(f'Checksum verification failed for {description} file {name}.')
            for name in (*[name for name in files if name != primary], primary):
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary / 'files' / name, destination / name)


def _install_files(destination, base_url, downloads, description):
    with _download_lock:
        if _valid_bundle(destination, {name: checksum for name, (_, checksum, _) in downloads.items()}):
            return
        print(f'Downloading the {description}...', flush=True)
        for name, (remote, checksum, size) in downloads.items():
            with tqdm(total=size, unit='B', unit_scale=True, desc=f'{description}: {name}') as progress:
                download_file(base_url + remote, str(destination / name), progress, expected_sha256=checksum)


def custom_vocoders():
    if not CUSTOM_VOCODER_ROOT.is_dir():
        return []
    return sorted(path.relative_to(ROOT).as_posix() for path in CUSTOM_VOCODER_ROOT.rglob('*')
                  if path.is_file() and path.suffix.lower() in VOCODER_EXTENSIONS)


def custom_vocoder_file(name):
    if not isinstance(name, str) or not name.strip():
        return None
    path = Path(name.strip().strip('"'))
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path.suffix.lower() in VOCODER_EXTENSIONS and path.is_file() and path.is_relative_to(CUSTOM_VOCODER_ROOT.resolve()):
        return path
    return None


def vocoder_title(name):
    if name in VOCODERS:
        return VOCODERS[name]['title']
    return Path(name).relative_to(CUSTOM_VOCODER_ROOT.relative_to(ROOT)).as_posix()


def vocoder_path(name=DEFAULT_VOCODER):
    if name not in VOCODERS:
        custom = custom_vocoder_file(name)
        if custom is not None:
            return str(custom)
        raise ValueError(f'Unknown flow vocoder {name!r}. Choose one of {", ".join(VOCODERS)} or a vocoder file in '
                         f'{CUSTOM_VOCODER_ROOT.relative_to(ROOT).as_posix()}.')
    spec = VOCODERS[name]
    destination = VOCODER_ROOT / spec['directory']
    if 'downloads' in spec:
        _install_files(destination, spec['base_url'], spec['downloads'], f"{spec['title']} vocoder")
    else:
        _install_bundle(destination, spec['url'], spec['archive_sha256'], spec['archive_size'],
                        spec['archive_folder'], spec['files'], spec['filename'], f"{spec['title']} vocoder",
                        spec.get('members'))
    return str(destination / spec['filename'])


def recorded_name(name):
    if name in VOCODERS:
        return name
    custom = custom_vocoder_file(name)
    return custom.relative_to(ROOT).as_posix() if custom is not None else None


def recorded_vocoder(checkpoint):
    recorded = checkpoint.get('vocoder')
    name = recorded_name(recorded)
    if name is None and recorded:
        print(f'The recorded flow vocoder {recorded} was not found; using the {VOCODERS[DEFAULT_VOCODER]["title"]}.',
              flush=True)
    return name or DEFAULT_VOCODER


def hnsep_model():
    destination = ROOT / 'models' / 'predictors' / 'hnsep' / 'vr'
    _install_bundle(destination, HNSEP_URL, HNSEP_ARCHIVE_SHA256, 54974388, 'vr', HNSEP_FILES, 'model.pt',
                    'DiffSinger harmonic-noise separation model')
    return str(destination / 'model.pt')
