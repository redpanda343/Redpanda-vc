from pathlib import Path

from tqdm import tqdm

from rvc.lib.tools.prerequisites_download import _sha256, download_file

ROOT = Path(__file__).resolve().parents[2]
VOCODER_FILENAME = 'pc_nsf_hifigan_44.1k_hop512_128bin_vocoder.pth'
VOCODER_URL = ('https://huggingface.co/shiromiya/ShiroRVC-Resources/resolve/'
               'f84d2dfb2f6cc0a3205e437c7676455971dc6bf5/vocoders/' + VOCODER_FILENAME)
VOCODER_SHA256 = '4d7c843cb663137a28b94e8707503d540e5eec4b16e40867ae0d17f07440cc8d'


def default_vocoder(recorded=''):
    if recorded:
        path = Path(recorded)
        is_default = str(recorded).replace('\\', '/').split('/')[-1] == VOCODER_FILENAME
        for candidate in (path, ROOT / path):
            if candidate.is_file() and (not is_default or _sha256(candidate) == VOCODER_SHA256):
                return str(candidate)
        if not is_default:
            raise ValueError(f'The flow requires its custom NSF-HiFiGAN vocoder: {recorded}. Restore that file or supply rectified_vocoder_path.')
    destination = ROOT / 'rvc' / 'models' / 'pretraineds' / 'rectified' / VOCODER_FILENAME
    if not destination.is_file() or _sha256(destination) != VOCODER_SHA256:
        print('Downloading the default Rectified Flow NSF-HiFiGAN vocoder...', flush=True)
        with tqdm(total=56600485, unit='B', unit_scale=True, desc='NSF-HiFiGAN vocoder') as progress:
            download_file(VOCODER_URL, str(destination), progress, expected_sha256=VOCODER_SHA256)
    return str(destination)
