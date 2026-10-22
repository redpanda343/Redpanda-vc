from tqdm import tqdm

from rectified_flow.resources import ROOT, _download_lock, _install_bundle, _valid_bundle
from shared.tools.prerequisites_download import download_file

FINETUNE_SOURCE = 'pc_nsf_hifigan_2025_02'
FINETUNE_TITLE = 'OpenVPI PC-NSF-HiFiGAN 2025.02 (generator + discriminator)'
PRETRAINED_DIRECTORY = ROOT / 'models' / 'pretraineds' / 'nsf_hifigan' / 'pc_nsf_hifigan_2025.02'
PRETRAINED_FILENAME = 'pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt'
PRETRAINED_URL = ('https://github.com/openvpi/SingingVocoders/releases/download/v1.0.0/'
                  'pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip')
PRETRAINED_SHA256 = '636845f96066ab95c188c677713f4359ae53eae9669a7500d926c97174c3590c'
PRETRAINED_FILES = {
    PRETRAINED_FILENAME: '12884ead83ad17406098b2b32f12e613e78fcfc1436a87ac9d2d1bb0a27b21d6',
    'NOTICE.txt': 'fcfca845812eb462e82f4c19985fce2abc772af979226b9297287c97e9a97f65',
    'NOTICE.zh-CN.txt': '8983f1ac07a0b240b9d4c970c8081007b0f932e5e223d88f8f8aef0e6570ed81',
    'STATEMENTS.txt': 'fe208abd522fd9e77ee16063ca2b3d6466fe3a35db70578f615d28543dd62d85',
}


VOCOS_SOURCE = 'vocos_mel_24khz'
VOCOS_TITLE = 'Vocos 24 kHz backbone (charactr/vocos-mel-24khz)'
VOCOS_DIRECTORY = ROOT / 'models' / 'pretraineds' / 'vocos' / 'vocos_mel_24khz'
VOCOS_URL = 'https://huggingface.co/charactr/vocos-mel-24khz/resolve/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21/'
VOCOS_FILES = {
    'model.pth': ('pytorch_model.bin', '97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a', 54365991),
    'config.yaml': ('config.yaml', 'da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc', 461),
}


def pretrained_path():
    _install_bundle(PRETRAINED_DIRECTORY, PRETRAINED_URL, PRETRAINED_SHA256, 377291683, '', PRETRAINED_FILES,
                    PRETRAINED_FILENAME, FINETUNE_TITLE)
    return str(PRETRAINED_DIRECTORY / PRETRAINED_FILENAME)


def vocos_backbone_path():
    with _download_lock:
        if not _valid_bundle(VOCOS_DIRECTORY, {name: checksum for name, (_, checksum, _) in VOCOS_FILES.items()}):
            print(f'Downloading the {VOCOS_TITLE}...', flush=True)
            for name, (remote, checksum, size) in VOCOS_FILES.items():
                with tqdm(total=size, unit='B', unit_scale=True, desc=name) as progress:
                    download_file(VOCOS_URL + remote, str(VOCOS_DIRECTORY / name), progress, expected_sha256=checksum)
    return str(VOCOS_DIRECTORY / 'model.pth')
