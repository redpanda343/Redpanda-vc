from rectified_flow.resources import ROOT, _install_bundle

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


def pretrained_path():
    _install_bundle(PRETRAINED_DIRECTORY, PRETRAINED_URL, PRETRAINED_SHA256, 377291683, '', PRETRAINED_FILES,
                    PRETRAINED_FILENAME, FINETUNE_TITLE)
    return str(PRETRAINED_DIRECTORY / PRETRAINED_FILENAME)
