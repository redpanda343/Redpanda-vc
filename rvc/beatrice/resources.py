from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'fierce-cats/beatrice-trainer'
REVISION = 'f34836de014b86956096878aecb8d3b17feaaa0b'
TRAINER_ROOT = ROOT / 'rvc' / 'models' / 'beatrice' / 'beatrice-trainer'
REQUIRED_FILES = (
    'beatrice_trainer/__main__.py',
    'assets/default_config.json',
    'assets/images/noimage.png',
    'assets/pretrained/104_3_checkpoint_00300000.pt',
    'assets/pretrained/122_checkpoint_03000000.pt',
    'assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz',
)


def trainer_ready():
    marker = TRAINER_ROOT / '.redpanda-revision'
    return (marker.is_file() and marker.read_text(encoding='utf-8').strip() == REVISION
            and all((TRAINER_ROOT / name).is_file() for name in REQUIRED_FILES))


def ensure_trainer():
    if trainer_ready():
        return TRAINER_ROOT
    from huggingface_hub import snapshot_download

    print(f'Downloading Beatrice Trainer {REVISION[:7]} with its pretrained models from {REPOSITORY} '
          '(about 440 MB)...', flush=True)
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=TRAINER_ROOT)
    (TRAINER_ROOT / '.redpanda-revision').write_text(REVISION, encoding='utf-8')
    return TRAINER_ROOT
