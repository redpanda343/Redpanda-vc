import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'fierce-cats/beatrice-trainer'
REVISION = 'f34836de014b86956096878aecb8d3b17feaaa0b'
TRAINER_ROOT = ROOT / 'models' / 'beatrice' / 'beatrice-trainer'
REQUIRED_FILES = (
    'beatrice_trainer/__main__.py',
    'assets/default_config.json',
    'assets/images/noimage.png',
    'assets/pretrained/104_3_checkpoint_00300000.pt',
    'assets/pretrained/122_checkpoint_03000000.pt',
    'assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz',
)
DOWNLOAD_ATTEMPTS = 5
_rate_limit_lock = threading.Lock()
_resume_at = 0.0


def trainer_ready():
    marker = TRAINER_ROOT / '.redpanda-revision'
    return (marker.is_file() and marker.read_text(encoding='utf-8').strip() == REVISION
            and all((TRAINER_ROOT / name).is_file() for name in REQUIRED_FILES))


def hub_headers():
    from huggingface_hub import get_token

    token = get_token()
    return {'Authorization': f'Bearer {token}'} if token else {}


def repository_files(headers):
    import requests

    from shared.tools.prerequisites_download import DOWNLOAD_HEADERS, DOWNLOAD_TIMEOUT

    url = f'https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}'
    params = {'recursive': 'true'}
    files = []
    while url:
        with requests.get(url, params=params, headers={**DOWNLOAD_HEADERS, **headers},
                          timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            files.extend(entry for entry in response.json() if entry['type'] == 'file')
            url = response.links.get('next', {}).get('url')
        params = None
    return files


def missing_files(headers):
    return [entry for entry in repository_files(headers)
            if not (TRAINER_ROOT / entry['path']).is_file()
            or (TRAINER_ROOT / entry['path']).stat().st_size != entry['size']]


def wait_for_rate_limit(response, progress):
    global _resume_at
    match = re.search(r';t=(\d+)', response.headers.get('RateLimit', ''))
    with _rate_limit_lock:
        now = time.monotonic()
        if now >= _resume_at:
            _resume_at = now + (int(match.group(1)) if match else 60) + 1
            progress.write(f'Hugging Face rate limit reached. Resuming in {round(_resume_at - now)} s. '
                           'Set HF_TOKEN for higher limits.')
        resume_at = _resume_at
    time.sleep(max(0.0, resume_at - time.monotonic()))


def download_entry(entry, progress, headers):
    import requests

    from shared.tools.prerequisites_download import download_file

    url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{entry["path"]}'
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            return download_file(url, str(TRAINER_ROOT / entry['path']), progress,
                                 entry.get('lfs', {}).get('oid'), headers=headers)
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 429 or attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            wait_for_rate_limit(error.response, progress)


def ensure_trainer():
    if trainer_ready():
        return TRAINER_ROOT
    from tqdm import tqdm

    headers = hub_headers()
    missing = missing_files(headers)
    if missing:
        print(f'Downloading Beatrice Trainer {REVISION[:7]} with its pretrained models from {REPOSITORY} '
              '(about 440 MB)...', flush=True)
        with tqdm(total=sum(entry['size'] for entry in missing), unit='B', unit_scale=True,
                  desc='Beatrice Trainer') as progress, ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(download_entry, entry, progress, headers) for entry in missing]
            for future in futures:
                future.result()
    (TRAINER_ROOT / '.redpanda-revision').write_text(REVISION, encoding='utf-8')
    return TRAINER_ROOT
