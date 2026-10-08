import hashlib
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm


DOWNLOAD_TIMEOUT = (10, 120)
DOWNLOAD_HEADERS = {"Accept-Encoding": "identity"}
FIREREDVAD_REVISION = "7990aaccc6b7aec1e527743bd30201f2c4a03b8c"
ECAPA_TDNN_REVISION = "4904fda6c5da208998321e3ea75e24320bab8aad"

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

pretraineds_hifigan_list = [
    (
        "pretrained_v1/",
        [
            "f0D40k.pth",
            "f0G40k.pth",
        ],
    ),
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]
pretraineds_refinegan_list = [
    (
        "refinegan/",
        [
            "f0D24k.pth",
            "f0G24k.pth",
            "f0D32k.pth",
            "f0G32k.pth",
        ],
    ),
]
models_list = [
    ("predictors/", ["rmvpe.pt"]),
    ("FireRedVAD/AED/", ["model.pth.tar", "cmvn.ark"]),
    ("ecapa_tdnn/", ["pretrain.model"]),
]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v1/": "models/pretraineds/hifi-gan/v1/",
    "pretrained_v2/": "models/pretraineds/hifi-gan/",
    "refinegan/": "models/pretraineds/refinegan/",
    "embedders/contentvec/": "models/embedders/contentvec/",
    "predictors/": "models/predictors/",
    "FireRedVAD/AED/": "models/pretraineds/FireRedVAD/AED/",
    "ecapa_tdnn/": "models/pretraineds/ecapa_tdnn/",
}

remote_base_mapping = {
    "pretrained_v1/": (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/"
    ),
    "FireRedVAD/AED/": (
        "https://huggingface.co/FireRedTeam/FireRedVAD/resolve/"
        f"{FIREREDVAD_REVISION}/AED/"
    ),
    "ecapa_tdnn/": (
        "https://raw.githubusercontent.com/TaoRuijie/ECAPA-TDNN/"
        f"{ECAPA_TDNN_REVISION}/exps/"
    ),
}

archive_mapping = {
    ("predictors/", "rmvpe.pt"): (
        "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip",
        "model.pt",
    ),
}

CONTENTVEC_SHA256 = "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e"
CONTENTVEC_CONFIG_SHA256 = "2ddde063b795d38d9051a7215a092fecf4cfe148b54251e38de51d88d356898b"
CONTENTVEC_PREPROCESSOR_PATH = (
    "models/embedders/contentvec/preprocessor_config.json"
)


expected_sha256_mapping = {
    ("predictors/", "rmvpe.pt"): (
        "19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2"
    ),
    ("embedders/contentvec/", "pytorch_model.bin"): CONTENTVEC_SHA256,
    ("embedders/contentvec/", "config.json"): CONTENTVEC_CONFIG_SHA256,
    ("FireRedVAD/AED/", "cmvn.ark"): (
        "c87f6f13edf0f0ec7535ddfc9cc3387d9268cb234b70182d566c5e2edf3ca473"
    ),
    ("FireRedVAD/AED/", "model.pth.tar"): (
        "ad08a4e05b58ca328154e158d24cff57a2fe796ceabb63bb701544c3f7d4f7ad"
    ),
    ("ecapa_tdnn/", "pretrain.model"): (
        "a435faa17622a6bf99db8ec6fb51aafe41eb5396d3d1b837eed7e74af8c5d64f"
    ),
}


def get_download_url(remote_folder, file):
    archive = archive_mapping.get((remote_folder, file))
    if archive is not None:
        return archive[0]
    remote_base = remote_base_mapping.get(remote_folder)
    if remote_base is not None:
        return f"{remote_base}{file}"
    return f"{url_base}/{remote_folder}{file}"


def _sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_is_valid(remote_folder, file, destination_path):
    if not os.path.isfile(destination_path) or os.path.getsize(destination_path) == 0:
        return False
    expected_sha256 = expected_sha256_mapping.get((remote_folder, file))
    return expected_sha256 is None or _sha256(destination_path) == expected_sha256


def _files_need_download(file_list):
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not _file_is_valid(remote_folder, file, destination_path):
                return True
    return False


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files that are missing or fail validation.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not _file_is_valid(remote_folder, file, destination_path):
                url = get_download_url(remote_folder, file)
                with requests.head(
                    url,
                    allow_redirects=True,
                    headers=DOWNLOAD_HEADERS,
                    timeout=DOWNLOAD_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    total_size += int(response.headers.get("content-length", 0))
    return total_size


def download_file(url, destination_path, global_bar, expected_sha256=None, archive_member=None):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    temporary_path = None
    try:
        with requests.get(
            url,
            stream=True,
            headers=DOWNLOAD_HEADERS,
            timeout=DOWNLOAD_TIMEOUT,
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            expected_size = int(content_length) if content_length is not None else None
            bytes_written = 0

            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=dir_name or ".",
                prefix=f".{os.path.basename(destination_path)}.",
                suffix=".part",
            ) as temporary_file:
                temporary_path = temporary_file.name
                for data in response.iter_content(1024 * 1024):
                    if not data:
                        continue
                    temporary_file.write(data)
                    bytes_written += len(data)
                    global_bar.update(len(data))

        if expected_size is not None and bytes_written != expected_size:
            raise IOError(
                f"Incomplete download for {destination_path}: "
                f"expected {expected_size} bytes, received {bytes_written}"
            )
        if archive_member is not None:
            extracted_path = f"{temporary_path}.member"
            with zipfile.ZipFile(temporary_path) as archive, archive.open(
                archive_member
            ) as source, open(extracted_path, "wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            os.remove(temporary_path)
            temporary_path = extracted_path
        if expected_sha256 is not None and _sha256(temporary_path) != expected_sha256:
            raise IOError(f"Checksum verification failed for {destination_path}")

        os.replace(temporary_path, destination_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not _file_is_valid(remote_folder, file, destination_path):
                    expected_sha256 = expected_sha256_mapping.get((remote_folder, file))
                    if (
                        remote_folder == "embedders/contentvec/"
                        and expected_sha256 is not None
                        and os.path.isfile(destination_path)
                        and os.path.getsize(destination_path) > 0
                    ):
                        if file == "pytorch_model.bin":
                            print(
                                "ContentVec checkpoint SHA-256 mismatch; "
                                "replacing it with Applio's checkpoint."
                            )
                        elif file == "config.json":
                            print(
                                "ContentVec config SHA-256 mismatch; "
                                "replacing it with Applio's config."
                            )
                    url = get_download_url(remote_folder, file)
                    futures.append(
                        executor.submit(
                            download_file,
                            url,
                            destination_path,
                            global_bar,
                            expected_sha256_mapping.get((remote_folder, file)),
                            archive_mapping.get((remote_folder, file), (None, None))[1],
                        )
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_hifigan)
    if pretraineds_hifigan:
        total_size += get_file_size_if_missing(pretraineds_refinegan_list)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Manage the download pipeline for different categories of files.
    """
    if os.path.isfile(CONTENTVEC_PREPROCESSOR_PATH):
        os.remove(CONTENTVEC_PREPROCESSOR_PATH)
        print(
            f"Removed legacy ContentVec preprocessor config: "
            f"{CONTENTVEC_PREPROCESSOR_PATH}"
        )
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
    )

    files_need_download = (
        (
            models
            and (
                _files_need_download(models_list)
                or _files_need_download(embedders_list)
            )
        )
        or (exe and os.name == "nt" and _files_need_download(executables_list))
        or (
            pretraineds_hifigan
            and (
                _files_need_download(pretraineds_hifigan_list)
                or _files_need_download(pretraineds_refinegan_list)
            )
        )
    )

    if files_need_download:
        with tqdm(
            total=total_size, unit="iB", unit_scale=True, desc="Downloading all files"
        ) as global_bar:
            if models:
                download_mapping_files(models_list, global_bar)
                download_mapping_files(embedders_list, global_bar)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, global_bar)
                else:
                    print("No executables needed")
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, global_bar)
                download_mapping_files(pretraineds_refinegan_list, global_bar)
    else:
        pass
