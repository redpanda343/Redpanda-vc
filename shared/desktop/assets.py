"""Run the existing weight installer in the background."""

from threading import Lock


class Assets:
    def __init__(self, emit):
        self.emit = emit
        self.locks = {key: Lock() for key in ("hifigan", "rmvpe", "vad")}
        self.prerequisites = set()

    def ensure(self, *, pretrain=False, rmvpe=False, vad=False):
        from tqdm import tqdm

        from shared.tools.prerequisites_download import (
            _files_need_download, download_mapping_files, get_file_size_if_missing, pretraineds_hifigan_list,
        )

        for requested, category, files in (
            (pretrain, "hifigan", pretraineds_hifigan_list),
            (rmvpe, "rmvpe", [("predictors/", ["rmvpe_rvc.pt"])]),
            (vad, "vad", [("FireRedVAD/AED/", ["model.pth.tar", "cmvn.ark"])]),
        ):
            if not requested:
                continue
            with self.locks[category]:
                if category in self.prerequisites:
                    continue
                if _files_need_download(files):
                    size = get_file_size_if_missing(files)
                    with tqdm(total=size, unit="B", unit_scale=True, desc=category) as progress:
                        download_mapping_files(files, progress)
                self.prerequisites.add(category)

    def startup(self):
        try:
            self.ensure(pretrain=True)
        except Exception:
            pass
