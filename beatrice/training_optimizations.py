import ast
from textwrap import indent

import torch

FAST_AUGMENTATION_ANCHOR = '# %%\nAUDIO_FILE_SUFFIXES = {\n'
FAST_AUGMENTATION_SOURCE = '''import soundfile as _soundfile

_FAST_FULL_SECONDS = 12.0
_FAST_BEFORE_SECONDS = 1.5
_FAST_AFTER_SECONDS = 0.5
_FAST_FORMANT_MARGIN = 4096
_FAST_FORMANT_HOP = 256
_FAST_RESAMPLE_MARGIN = 512
_FAST_FORMANT_SHIFTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
_random_formant_shift_full = random_formant_shift
_wav_dataset_getitem_full = WavDataset.__getitem__


def random_formant_shift(
    wav: torch.Tensor,
    sample_rate: int,
    formant_shift_semitone_min: float = -3.0,
    formant_shift_semitone_max: float = 3.0,
) -> torch.Tensor:
    (nonzero,) = wav[0].nonzero(as_tuple=True)
    begin, end = 0, wav.size(1)
    if nonzero.numel():
        begin = max(0, (nonzero[0].item() - _FAST_FORMANT_MARGIN) // _FAST_FORMANT_HOP * _FAST_FORMANT_HOP)
        end = min(end, nonzero[-1].item() + 1 + _FAST_FORMANT_MARGIN)
    if begin == 0 and end == wav.size(1):
        return _random_formant_shift_full(
            wav, sample_rate, formant_shift_semitone_min, formant_shift_semitone_max
        )
    out = wav.new_zeros(1, wav.size(1))
    out[:, begin:end] = _random_formant_shift_full(
        wav[:, begin:end], sample_rate, formant_shift_semitone_min, formant_shift_semitone_max
    )
    return out


def _wav_dataset_long_file(self, file):
    cache = self.__dict__.setdefault("_fast_long_files", {})
    if file not in cache:
        info = _soundfile.info(str(file))
        cache[file] = None
        if info.frames / info.samplerate > _FAST_FULL_SECONDS:
            wav, sample_rate = torchaudio.load(file, backend="soundfile")
            audible = (wav.abs().amax(0) > 0).numpy().astype(np.int8)
            edges = np.flatnonzero(np.diff(np.concatenate(([0], audible, [0]))))
            starts, ends = edges[0::2], edges[1::2]
            if starts.size:
                cache[file] = (
                    sample_rate,
                    wav.size(1),
                    wav.size(0),
                    wav.abs().max().item(),
                    starts,
                    np.cumsum(ends - starts),
                )
    return cache[file]


def _wav_dataset_select(self, clean_wav, noisy_wav):
    length = self.wav_length // self.out_hop_length
    (voiced,) = clean_wav.nonzero(as_tuple=True)
    assert voiced.numel() != 0
    center = voiced[torch.randint(0, voiced.numel(), ()).item()].item()
    slice_start = center - self.segment_length * self.out_hop_length // 2
    assert slice_start >= 0
    r = torch.randint(0, length - self.segment_length + 1, ()).item()
    offset = slice_start - r * self.out_hop_length
    offset_in_sample_rate = int(round(offset * self.in_sample_rate / self.out_sample_rate))
    return (
        clean_wav[offset : offset + self.wav_length],
        noisy_wav[offset_in_sample_rate : offset_in_sample_rate + length * self.in_hop_length],
        r,
    )


def _wav_dataset_crop(self, file, speaker_id, info):
    sample_rate, frames, channels, peak, starts, cumulative = info
    length = self.wav_length // self.out_hop_length
    channel = torch.randint(0, channels, ()).item() if channels != 1 else 0
    formant_shift = _FAST_FORMANT_SHIFTS[torch.randint(0, len(_FAST_FORMANT_SHIFTS), ()).item()]
    fraction = Fraction(
        sample_rate / self.out_sample_rate * 2.0 ** (formant_shift / 12.0)
    ).limit_denominator(300)
    numerator, denominator = fraction.numerator, fraction.denominator
    k = torch.randint(0, int(cumulative[-1]), ()).item()
    run = int(np.searchsorted(cumulative, k, side="right"))
    raw_center = int(starts[run]) + k - (int(cumulative[run - 1]) if run else 0)
    center = raw_center * denominator // numerator
    slice_start = center - self.segment_length * self.out_hop_length // 2
    r = torch.randint(0, length - self.segment_length + 1, ()).item()
    offset = slice_start - r * self.out_hop_length
    in_step = self.out_sample_rate // math.gcd(self.out_sample_rate, self.in_sample_rate)
    step = math.lcm(denominator, in_step)
    begin = (offset - int(self.out_sample_rate * _FAST_BEFORE_SECONDS)) // step * step
    finish = offset + self.wav_length + int(self.out_sample_rate * _FAST_AFTER_SECONDS)
    blocks = -(-_FAST_RESAMPLE_MARGIN // numerator)
    raw_begin = begin // denominator * numerator - blocks * numerator
    raw_end = -(-finish * numerator // denominator) + blocks * numerator
    read_begin, read_end = max(0, raw_begin), min(frames, raw_end)
    raw = torch.zeros(1, raw_end - raw_begin)
    if read_end > read_begin:
        wav, _ = torchaudio.load(
            file, frame_offset=read_begin, num_frames=read_end - read_begin, backend="soundfile"
        )
        raw[:, read_begin - raw_begin : read_end - raw_begin] = wav[channel : channel + 1]
    clean_wav = get_resampler(numerator, denominator)(raw)
    clean_wav = clean_wav[:, blocks * denominator : blocks * denominator + finish - begin]
    clean_wav_16k = get_resampler(self.out_sample_rate, self.in_sample_rate)(clean_wav)
    noisy_wav_16k = augment_audio(
        clean_wav_16k,
        self.in_sample_rate,
        self.noise_files,
        self.ir_files,
        self.augmentation_snr_candidates,
        self.augmentation_formant_shift_probability,
        self.augmentation_formant_shift_semitone_min,
        self.augmentation_formant_shift_semitone_max,
        self.augmentation_reverb_probability,
        self.augmentation_lpf_probability,
        self.augmentation_lpf_cutoff_freq_candidates,
    )
    clean_wav = clean_wav.squeeze_(0)
    noisy_wav_16k = noisy_wav_16k.squeeze_(0)
    amplitude = torch.rand(()).item() * 0.899 + 0.1
    factor = amplitude / max(peak, 1e-5)
    clean_wav *= factor
    noisy_wav_16k *= factor
    while noisy_wav_16k.abs().max() >= 1.0:
        clean_wav *= 0.5
        noisy_wav_16k *= 0.5
    local = offset - begin
    local_in = int(round(offset * self.in_sample_rate / self.out_sample_rate)) - (
        begin * self.in_sample_rate // self.out_sample_rate
    )
    return (
        clean_wav[local : local + self.wav_length],
        noisy_wav_16k[local_in : local_in + length * self.in_hop_length],
        r,
        speaker_id,
        formant_shift,
    )


def _wav_dataset_getitem(self, index):
    file, speaker_id = self.audio_files[index]
    info = _wav_dataset_long_file(self, file)
    if info is not None:
        return _wav_dataset_crop(self, file, speaker_id, info)
    clean_wav, noisy_wav, speaker_id, formant_shift = _wav_dataset_getitem_full(self, index)
    clean_wav, noisy_wav, slice_start = _wav_dataset_select(self, clean_wav, noisy_wav)
    return clean_wav, noisy_wav, slice_start, speaker_id, formant_shift


def _wav_dataset_collate(self, batch):
    clean_wavs, noisy_wavs, slice_starts, speaker_ids, formant_shifts = zip(*batch)
    return (
        torch.stack(clean_wavs),
        torch.stack(noisy_wavs),
        torch.tensor(slice_starts),
        torch.tensor(speaker_ids),
        torch.tensor(formant_shifts),
    )


_wav_dataset_getitem.__name__ = "__getitem__"
_wav_dataset_getitem.__qualname__ = "WavDataset.__getitem__"
_wav_dataset_collate.__name__ = "collate"
_wav_dataset_collate.__qualname__ = "WavDataset.collate"
WavDataset.__getitem__ = _wav_dataset_getitem
WavDataset.collate = _wav_dataset_collate


'''
CODEBOOK_INPUT_SOURCE = '''            yield wav[:, None, :]
'''
CODEBOOK_INPUT_REPLACEMENT = '''            chunks = max(1, -(-wav.size(1) // (30 * h.in_sample_rate)))
            size = -(-wav.size(1) // chunks)
            size = -(-size // 640) * 640
            for start in range(0, wav.size(1), size):
                yield wav[:, None, start : start + size]
'''
SMALL_DATASET_SOURCE = '''        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
'''
SMALL_DATASET_REPLACEMENT = '''        shuffle=len(training_dataset) >= h.batch_size,
        sampler=None
        if len(training_dataset) >= h.batch_size
        else torch.utils.data.RandomSampler(
            training_dataset, replacement=True, num_samples=h.batch_size * 64
        ),
        batch_size=h.batch_size,
'''


def optimizer_state_for_load(optimizer, state):
    groups = [dict(group, fused=bool(optimizer.defaults.get('fused')), foreach=None)
              for group in state['param_groups']]
    return dict(state, param_groups=groups)


def compressed_optimizer_state_dict(optimizer):
    original = optimizer.state_dict()
    states = {}
    for parameter, values in original['state'].items():
        states[parameter] = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().to(device='cpu', dtype=torch.bfloat16)
                if not torch.isfinite(value).all():
                    raise FloatingPointError('The Beatrice optimizer contains nonfinite values.')
            states[parameter][name] = value
    return dict(original, state=states)


def guard_loss_stats(source):
    lines = source.splitlines(keepends=True)
    assignments = []
    for definition in ast.parse(source).body:
        if not isinstance(definition, ast.ClassDef) or definition.name not in ('ConverterNetwork',
                                                                                'MultiPeriodDiscriminator'):
            continue
        function = next(node for node in definition.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'forward_and_compute_loss')
        for node in ast.walk(function):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'item'
                    and any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                            and target.value.id == 'stats' for target in node.targets)):
                assignments.append(node)
    if len(assignments) != 7:
        raise RuntimeError('This Beatrice Trainer version has an unsupported loss statistics layout.')
    for node in sorted(assignments, key=lambda item: item.lineno, reverse=True):
        block = ''.join(lines[node.lineno - 1:node.end_lineno])
        lines[node.lineno - 1:node.end_lineno] = [' ' * node.col_offset + 'if record_stats:\n', indent(block, '    ')]
    return ''.join(lines)


def optimize_trainer_source(source):
    patches = (
        ('from beatrice.compat import patch_torchaudio\n',
         'from beatrice.compat import patch_torchaudio\n'
         'from beatrice.training_optimizations import compressed_optimizer_state_dict, optimizer_state_for_load\n'),
        ('        enable_loss_ap: bool = False,\n    ) -> tuple[',
         '        enable_loss_ap: bool = False,\n        record_stats: bool = True,\n    ) -> tuple['),
        ('        self, y: torch.Tensor, y_hat: torch.Tensor\n'
         '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:',
         '        self, y: torch.Tensor, y_hat: torch.Tensor, record_stats: bool = True\n'
         '    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:'),
        ('\n                    enable_loss_ap=h.grad_weight_ap != 0.0,\n',
         '\n                    enable_loss_ap=h.grad_weight_ap != 0.0,\n'
         '                    record_stats=h.record_metrics,\n'),
        ('net_d.forward_and_compute_loss(y, y_hat)',
         'net_d.forward_and_compute_loss(y, y_hat, record_stats=h.record_metrics)'),
        ('            loss_discriminator = loss_discriminator.item()\n',
         '            loss_discriminator = loss_discriminator.item() if h.record_metrics else None\n'),
        ('            loss_loudness = loss_loudness.item()\n'
         '            loss_mel = loss_mel.item()\n'
         '            loss_adv = loss_adv.item()\n'
         '            loss_fm = loss_fm.item()\n'
         '            if h.grad_weight_ap:\n'
         '                loss_ap = loss_ap.item()\n',
         '            if h.record_metrics:\n'
         '                loss_loudness = loss_loudness.item()\n'
         '                loss_mel = loss_mel.item()\n'
         '                loss_adv = loss_adv.item()\n'
         '                loss_fm = loss_fm.item()\n'
         '                if h.grad_weight_ap:\n'
         '                    loss_ap = loss_ap.item()\n'
         '            else:\n'
         '                loss_loudness = loss_mel = loss_adv = loss_fm = loss_ap = None\n'),
        ('            if (iteration + 1) % 1000 == 0 or iteration == 0:\n',
         '            if h.record_metrics and ((iteration + 1) % 1000 == 0 or iteration == 0):\n'),
        ('        max_n_test_files = 1000\n', '        max_n_test_files = 1000 if h.record_metrics else 12\n'),
        ('    quality_tester = QualityTester().eval().to(device)\n',
         '    quality_tester = QualityTester().eval().to(device) if h.record_metrics else None\n'),
        ('                        converted = resample_to_in_sample_rate(converted)\n'
         '                        quality = quality_tester.test(converted, source_wav)\n'
         '                        for metric_name, values in quality.items():\n'
         '                            dict_qualities_all[metric_name].extend(values)\n',
         '                        if quality_tester is not None:\n'
         '                            converted = resample_to_in_sample_rate(converted)\n'
         '                            quality = quality_tester.test(converted, source_wav)\n'
         '                            for metric_name, values in quality.items():\n'
         '                                dict_qualities_all[metric_name].extend(values)\n'),
        ('get_compressed_optimizer_state_dict(optim_g)', 'compressed_optimizer_state_dict(optim_g)'),
        ('get_compressed_optimizer_state_dict(optim_d)', 'compressed_optimizer_state_dict(optim_d)'),
        ('get_decompressed_optimizer_state_dict(checkpoint["optim_g"])',
         'optimizer_state_for_load(optim_g, get_decompressed_optimizer_state_dict(checkpoint["optim_g"]))'),
        ('get_decompressed_optimizer_state_dict(checkpoint["optim_d"])',
         'optimizer_state_for_load(optim_d, get_decompressed_optimizer_state_dict(checkpoint["optim_d"]))'),
        ('                with gzip.open(checkpoint_file_save, "wb") as f:\n',
         '                with gzip.open(checkpoint_file_save, "wb", compresslevel=1) as f:\n'),
        ('                                k: v.half() for k, v in net_d.state_dict().items()\n',
         '                                k: v.detach().to(device="cpu", dtype=torch.float16)\n'
         '                                for k, v in net_d.state_dict().items()\n'),
    )
    for original, replacement in patches:
        if source.count(original) != 1:
            raise RuntimeError('This Beatrice Trainer version could not be optimized safely: '
                               f'unsupported source near {original.strip()[:70]}.')
        source = source.replace(original, replacement)
    norm_condition = '            if iteration % 5 == 0:\n'
    if source.count(norm_condition) != 2:
        raise RuntimeError('This Beatrice Trainer version has an unsupported gradient statistics layout.')
    source = source.replace(norm_condition, '            if h.record_metrics and iteration % 5 == 0:\n')
    start = '            dict_scalars["loss_g/loss_loudness"].append(loss_loudness)\n'
    end = '            if h.record_metrics and ((iteration + 1) % 1000 == 0 or iteration == 0):\n'
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError('This Beatrice Trainer version has an unsupported metrics layout.')
    begin, finish = source.index(start), source.index(end)
    source = (source[:begin] + '            if h.record_metrics:\n' + indent(source[begin:finish], '    ')
              + source[finish:])
    source = guard_loss_stats(source)
    if any(source.count(part) != 1 for part in (FAST_AUGMENTATION_ANCHOR, SMALL_DATASET_SOURCE,
                                                CODEBOOK_INPUT_SOURCE)):
        raise RuntimeError('This Beatrice Trainer version has an unsupported dataset layout.')
    source = source.replace(FAST_AUGMENTATION_ANCHOR, FAST_AUGMENTATION_SOURCE + FAST_AUGMENTATION_ANCHOR)
    source = source.replace(SMALL_DATASET_SOURCE, SMALL_DATASET_REPLACEMENT)
    source = source.replace(CODEBOOK_INPUT_SOURCE, CODEBOOK_INPUT_REPLACEMENT)
    for optimizer in ('optim_g', 'optim_d'):
        start = f'    {optimizer} = torch.optim.AdamW(\n'
        if source.count(start) != 1:
            raise RuntimeError('This Beatrice Trainer version has an unsupported optimizer layout.')
        begin = source.index(start)
        finish = source.index('    )\n', begin)
        source = source[:finish] + '        fused=device.type == "cuda",\n' + source[finish:]
    ast.parse(source)
    return source
