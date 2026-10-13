import ast
from textwrap import indent

import torch


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
    for optimizer in ('optim_g', 'optim_d'):
        start = f'    {optimizer} = torch.optim.AdamW(\n'
        if source.count(start) != 1:
            raise RuntimeError('This Beatrice Trainer version has an unsupported optimizer layout.')
        begin = source.index(start)
        finish = source.index('    )\n', begin)
        source = source[:finish] + '        fused=device.type == "cuda",\n' + source[finish:]
    ast.parse(source)
    return source
