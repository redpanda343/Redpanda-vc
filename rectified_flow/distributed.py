import torch


def parse_devices(value):
    if str(value).strip().lower() == 'auto':
        return [f'cuda:{index}' for index in range(torch.cuda.device_count())] or ['cpu']
    items = [item.strip().lower() for item in str(value).split(',')]
    if items == ['cpu']:
        return items
    if not items or any(not item.startswith('cuda:') or not item[5:].isdigit() for item in items):
        raise ValueError('Use cpu, cuda:0, or a comma-separated GPU list such as cuda:0,cuda:1.')
    devices = [f'cuda:{int(item[5:])}' for item in items]
    if len(devices) != len(set(devices)):
        raise ValueError('Each selected GPU must be unique.')
    return devices
