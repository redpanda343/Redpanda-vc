from pathlib import Path

import torch
import yaml

from .nets import CascadedNet


def load_sep_model(model_path, device='cpu'):
    model_path = Path(model_path)
    with open(model_path.with_name('config.yaml'), 'r', encoding='utf-8') as config:
        args = yaml.safe_load(config)
    model = CascadedNet(args['n_fft'], args['hop_length'], args['n_out'], args['n_out_lstm'], True,
                        is_mono=args['is_mono'])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model
