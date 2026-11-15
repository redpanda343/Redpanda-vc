import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def _get_xla_grus(self):
        pairs = self.__dict__.get("_xla_grus")
        if pairs is None:
            pairs = []
            for layer_idx in range(self.gru.num_layers):
                in_dim = self.gru.input_size if layer_idx == 0 else self.gru.hidden_size * 2
                fwd = nn.GRU(in_dim, self.gru.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
                rev = nn.GRU(in_dim, self.gru.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
                fwd.eval()
                rev.eval()
                pairs.append((fwd, rev))
            self.__dict__["_xla_grus"] = pairs
        for layer_idx, (fwd, rev) in enumerate(pairs):
            fwd.training = self.training
            rev.training = self.training
            fwd.weight_ih_l0 = getattr(self.gru, f"weight_ih_l{layer_idx}")
            fwd.weight_hh_l0 = getattr(self.gru, f"weight_hh_l{layer_idx}")
            fwd.bias_ih_l0 = getattr(self.gru, f"bias_ih_l{layer_idx}")
            fwd.bias_hh_l0 = getattr(self.gru, f"bias_ih_l{layer_idx}")
            fwd._flat_weights = [fwd.weight_ih_l0, fwd.weight_hh_l0, fwd.bias_ih_l0, fwd.bias_hh_l0]
            rev.weight_ih_l0 = getattr(self.gru, f"weight_ih_l{layer_idx}_reverse")
            rev.weight_hh_l0 = getattr(self.gru, f"weight_hh_l{layer_idx}_reverse")
            rev.bias_ih_l0 = getattr(self.gru, f"bias_ih_l{layer_idx}_reverse")
            rev.bias_hh_l0 = getattr(self.gru, f"bias_hh_l{layer_idx}_reverse")
            rev._flat_weights = [rev.weight_ih_l0, rev.weight_hh_l0, rev.bias_ih_l0, rev.bias_hh_l0]
        return pairs

    def forward(self, x):
        if x.device.type == "xla":
            out = x
            for fwd, rev in self._get_xla_grus():
                out_fwd = fwd(out)[0]
                out_rev = torch.flip(rev(torch.flip(out, dims=[1]))[0], dims=[1])
                out = torch.cat([out_fwd, out_rev], dim=-1)
            return out
        return self.gru(x)[0]
