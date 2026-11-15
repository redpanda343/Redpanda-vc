import struct
import numpy as np
import torch
import torch.nn.functional as F


_DTYPES = {
    1: np.float32,
    6: np.int32,
    7: np.int64,
    11: np.float64,
}


def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7


def _iter_fields(buf, start=0, end=None):
    if end is None:
        end = len(buf)
    pos = start
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            val, pos = _read_varint(buf, pos)
        elif wire_type == 1:
            val = buf[pos : pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            val = (pos, pos + length)
            pos += length
        elif wire_type == 5:
            val = buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire_type}")
        yield field_num, wire_type, val


def _load_onnx_initializers(model_path):
    with open(model_path, "rb") as f:
        data = f.read()

    graph_span = None
    for field_num, _, val in _iter_fields(data):
        if field_num == 7:
            graph_span = val
            break
    if graph_span is None:
        raise ValueError(f"No ONNX graph found in {model_path}")

    initializers = {}
    for field_num, _, val in _iter_fields(data, graph_span[0], graph_span[1]):
        if field_num != 5:
            continue
        name = ""
        dims = []
        data_type = 0
        raw_data = None
        floats = []
        int32s = []
        int64s = []
        doubles = []
        for f2, w2, v2 in _iter_fields(data, val[0], val[1]):
            if f2 == 1:
                dims.append(v2)
            elif f2 == 2:
                data_type = v2
            elif f2 == 4:
                if w2 == 2:
                    floats.extend(np.frombuffer(data[v2[0] : v2[1]], dtype="<f4"))
                else:
                    floats.append(struct.unpack("<f", v2)[0])
            elif f2 == 5:
                if w2 == 2:
                    int32s.extend(np.frombuffer(data[v2[0] : v2[1]], dtype="<i4"))
                else:
                    int32s.append(v2 if v2 < (1 << 31) else v2 - (1 << 32))
            elif f2 == 7:
                if w2 == 2:
                    pos = v2[0]
                    while pos < v2[1]:
                        v, pos = _read_varint(data, pos)
                        int64s.append(v if v < (1 << 63) else v - (1 << 64))
                else:
                    int64s.append(v2 if v2 < (1 << 63) else v2 - (1 << 64))
            elif f2 == 8:
                name = data[v2[0] : v2[1]].decode("utf-8")
            elif f2 == 9:
                raw_data = data[v2[0] : v2[1]]
            elif f2 == 10:
                if w2 == 2:
                    doubles.extend(np.frombuffer(data[v2[0] : v2[1]], dtype="<f8"))
                else:
                    doubles.append(struct.unpack("<d", v2)[0])

        if raw_data is not None:
            arr = np.frombuffer(raw_data, dtype=_DTYPES[data_type]).reshape(dims).copy()
        elif floats:
            arr = np.array(floats, dtype=np.float32).reshape(dims)
        elif int64s:
            arr = np.array(int64s, dtype=np.int64).reshape(dims)
        elif int32s:
            arr = np.array(int32s, dtype=np.int32).reshape(dims)
        elif doubles:
            arr = np.array(doubles, dtype=np.float64).reshape(dims)
        else:
            continue
        initializers[name] = arr

    return initializers


class SwiftF0Torch:
    def __init__(self, model_path, device):
        self.device = torch.device(device)
        inits = _load_onnx_initializers(model_path)
        self.eps = float(inits["v129"])
        self.scales = []
        for n_fft, win_k, idx_k, val_k in (
            (1024, "v41", "v123", "v124"),
            (2048, "v161", "v241", "v242"),
            (512, "v262", "v342", "v343"),
        ):
            win = torch.from_numpy(inits[win_k]).float()
            fb = np.zeros((n_fft // 2 + 1) * 128, dtype=np.float32)
            fb[inits[idx_k]] = inits[val_k]
            fb_t = torch.from_numpy(fb.reshape(n_fft // 2 + 1, 128)).float()
            self.scales.append((n_fft, win, fb_t))

        def to_dev(key):
            return torch.from_numpy(inits[key]).float().to(self.device)

        self.stem = (to_dev("v353"), to_dev("v354"))
        self.blocks = [
            (to_dev("v357"), to_dev("v358"), to_dev("v361"), to_dev("v362"), (3, 1), (1, 1), True),
            (to_dev("v366"), to_dev("v367"), to_dev("v370"), to_dev("v371"), (51, 0), (17, 1), False),
            (to_dev("v375"), to_dev("v376"), to_dev("v379"), to_dev("v380"), (54, 0), (27, 1), False),
            (to_dev("v384"), to_dev("v385"), to_dev("v388"), to_dev("v389"), (40, 0), (40, 1), False),
            (to_dev("v393"), to_dev("v394"), to_dev("v397"), to_dev("v398"), (48, 0), (48, 1), False),
            (to_dev("v402"), to_dev("v403"), to_dev("v406"), to_dev("v407"), (6, 1), (2, 1), True),
        ]
        self.head_conv = (to_dev("v413"), to_dev("v414"))
        self.voicing_conv2d = (to_dev("v417"), to_dev("v418"))
        self.pitch_conv2d = (to_dev("v425"), to_dev("v426"))
        self.voicing_conv1d_1 = (to_dev("v430"), to_dev("v431"))
        self.voicing_conv1d_2 = (to_dev("v434"), to_dev("v435"))

        self.cents_hz = inits["v438"].astype(np.float32)
        self.log_hz = inits["v455"].astype(np.float64)
        self.offsets = np.array([-1, 0, 1], dtype=np.int64)

    def _compute_log_mel(self, audio):
        t_len = int(audio.shape[1])
        n_frames = max(t_len // 256, 1)
        audio_t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        mels = []
        for n_fft, win, fb in self.scales:
            length = max(t_len, 256)
            padded = F.pad(audio_t, (0, length - t_len))
            pad_right = n_fft // 2 + ((length + 255) // 256) * 256 - length
            padded = F.pad(padded, (n_fft // 2, pad_right)).reshape(-1, 256)
            frames = torch.cat(
                [padded[i : i + n_frames] for i in range(n_fft // 256)], dim=1
            )
            mag = torch.fft.rfft(frames * win, dim=-1).abs()
            mel = torch.log(mag @ fb + self.eps).transpose(0, 1).unsqueeze(0).unsqueeze(0)
            mels.append(mel)
        return torch.cat(mels, dim=1), n_frames

    @torch.no_grad()
    def run(self, audio, fmin, fmax):
        if audio.ndim == 1:
            audio = audio[None, :]
        mel_spec, n_frames = self._compute_log_mel(audio)
        is_xla = self.device.type == "xla"
        if is_xla:
            pad_frames = ((n_frames + 63) // 64) * 64
            if pad_frames > n_frames:
                mel_spec = F.pad(mel_spec, (0, pad_frames - n_frames))
            mask = torch.zeros((1, 1, 1, pad_frames), dtype=torch.float32)
            mask[..., :n_frames] = 1.0
            mask = mask.to(self.device)
        else:
            mask = None

        x_in = mel_spec.to(self.device)
        w0, b0 = self.stem
        x = F.relu(F.conv2d(x_in, w0, b0, padding=(1, 1)))
        if mask is not None:
            x = x * mask

        for w1, b1, w2, b2, pad, dil, temporal in self.blocks:
            if temporal and mask is not None:
                x = x * mask
            h = F.relu(F.conv2d(x, w1, b1, padding=pad, dilation=dil))
            if temporal and mask is not None:
                h = h * mask
            h = F.conv2d(h, w2, b2, padding=pad, dilation=dil)
            x = F.relu(x + h)

        if mask is not None:
            x = x * mask
        x_slice = x[:, :, :96, :]
        wh, bh = self.head_conv
        feat = F.relu(F.conv2d(F.pad(x_slice, (1, 1, 1, 0)), wh, bh))
        if mask is not None:
            feat = feat * mask

        wp, bp = self.pitch_conv2d
        pitch_logits = F.conv2d(feat, wp, bp).squeeze(1).transpose(1, 2)

        wv, bv = self.voicing_conv2d
        v_max = F.conv2d(feat, wv, bv, padding=(2, 0)).amax(dim=2)
        v_mean = x_in.mean(dim=(1, 2), keepdim=True).squeeze(1)
        v_cat = torch.cat([v_max, v_mean], dim=1)
        if mask is not None:
            v_cat = v_cat * mask.squeeze(2)
        wc1, bc1 = self.voicing_conv1d_1
        wc2, bc2 = self.voicing_conv1d_2
        v_hidden = F.relu(F.conv1d(v_cat, wc1, bc1, padding=1))
        voicing_logits = F.conv1d(v_hidden, wc2, bc2).squeeze(1)

        logits = pitch_logits[0].float().cpu().numpy()[:n_frames]
        v_logits = voicing_logits[0].float().cpu().numpy()[:n_frames]

        in_range = (self.cents_hz >= float(fmin)) & (self.cents_hz <= float(fmax))
        masked_logits = np.where(in_range[None, :], logits, -np.inf).astype(np.float32)
        center = np.argmax(masked_logits, axis=-1, keepdims=True)
        indices = center + self.offsets
        padded_logits = np.pad(
            masked_logits, ((0, 0), (1, 1)), mode="constant", constant_values=-np.inf
        )
        local_logits = np.take_along_axis(padded_logits, indices + 1, axis=-1).astype(
            np.float64
        )
        local_max = np.max(local_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(
            np.where(np.isfinite(local_max), local_logits - local_max, -np.inf)
        )
        denom = np.sum(exp_logits, axis=-1, keepdims=True)
        weights = np.divide(
            exp_logits, denom, out=np.zeros_like(exp_logits), where=denom > 0
        )
        clipped_indices = np.clip(indices, 0, len(self.log_hz) - 1)
        log_pitch = np.sum(weights * self.log_hz[clipped_indices], axis=-1)
        pitch = np.exp(log_pitch).astype(np.float32)

        raw_argmax = np.argmax(logits, axis=-1)
        valid_peak = in_range[raw_argmax]
        voicing_prob = 1.0 / (1.0 + np.exp(-v_logits.astype(np.float64)))
        confidence = np.where(valid_peak, voicing_prob, 0.0).astype(np.float32)
        return pitch, confidence
