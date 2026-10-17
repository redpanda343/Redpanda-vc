import torch
import torch.nn.functional as F

from rectified_flow.realtime import RealtimeGraph


def conv_geometry(config):
    stride = field = 1
    for kernel, step in zip(reversed(config.conv_kernel), reversed(config.conv_stride)):
        field = (field - 1) * step + kernel
        stride *= step
    return stride, field


class SpeechGroupNormConv(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.conv = layer.conv
        self.norm = layer.layer_norm
        self.activation = layer.activation
        self.kernel = self.conv.kernel_size[0]
        self.step = self.conv.stride[0]
        size = (1, self.conv.out_channels, 1)
        self.register_buffer("mean", torch.zeros(size, device=self.conv.weight.device))
        self.register_buffer("variance", torch.zeros(size, device=self.conv.weight.device))

    def reset(self):
        self.mean.zero_()
        self.variance.zero_()

    def statistics(self, audio):
        hidden = self.conv(audio)
        upper = F.max_pool1d(audio, self.kernel, self.step)
        lower = -F.max_pool1d(-audio, self.kernel, self.step)
        speech = (upper > lower).to(hidden.dtype)
        total = hidden.shape[-1]
        silent = total - speech.sum()
        mean = ((hidden * speech).sum(-1, keepdim=True) + silent * self.mean) / total
        variance = ((hidden - mean).square() * speech).sum(-1, keepdim=True)
        variance = (variance + silent * (self.variance + (self.mean - mean).square())) / total
        self.mean.copy_(mean)
        self.variance.copy_(variance)
        return hidden, mean, variance

    def normalize(self, hidden, mean, variance):
        hidden = (hidden - mean) * torch.rsqrt(variance + self.norm.eps)
        return self.activation(hidden * self.norm.weight[:, None] + self.norm.bias[:, None])

    def forward(self, audio):
        return self.normalize(*self.statistics(audio))


class CachedContentEncoder:
    def __init__(self, embedder, version, window, active, cuda_graph):
        config = embedder.config
        self.embedder = embedder
        self.version = version
        self.window = int(window)
        self.active = int(active)
        self.stride, self.field = conv_geometry(config)
        if not self.fits(embedder, self.window, self.active):
            raise ValueError("Extra context is too short for cached context.")
        self.past = self.window // self.stride + 1 - self.active
        self.context = config.num_conv_pos_embeddings // 2
        self.stats_samples = self.window - self.stride // 2
        self.active_samples = self.stride * (self.active - 1) + self.field
        self.normalize = bool(getattr(embedder, "audio_requires_normalization", False))
        device = next(embedder.parameters()).device
        layers = len(embedder.encoder.layers)
        heads = config.num_attention_heads
        size = (layers, 1, heads, self.past + self.active, config.hidden_size // heads)
        self.keys = torch.zeros(size, device=device)
        self.values = torch.zeros(size, device=device)
        self.mask = torch.zeros(1, 1, 1, self.past + self.active, dtype=torch.bool, device=device)
        self.mask[..., self.past:] = True
        self.valid = self.mask[0, 0, 0, :self.past]
        self.projection = torch.zeros(1, self.context + self.active, config.hidden_size, device=device)
        self.history = None
        self.output = None
        self.first = None
        self.speech = None
        self.cursor = 0
        self.pending = 0
        self.graph = RealtimeGraph(self._encode, "cached content feature", cuda_graph)

    @staticmethod
    def supports(embedder):
        config = getattr(embedder, "config", None)
        if config is None or config.do_stable_layer_norm:
            return False
        if config.feat_extract_norm == "group":
            return isinstance(embedder.feature_extractor.conv_layers[0], SpeechGroupNormConv)
        return config.feat_extract_norm == "layer"

    @staticmethod
    def fits(embedder, window, active):
        stride, field = conv_geometry(embedder.config)
        return window >= stride * active + field

    def reset(self):
        self.first = None

    def _clear(self):
        self.valid.zero_()
        self.projection[:, :self.context].zero_()
        self.cursor = 0
        self.pending = self.past

    def _advance(self, shift):
        past = self.past
        moved = min(shift, self.active)
        gap = shift - moved
        kept = min(moved, past)
        if kept:
            slots = torch.arange(self.cursor, self.cursor + kept, device=self.keys.device) % past
            source = slice(past + moved - kept, past + moved)
            self.keys.index_copy_(3, slots, self.keys[:, :, :, source].clone())
            self.values.index_copy_(3, slots, self.values[:, :, :, source].clone())
            self.valid[slots] = True
            self.cursor = (self.cursor + kept) % past
        if gap:
            count = min(gap, past)
            slots = torch.arange(self.cursor, self.cursor + count, device=self.keys.device) % past
            self.valid[slots] = False
            self.cursor = (self.cursor + count) % past
        context = self.context
        projection = torch.cat(
            (
                self.projection[:, :context],
                self.projection[:, context:context + moved],
                self.projection.new_zeros(1, min(gap, context), self.projection.shape[-1]),
            ),
            dim=1,
        )
        self.projection[:, :context].copy_(projection[:, -context:])
        self.history = torch.cat(
            (self.history, self.output[:, :moved], self.history.new_zeros(1, min(gap, past), self.history.shape[-1])),
            dim=1,
        )[:, -past:]
        self.pending = min(gap, past)

    def _encode(self, stats, audio):
        embedder = self.embedder
        if self.normalize:
            variance, mean = torch.var_mean(stats, correction=0)
            scale = torch.rsqrt(variance + 1e-5)
            stats = (stats - mean) * scale
            audio = (audio - mean) * scale
        layers = embedder.feature_extractor.conv_layers
        hidden = audio[:, None]
        if isinstance(layers[0], SpeechGroupNormConv):
            first = layers[0]
            _, mean, variance = first.statistics(stats[:, None])
            hidden = first.normalize(first.conv(hidden), mean, variance)
            layers = layers[1:]
        for layer in layers:
            hidden = layer(hidden)
        hidden = embedder.feature_projection(hidden.transpose(1, 2))
        context = self.context
        self.projection[:, context:].copy_(hidden)
        encoder = embedder.encoder
        hidden = encoder.layer_norm(hidden + encoder.pos_conv_embed(self.projection)[:, context:])
        past = self.past
        for index, layer in enumerate(encoder.layers):
            attention = layer.attention
            shape = (1, self.active, attention.num_heads, attention.head_dim)
            query = attention.q_proj(hidden).view(shape).transpose(1, 2)
            self.keys[index, :, :, past:].copy_(attention.k_proj(hidden).view(shape).transpose(1, 2))
            self.values[index, :, :, past:].copy_(attention.v_proj(hidden).view(shape).transpose(1, 2))
            attended = F.scaled_dot_product_attention(
                query, self.keys[index], self.values[index], attn_mask=self.mask, scale=attention.scaling
            )
            attended = attention.out_proj(attended.transpose(1, 2).reshape(1, self.active, -1))
            hidden = layer.layer_norm(hidden + attended)
            hidden = layer.final_layer_norm(hidden + layer.feed_forward(hidden))
        if self.version == "v1":
            hidden = embedder.final_proj(hidden)
        return hidden

    def _step(self, audio, end):
        first = (end - self.field) // self.stride - self.active + 1
        shift = 0 if self.first is None else first - self.first
        if shift <= 0 or shift >= self.past + self.active:
            self._clear()
        else:
            self._advance(shift)
        self.first = first
        start = self.stride * first - (end - audio.shape[0])
        stats = audio[audio.shape[0] - self.stats_samples:].view(1, -1)
        active = audio[start:start + self.active_samples].view(1, -1)
        output = self.graph(stats, active)
        if self.history is None:
            self.history = output.new_zeros(1, self.past, output.shape[-1])
        if self.pending:
            self.history[:, self.past - self.pending:] = output[:, :1]
            self.pending = 0
        self.output = output
        return output

    def keep(self, audio, end, step, limit, speech):
        if self.first is None:
            return
        last = (end - self.field) // self.stride
        if speech:
            self.speech = last
        if self.first > self.speech:
            limit = self.active
        if last - self.active + 1 - self.first + step > limit:
            self._step(audio, end)

    def __call__(self, audio, end):
        output = self._step(audio, end)
        self.speech = self.first + self.active - 1
        return torch.cat((self.history, output), dim=1), self.first - self.past
