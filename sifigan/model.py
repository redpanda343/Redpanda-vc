import torch
import torch.nn as nn


def pd_indexing(x, d, dilation):
    B, C, T = x.size()
    batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1)
    dilations = torch.clamp((d * dilation).long(), min=1)
    idx_base = torch.arange(0, T, dtype=torch.long, device=x.device).reshape(1, 1, T)
    idxP = (idx_base - dilations).abs() % T
    idxP = (batch_index, ch_index, idxP)
    idxF = idx_base + dilations
    overflowed = idxF >= T
    idxF[overflowed] = -(idxF[overflowed] % T)
    idxF = (batch_index, ch_index, idxF)
    return (x[idxP], x[idxF])


class Snake(nn.Module):

    def __init__(self, channels, init=50):
        super(Snake, self).__init__()
        alpha = init * torch.ones(1, channels, 1)
        self.alpha = nn.Parameter(alpha)

    def forward(self, x):
        return x + torch.sin(self.alpha * x) ** 2 / self.alpha


class Conv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):

    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class ResidualBlock(nn.Module):

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
    ):
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()
        assert kernel_size % 2 == 1, 'Kernel size must be odd number.'
        for dilation in dilations:
            if nonlinear_activation == 'Snake':
                nonlinear = Snake(channels, **nonlinear_activation_params)
            else:
                nonlinear = getattr(nn, nonlinear_activation)(
                    **nonlinear_activation_params
                )
            self.convs1 += [
                nn.Sequential(
                    nonlinear,
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                if nonlinear_activation == 'Snake':
                    nonlinear = Snake(channels, **nonlinear_activation_params)
                else:
                    nonlinear = getattr(nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    )
                self.convs2 += [
                    nn.Sequential(
                        nonlinear,
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x):
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x


class AdaptiveResidualBlock(nn.Module):

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 2, 4),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
    ):
        super().__init__()
        self.use_additional_convs = use_additional_convs
        assert kernel_size == 3, 'Currently only kernel_size = 3 is supported.'
        self.channels = channels
        self.dilations = dilations
        self.nonlinears = nn.ModuleList()
        self.convsC = nn.ModuleList()
        self.convsP = nn.ModuleList()
        self.convsF = nn.ModuleList()
        if use_additional_convs:
            self.convsA = nn.ModuleList()
        for _ in dilations:
            if nonlinear_activation == 'Snake':
                self.nonlinears += [Snake(channels, **nonlinear_activation_params)]
            else:
                self.nonlinears += [
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
                ]
            self.convsC += [Conv1d1x1(channels, channels, bias=bias)]
            self.convsP += [Conv1d1x1(channels, channels, bias=bias)]
            self.convsF += [Conv1d1x1(channels, channels, bias=bias)]
            if use_additional_convs:
                if nonlinear_activation == 'Snake':
                    nonlinear = Snake(channels, **nonlinear_activation_params)
                else:
                    nonlinear = getattr(nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    )
                self.convsA += [
                    nn.Sequential(
                        nonlinear,
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x, d):
        for i, dilation in enumerate(self.dilations):
            xt = self.nonlinears[i](x)
            xP, xF = pd_indexing(xt, d, dilation)
            xt = self.convsC[i](xt) + self.convsP[i](xP) + self.convsF[i](xF)
            if self.use_additional_convs:
                xt = self.convsA[i](xt)
            x = xt + x
        return x


class SiFiGANGenerator(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(5, 4, 3, 2),
        upsample_kernel_sizes=(10, 8, 6, 4),
        source_network_params={
            'resblock_kernel_size': 3,
            'resblock_dilations': [1, (1, 2), (1, 2, 4), (1, 2, 4, 8)],
            'use_additional_convs': True,
        },
        filter_network_params={
            'resblock_kernel_sizes': (3, 5, 7),
            'resblock_dilations': [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            'use_additional_convs': False,
        },
        share_upsamples=False,
        share_downsamples=False,
        bias=True,
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
        use_weight_norm=True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd number.'
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        self.num_upsamples = len(upsample_kernel_sizes)
        self.source_network_params = source_network_params
        self.filter_network_params = filter_network_params
        self.share_upsamples = share_upsamples
        self.share_downsamples = share_downsamples
        self.sn = nn.ModuleDict()
        self.fn = nn.ModuleDict()
        self.input_conv = Conv1d(
            in_channels,
            channels,
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        self.sn['upsamples'] = nn.ModuleList()
        self.fn['upsamples'] = nn.ModuleList()
        self.sn['blocks'] = nn.ModuleList()
        self.fn['blocks'] = nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.sn['upsamples'] += [
                nn.Sequential(
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    nn.ConvTranspose1d(
                        channels // 2**i,
                        channels // 2 ** (i + 1),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                        bias=bias,
                    ),
                )
            ]
            if not share_upsamples:
                self.fn['upsamples'] += [
                    nn.Sequential(
                        getattr(nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        nn.ConvTranspose1d(
                            channels // 2**i,
                            channels // 2 ** (i + 1),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                            output_padding=upsample_scales[i] % 2,
                            bias=bias,
                        ),
                    )
                ]
            self.sn['blocks'] += [
                AdaptiveResidualBlock(
                    kernel_size=source_network_params['resblock_kernel_size'],
                    channels=channels // 2 ** (i + 1),
                    dilations=source_network_params['resblock_dilations'][i],
                    bias=bias,
                    use_additional_convs=source_network_params['use_additional_convs'],
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]
            for j in range(len(filter_network_params['resblock_kernel_sizes'])):
                self.fn['blocks'] += [
                    ResidualBlock(
                        kernel_size=filter_network_params['resblock_kernel_sizes'][j],
                        channels=channels // 2 ** (i + 1),
                        dilations=filter_network_params['resblock_dilations'][j],
                        bias=bias,
                        use_additional_convs=filter_network_params[
                            'use_additional_convs'
                        ],
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.sn['output_conv'] = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                channels // 2 ** (i + 1),
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.fn['output_conv'] = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                channels // 2 ** (i + 1),
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            nn.Tanh(),
        )
        self.sn['emb'] = Conv1d(
            1,
            channels // 2 ** len(upsample_kernel_sizes),
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        self.sn['downsamples'] = nn.ModuleList()
        for i in reversed(range(len(upsample_kernel_sizes))):
            self.sn['downsamples'] += [
                nn.Sequential(
                    nn.Conv1d(
                        channels // 2 ** (i + 1),
                        channels // 2**i,
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i]
                        - (upsample_kernel_sizes[i] % 2 == 0),
                        bias=bias,
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
        if not share_downsamples:
            self.fn['downsamples'] = nn.ModuleList()
            for i in reversed(range(len(upsample_kernel_sizes))):
                self.fn['downsamples'] += [
                    nn.Sequential(
                        nn.Conv1d(
                            channels // 2 ** (i + 1),
                            channels // 2**i,
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i]
                            - (upsample_kernel_sizes[i] % 2 == 0),
                            bias=bias,
                        ),
                        getattr(nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                    )
                ]
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x, c, d):
        c = self.input_conv(c)
        e = c
        x = self.sn['emb'](x)
        embs = [x]
        for i in range(self.num_upsamples - 1):
            x = self.sn['downsamples'][i](x)
            embs += [x]
        for i in range(self.num_upsamples):
            e = self.sn['upsamples'][i](e) + embs[-i - 1]
            e = self.sn['blocks'][i](e, d[i])
        e_ = self.sn['output_conv'](e)
        embs = [e]
        for i in range(self.num_upsamples - 1):
            if self.share_downsamples:
                e = self.sn['downsamples'][i](e)
            else:
                e = self.fn['downsamples'][i](e)
            embs += [e]
        num_blocks = len(self.filter_network_params['resblock_kernel_sizes'])
        for i in range(self.num_upsamples):
            if self.share_upsamples:
                c = self.sn['upsamples'][i](c) + embs[-i - 1]
            else:
                c = self.fn['upsamples'][i](c) + embs[-i - 1]
            cs = 0.0
            for j in range(num_blocks):
                cs += self.fn['blocks'][i * num_blocks + j](c)
            c = cs / num_blocks
        c = self.fn['output_conv'](c)
        return (c, e_)

    def reset_parameters(self):

        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_reset_parameters)

    def remove_weight_norm(self):

        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
