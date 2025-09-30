import logging
import os
from typing import Optional, List, Tuple

import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class Spectral2DCNN(nn.Module):
    def __init__(
        self,
        n_bins: int,
        n_frames: int,
        in_ch: int = 1,
        kernel_size: Tuple[int, int] = (3, 3),
        out_channels: Optional[List[int]] = None,
        bin_dilations: Optional[List[int]] = None,
        temp_dilations: Optional[List[int]] = None,
        pool_sizes: Optional[List[List[int]]] = None,
        latent_dim: int = 32,
        use_ln: bool = True,
        dropout_prob: float = 0.25,
        n_params: int = 2,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.n_frames = n_frames
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.use_ln = use_ln
        self.dropout_prob = dropout_prob
        self.n_params = n_params

        if out_channels is None:
            out_channels = [128] * 5
        self.out_channels = out_channels
        if bin_dilations is None:
            bin_dilations = [1] * len(out_channels)
        self.bin_dilations = bin_dilations
        if temp_dilations is None:
            temp_dilations = [1] * len(out_channels)
        self.temp_dilations = temp_dilations
        if pool_sizes is None:
            pool_sizes = [(2, 2)] * len(out_channels)
        self.pool_sizes = pool_sizes
        assert (
            len(out_channels)
            == len(bin_dilations)
            == len(temp_dilations)
            == len(pool_sizes)
        )

        layers = []
        for out_ch, b_dil, t_dil, pool_size in zip(
            out_channels, bin_dilations, temp_dilations, pool_sizes
        ):
            if use_ln:
                layers.append(
                    nn.LayerNorm([n_bins, n_frames], elementwise_affine=False)
                )
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=(1, 1),
                    dilation=(b_dil, t_dil),
                    padding="same",
                )
            )
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
            layers.append(nn.PReLU())
            in_ch = out_ch
            n_bins = n_bins // pool_size[0]
            n_frames = n_frames // pool_size[1]
        self.cnn = nn.Sequential(*layers)

        self.fc = nn.Linear(out_channels[-1], latent_dim)
        self.fc_act = nn.PReLU()
        self.do = nn.Dropout(p=dropout_prob)

        # self.fc_d = nn.Linear(latent_dim, latent_dim // 2)
        # self.fc_d_act = nn.PReLU()
        # self.out_d = nn.Linear(latent_dim // 2, 1)
        #
        # self.fc_s = nn.Linear(latent_dim, latent_dim // 2)
        # self.fc_s_act = nn.PReLU()
        # self.out_s = nn.Linear(latent_dim // 2, 1)

        param_mlps = []
        for _ in range(n_params):
            param_mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.PReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(latent_dim // 2, 1),
                nn.Sigmoid(),
            )
            param_mlps.append(param_mlp)
        self.param_mlps = nn.ModuleList(param_mlps)

    def forward(self, x: T) -> List[T]:
        assert x.ndim == 3
        x = x.unsqueeze(1)
        x = self.cnn(x)
        # log.info(f"x.shape after cnn: {x.shape}")
        x = tr.mean(x, dim=(2, 3))
        x = self.fc(x)
        x = self.fc_act(x)
        x = self.do(x)
        latent = x

        # x = self.fc_d(latent)
        # x = self.fc_d_act(x)
        # x = self.do(x)
        # d_hat = self.out_d(x)
        # d_hat = tr.sigmoid(d_hat).squeeze(1)
        # # d_hat = magic_clamp(d_hat, min_value=0.0, max_value=1.0).squeeze(1)
        #
        # x = self.fc_s(latent)
        # x = self.fc_s_act(x)
        # x = self.do(x)
        # s_hat = self.out_s(x)
        # s_hat = tr.sigmoid(s_hat).squeeze(1)
        # # s_hat = magic_clamp(s_hat, min_value=0.0, max_value=1.0).squeeze(1)
        # return d_hat, s_hat

        out = []
        for param_mlp in self.param_mlps:
            p_hat = param_mlp(latent)
            p_hat = p_hat.squeeze(dim=1)
            out.append(p_hat)
        return out
