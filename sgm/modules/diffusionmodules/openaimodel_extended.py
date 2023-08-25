from abc import abstractmethod
from functools import partial
from typing import Optional

import torch
import torch as th
import torch.nn as nn

from ...modules.attention import SpatialTransformer
from ...modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from ...util import default, exists

from .openaimodel import TimestepBlock, ResBlock, AttentionBlock, Upsample, Downsample
from .openaimodel import convert_module_to_f16, convert_module_to_f32

from .openaimodel import UNetModel

from .tb_lora_switchers import TBLoRaSwitcher
from ...util import instantiate_from_config


class BaseTBLoRa(nn.Module):
    def __init__(
            self,
            base: nn.Module,
            switcher: TBLoRaSwitcher,
            lora_r: int = 4,
            lora_alpha: Optional[float] = None,
            lora_dropout: float = 0.,
    ):
        super().__init__()
        self.base = base
        self.switcher = switcher
        self.lora_r = lora_r
        self.lora_scale = 1 if lora_alpha is None else lora_alpha / lora_r
        self.lora_dropout = nn.Dropout1d(lora_dropout) if lora_dropout > 0. else nn.Identity()

        # self.lora_A = nn.ModuleList([
        #     self.makeA_()
        #     for _ in range(switcher.segments_count())
        # ])
        # self.lora_B = nn.ModuleList([
        #     self.makeB_()
        #     for _ in range(switcher.segments_count())
        # ])

        self.loras = nn.ModuleList([
            nn.Sequential(self.makeA_(), zero_module(self.makeB_()))
            for _ in range(switcher.segments_count())
        ])

    @abstractmethod
    def makeA_(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def makeB_(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, x):
        segment = self.switcher.current_segment()
        res = self.base(x)
        x = self.lora_dropout(x)

        # # 30 it 5.37 s
        # lora = torch.zeros_like(res)
        #
        # for i in range(self.switcher.total_segments):
        #     mask = (segment == i)
        #     lora[mask] = self.lora_B[i](self.lora_A[i](x[mask]))

        # # 30 it 4.57 s (6 segments); 3.57s for 4 segments
        # lora = torch.stack(
        #     [
        #         self.lora_B[i](self.lora_A[i](x))
        #         for i in range(self.switcher.total_segments)
        #     ],
        #     dim=1
        # )
        # lora = lora[torch.arange(x.shape[0]), segment]

        # 3.38 for 4 segments
        lora = torch.stack(
            [
                lora(x) for lora in self.loras
            ],
            dim=1
        )
        lora = lora[torch.arange(x.shape[0]), segment]

        return res + lora*self.lora_scale


class LinearTBLoRa(BaseTBLoRa):
    def makeA_(self) -> nn.Module:
        return nn.Linear(
            self.base.in_features, self.lora_r,
            bias=False
        )

    @abstractmethod
    def makeB_(self) -> nn.Module:
        return nn.Linear(
            self.lora_r, self.base.out_features,
            bias=False
        )


class Conv2dTBLoRa(BaseTBLoRa):
    def makeA_(self) -> nn.Module:
        return nn.Conv2d(
            self.base.in_channels, self.lora_r,
            kernel_size=self.base.kernel_size,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
            bias=False
        )

    @abstractmethod
    def makeB_(self) -> nn.Module:
        return nn.Conv2d(
            self.lora_r, self.base.out_channels,
            kernel_size=1,
            bias=False
        )


class UNetExtModel(UNetModel, TBLoRaSwitcher):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        # pop switcher kwargs first
        switcher_config = kwargs.pop("switcher_config")

        super().__init__(*args, **kwargs)
        self.switcher = instantiate_from_config(switcher_config)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Base params count: {total_params}")

        self.add_lora_()

        total_params_lora = sum(p.numel() for p in self.parameters())
        print(f"Extended params count: {total_params_lora}")

        print(f"Lora only params: {total_params_lora - total_params}, +{(total_params_lora - total_params) * 100 / total_params:.2f}%")

    def add_lora_(self):
        def rec_add_lora(module: nn.Module):
            replaces = {}
            for name, submodule in module.named_children():
                new_attr = None
                if type(submodule) == torch.nn.Linear:
                    new_attr = LinearTBLoRa(submodule, self.switcher)
                elif type(submodule) == torch.nn.Conv2d:
                    new_attr = Conv2dTBLoRa(submodule, self.switcher)

                if new_attr is not None:
                    replaces[name] = new_attr
                else:
                    rec_add_lora(submodule)

            for name, new_attr in replaces.items():
                setattr(module, name, new_attr)

            pass

        rec_add_lora(self)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        self.switcher.select_segment(timesteps)
        return super().forward(x, timesteps=timesteps, context=context, y=y, **kwargs)


if __name__ == "__main__":
    model = UNetExtModel(
        use_checkpoint=True,
        image_size=64,
        in_channels=4,
        out_channels=4,
        model_channels=128,
        attention_resolutions=[4, 2],
        num_res_blocks=2,
        channel_mult=[1, 2, 4],
        num_head_channels=64,
        use_spatial_transformer=False,
        use_linear_in_transformer=True,
        transformer_depth=1,
        legacy=False,
    ).cuda()
    x = th.randn(11, 4, 64, 64).cuda()
    t = th.randint(low=0, high=10, size=(11,), device="cuda")
    o = model(x, t)
    print("done.")
