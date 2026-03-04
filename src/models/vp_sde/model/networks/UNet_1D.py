from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from torch import Tensor


# ============================================================
# Output container
# ============================================================
@dataclass
class UNet1DOutput:
    sample: Tensor


# ============================================================
# Time embeddings
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """(B,) -> (B, dim) sinusoidal embedding."""

    def __init__(
        self, dim: int, flip_sin_to_cos: bool = False, freq_shift: float = 0.0
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]  # (B,1)

        device = timesteps.device
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = exponent / (half_dim - self.freq_shift)
        freqs = torch.exp(-math.log(10000.0) * exponent)  # (half_dim,)
        args = timesteps.float() * freqs[None, :]  # (B, half_dim)

        sin = torch.sin(args)
        cos = torch.cos(args)
        if self.flip_sin_to_cos:
            return torch.cat([cos, sin], dim=-1)
        return torch.cat([sin, cos], dim=-1)


class TimeMLP(nn.Module):
    """(B, in_dim) -> (B, out_dim)."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_mult: int = 4, act: str = "silu"
    ):
        super().__init__()
        hidden_dim = in_dim * hidden_mult
        act_fn = {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
            act, nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================
# Core blocks
# ============================================================
class ResBlock1D(nn.Module):
    """
    Conv1d -> GN -> Act -> Conv1d -> GN -> Act, with optional temb projection added after first act.
    Expects temb as (B, D) and projects to (B, C) then unsqueeze to (B,C,1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: Optional[int] = None,
        num_groups: int = 8,
        act: str = "silu",
    ):
        super().__init__()
        act_fn = {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
            act, nn.SiLU()
        )

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )
        self.act = act_fn

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )

        self.time_proj = (
            nn.Linear(temb_dim, out_channels) if temb_dim is not None else None
        )
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        if temb is not None and self.time_proj is not None:
            h = h + self.time_proj(temb)[:, :, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class DownBlock1D(nn.Module):
    """
    Returns:
        hidden_states: (B, C_out, L') after optional downsample
        res_samples: tuple of residual tensors (here: last hidden state of each ResBlock)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        num_layers: int = 1,
        num_groups: int = 8,
        add_downsample: bool = True,
        act: str = "silu",
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResBlock1D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_dim=temb_dim,
                    num_groups=num_groups,
                    act=act,
                )
                for i in range(num_layers)
            ]
        )
        self.downsample = (
            nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if add_downsample
            else None
        )

    def forward(
        self, hidden_states: Tensor, temb: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        res_samples = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            res_samples.append(hidden_states)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        return hidden_states, tuple(res_samples)


class UpBlock1D(nn.Module):
    """
    Upsample (if requested) -> add skip at matching resolution -> ResBlocks.
    This matches typical UNet behavior and avoids L mismatches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        num_layers: int = 1,
        num_groups: int = 8,
        add_upsample: bool = True,
        act: str = "silu",
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResBlock1D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_dim=temb_dim,
                    num_groups=num_groups,
                    act=act,
                )
                for i in range(num_layers)
            ]
        )
        self.upsample = (
            nn.ConvTranspose1d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
            if add_upsample
            else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Tensor,
    ) -> Tensor:
        # 1) upsample first (if this block upsamples)
        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        # 2) add skip at matching resolution (fail fast if mismatch)
        if len(res_hidden_states_tuple) > 0:
            skip = res_hidden_states_tuple[-1]
            if hidden_states.shape[-1] != skip.shape[-1]:
                raise RuntimeError(
                    f"Length mismatch at skip-add: hidden={hidden_states.shape}, skip={skip.shape}"
                )
            hidden_states = hidden_states + skip

        # 3) refine
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class MidBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        temb_dim: int,
        num_layers: int = 1,
        num_groups: int = 8,
        act: str = "silu",
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResBlock1D(
                    channels,
                    channels,
                    temb_dim=temb_dim,
                    num_groups=num_groups,
                    act=act,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x, temb)
        return x


class OutBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        act: str = "silu",
    ):
        super().__init__()
        act_fn = {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
            act, nn.SiLU()
        )
        self.norm = nn.GroupNorm(
            num_groups=min(num_groups, in_channels), num_channels=in_channels
        )
        self.act = act_fn
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.act(self.norm(x)))


# ============================================================
# Block factories (diffusers-style API)
# ============================================================
def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    num_groups: int = 8,
    act_fn: str = "silu",
) -> nn.Module:
    if down_block_type in {"DownBlock1D", "DownBlock1DNoSkip", "AttnDownBlock1D"}:
        return DownBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_dim=temb_channels,
            num_layers=num_layers,
            num_groups=num_groups,
            add_downsample=add_downsample,
            act=act_fn,
        )
    raise ValueError(f"Unknown down_block_type: {down_block_type}")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_upsample: bool,
    num_groups: int = 8,
    act_fn: str = "silu",
) -> nn.Module:
    if up_block_type in {"UpBlock1D", "UpBlock1DNoSkip", "AttnUpBlock1D"}:
        return UpBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_dim=temb_channels,
            num_layers=num_layers,
            num_groups=num_groups,
            add_upsample=add_upsample,
            act=act_fn,
        )
    raise ValueError(f"Unknown up_block_type: {up_block_type}")


def get_mid_block(
    mid_block_type: str,
    in_channels: int,
    embed_dim: int,
    num_layers: int,
    num_groups: int = 8,
    act_fn: str = "silu",
) -> nn.Module:
    if mid_block_type in {"UNetMidBlock1D", "MidBlock1D"}:
        return MidBlock1D(
            channels=in_channels,
            temb_dim=embed_dim,
            num_layers=num_layers,
            num_groups=num_groups,
            act=act_fn,
        )
    raise ValueError(f"Unknown mid_block_type: {mid_block_type}")


def get_out_block(
    out_channels: int,
    in_channels: int,
    num_groups: int = 8,
    act_fn: str = "silu",
) -> nn.Module:
    return OutBlock1D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        act=act_fn,
    )


# ============================================================
# UNet1DModel (diffusers-style wiring)
# ============================================================
class UNet1DModel(nn.Module):
    def __init__(
        self,
        sample_size: int = 65536,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        time_embedding_type: str = "positional",  # API parity; implemented as sinusoidal
        time_embedding_dim: Optional[int] = None,
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = True,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str, ...] = (
            "DownBlock1D",
            "DownBlock1D",
            "DownBlock1D",
        ),
        up_block_types: Tuple[str, ...] = ("UpBlock1D", "UpBlock1D", "UpBlock1D"),
        mid_block_type: str = "UNetMidBlock1D",
        block_out_channels: Tuple[int, ...] = (32, 32, 64),
        act_fn: str = "silu",
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_timestep_embedding = use_timestep_embedding

        # --- time ---
        base = block_out_channels[0]
        time_embed_dim = time_embedding_dim or base * 2
        if time_embed_dim % 2 != 0:
            raise ValueError(f"time_embedding_dim must be even; got {time_embed_dim}")

        self.time_proj = SinusoidalTimeEmbedding(
            time_embed_dim, flip_sin_to_cos=flip_sin_to_cos, freq_shift=freq_shift
        )

        if use_timestep_embedding:
            # (B, time_embed_dim) -> (B, base)
            self.time_mlp = TimeMLP(
                in_dim=time_embed_dim, out_dim=base, hidden_mult=2, act=act_fn
            )
            temb_channels = base
        else:
            # (kept for API parity; this simple implementation expects temb as (B,D) in ResBlock)
            temb_channels = time_embed_dim

        # --- down ---
        self.down_blocks = nn.ModuleList([])
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
                input_channel += extra_in_channels

            is_final_block = i == len(block_out_channels) - 1
            add_downsample = (not is_final_block) or downsample_each_block

            self.down_blocks.append(
                get_down_block(
                    down_block_type=down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=temb_channels,
                    add_downsample=add_downsample,
                    num_groups=norm_num_groups,
                    act_fn=act_fn,
                )
            )

        # --- mid ---
        self.mid_block = get_mid_block(
            mid_block_type=mid_block_type,
            in_channels=block_out_channels[-1],
            embed_dim=temb_channels,
            num_layers=layers_per_block,
            num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        # --- up ---
        self.up_blocks = nn.ModuleList([])
        reversed_channels = list(reversed(block_out_channels))  # e.g. [64, 32, 32]
        output_channel = reversed_channels[0]  # bottleneck channels

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_channels[i + 1]
                if i < len(up_block_types) - 1
                else block_out_channels[0]
            )

            # CRITICAL: first up block stays at bottleneck resolution; others upsample first
            add_upsample = i != 0

            self.up_blocks.append(
                get_up_block(
                    up_block_type=up_block_type,
                    num_layers=layers_per_block,
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=temb_channels,
                    add_upsample=add_upsample,
                    num_groups=norm_num_groups,
                    act_fn=act_fn,
                )
            )

        # --- out ---
        self.out_block = get_out_block(
            out_channels=out_channels,
            in_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            act_fn=act_fn,
        )

    def forward(
        self,
        sample: Tensor,
        timestep: Union[Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple[Tensor]]:
        # 1) timesteps -> temb
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.float32, device=sample.device
            )
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device).float()
        else:
            timesteps = timesteps.to(sample.device).float()

        temb = self.time_proj(timesteps)  # (B, D)
        if self.use_timestep_embedding:
            temb = self.time_mlp(temb)  # (B, base)

        # 2) down
        # Store ONE skip per down block (the last ResNet output), so 3 downs -> 3 skips.
        down_block_res_samples: Tuple[Tensor, ...] = ()
        hidden_states = sample
        for down_block in self.down_blocks:
            hidden_states, res_samples = down_block(
                hidden_states=hidden_states, temb=temb
            )
            down_block_res_samples = down_block_res_samples + (res_samples[-1],)

        # 3) mid
        hidden_states = self.mid_block(hidden_states, temb)

        # 4) up (consume one residual per up block)
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            hidden_states = up_block(
                hidden_states, res_hidden_states_tuple=res_samples, temb=temb
            )

        # 5) out
        hidden_states = self.out_block(hidden_states)

        if not return_dict:
            return (hidden_states,)
        return hidden_states


# ============================================================
# Quick sanity test (optional)
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    model = UNet1DModel(
        sample_size=256,
        in_channels=1,
        out_channels=1,
        block_out_channels=(32, 64, 64),
        layers_per_block=2,
        norm_num_groups=8,
        use_timestep_embedding=True,
    )
    x = torch.randn(4, 1, 256)
    t = torch.randint(0, 1000, (4,))
    y = model(x, t, return_dict=True).sample
    print("out:", y.shape)  # (4,1,256)
