from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from torch import Tensor
# adapated form https://huggingface.co/docs/diffusers/main/api/models/unet

@dataclass
class UNet1DOutput:
    """Output container for UNet1DModel."""
    sample: Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard 1D sinusoidal time embedding used in many diffusers.

    Produces shape (B, dim).
    """

    def __init__(self, dim: int, flip_sin_to_cos: bool = False, freq_shift: float = 0.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbedding: dim must be even, got {dim}")
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        timesteps: (B,) or (B, 1) scalar timesteps
        returns: (B, dim)
        """
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]  # (B, 1)

        device = timesteps.device
        half_dim = self.dim // 2
        # log-spaced frequencies
        exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = exponent / (half_dim - self.freq_shift)
        freqs = torch.exp(-math.log(10000) * exponent)  # (half_dim,)
        args = timesteps.float() * freqs[None, :]       # (B, half_dim)

        sin = torch.sin(args)
        cos = torch.cos(args)

        if self.flip_sin_to_cos:
            emb = torch.cat([cos, sin], dim=-1)
        else:
            emb = torch.cat([sin, cos], dim=-1)
        return emb  # (B, dim)


class TimeMLP(nn.Module):
    """
    Simple MLP to process time embedding (like TimestepEmbedding).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4, act: str = "silu"):
        super().__init__()

        hidden_dim = in_dim * hidden_mult
        if act == "silu":
            act_fn = nn.SiLU()
        elif act == "relu":
            act_fn = nn.ReLU()
        elif act == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.SiLU()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



class ResBlock1D(nn.Module):
    """
    Simple residual block:
        Conv1d -> GroupNorm -> Act -> Conv1d -> GroupNorm, with optional time embedding.
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

        if act == "silu":
            act_fn = nn.SiLU()
        elif act == "relu":
            act_fn = nn.ReLU()
        elif act == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.SiLU()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.act = act_fn

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)

        # project time embedding (B, temb_dim) -> (B, out_channels)
        if temb_dim is not None:
            self.time_proj = nn.Linear(temb_dim, out_channels)
        else:
            self.time_proj = None

        # skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        # x: (B, C, L)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        if temb is not None and self.time_proj is not None:
            # temb: (B, D) -> (B, C, 1)
            t = self.time_proj(temb)[:, :, None]
            h = h + t

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)


class DownBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        num_layers: int = 1,
        num_groups: int = 8,
        downsample: bool = True,
    ):
        super().__init__()

        self.resblocks = nn.ModuleList(
            [
                ResBlock1D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_dim=temb_dim,
                    num_groups=num_groups,
                )
                for i in range(num_layers)
            ]
        )

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor, temb: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        res_samples = []
        for block in self.resblocks:
            x = block(x, temb)
            res_samples.append(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x, tuple(res_samples)


class UpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        num_layers: int = 1,
        num_groups: int = 8,
        upsample: bool = True,
    ):
        super().__init__()

        self.resblocks = nn.ModuleList(
            [
                ResBlock1D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    temb_dim=temb_dim,
                    num_groups=num_groups,
                )
                for i in range(num_layers)
            ]
        )

        self.upsample = None
        if upsample:
            self.upsample = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor, res_samples: Tuple[Tensor, ...], temb: Tensor) -> Tensor:
        # we only use the last residual from the corresponding down block here (simple version)
        if len(res_samples) > 0:
            x = x + res_samples[-1]

        for block in self.resblocks:
            x = block(x, temb)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class MidBlock1D(nn.Module):
    def __init__(self, channels: int, temb_dim: int, num_layers: int = 1, num_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock1D(channels, channels, temb_dim=temb_dim, num_groups=num_groups) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, temb)
        return x


class OutBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8, act: str = "silu"):
        super().__init__()

        if act == "silu":
            act_fn = nn.SiLU()
        elif act == "relu":
            act_fn = nn.ReLU()
        elif act == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.SiLU()

        self.norm = nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels)
        self.act = act_fn
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class UNet1DModel(nn.Module):
    """
    Args are similar in spirit to your original code, but simplified and
    not tied to diffusers mixins.
    """

    def __init__(
        self,
        sample_size: int = 65536,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        time_embedding_type: str = "fourier",
        time_embedding_dim: Optional[int] = None,
        use_timestep_embedding: bool = True,
        flip_sin_to_cos: bool = True,
        freq_shift: float = 0.0,
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

        # ---- time embedding ----
        base_channels = block_out_channels[0]
        time_embed_dim = time_embedding_dim or base_channels * 2
        if time_embed_dim % 2 != 0:
            raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")

        self.time_proj = SinusoidalTimeEmbedding(
            dim=time_embed_dim, flip_sin_to_cos=flip_sin_to_cos, freq_shift=freq_shift
        )

        self.use_timestep_embedding = use_timestep_embedding
        if use_timestep_embedding:
            self.time_mlp = TimeMLP(in_dim=time_embed_dim, out_dim=base_channels, hidden_mult=2, act=act_fn)
            temb_dim = base_channels
        else:
            # we will broadcast raw sinusoidal embedding along length
            temb_dim = time_embed_dim

        # ---- down blocks ----
        self.down_blocks = nn.ModuleList()
        out_ch = in_channels
        for i, ch in enumerate(block_out_channels):
            in_ch = out_ch
            out_ch = ch

            if i == 0:
                in_ch += extra_in_channels

            is_last = i == len(block_out_channels) - 1
            down_block = DownBlock1D(
                in_channels=in_ch,
                out_channels=out_ch,
                temb_dim=temb_dim,
                num_layers=layers_per_block,
                num_groups=norm_num_groups,
                downsample=not is_last or downsample_each_block,
            )
            self.down_blocks.append(down_block)

        # ---- mid block ----
        self.mid_block = MidBlock1D(
            channels=block_out_channels[-1],
            temb_dim=temb_dim,
            num_layers=layers_per_block,
            num_groups=norm_num_groups,
        )

        # ---- up blocks ----
        self.up_blocks = nn.ModuleList()
        reversed_ch = list(reversed(block_out_channels))
        out_ch = reversed_ch[0]
        for i, ch in enumerate(reversed_ch):
            in_ch = out_ch
            out_ch = reversed_ch[i + 1] if i < len(reversed_ch) - 1 else out_channels

            is_last = i == len(reversed_ch) - 1
            up_block = UpBlock1D(
                in_channels=in_ch,
                out_channels=out_ch,
                temb_dim=temb_dim,
                num_layers=layers_per_block,
                num_groups=norm_num_groups,
                upsample=not is_last,
            )
            self.up_blocks.append(up_block)

        # ---- out block ----
        self.out_block = OutBlock1D(
            in_channels=out_ch,
            out_channels=out_channels,
            num_groups=norm_num_groups,
            act=act_fn,
        )

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(
        self,
        sample: Tensor,
        timestep: Union[Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple[Tensor]]:
        """
        Args:
            sample: (B, C, L)
            
            timestep: scalar or (B,) or (B, 1)
        """
        x = sample

        # 1. time embedding
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.float32, device=x.device)
        else:
            timesteps = timestep.to(sample.device).float()
        if timesteps.ndim == 0:
            timesteps = timesteps[None]

        temb = self.time_proj(timesteps)  # (B, D)
        if self.use_timestep_embedding:
            temb = self.time_mlp(temb)  # (B, C0)

        # if we do NOT use time_mlp, we keep temb as (B, D) and broadcast inside ResBlock

        # 2. down path
        res_stack = []
        for down in self.down_blocks:
            x, res = down(x, temb)
            res_stack.append(res)  # tuple of res

        # 3. mid
        x = self.mid_block(x, temb)

        # 4. up path
        # we simply pop one residual tuple per up block
        for up in self.up_blocks:
            res = res_stack.pop() if len(res_stack) > 0 else ()
            x = up(x, res, temb)

        # 5. output
        x = self.out_block(x)
        
        return x
