import math
from inspect import isfunction
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_parameter_breakdown(model):
    print(f"{'Module':40s} | {'#params':>12s}")
    print("-" * 60)
    for name, module in model.named_modules():
        if len(list(module.parameters(recurse=False))) > 0:
            n = sum(p.numel() for p in module.parameters(recurse=False))
            print(f"{name:40s} | {n:12,d}")




def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# -----------------------
# Core building blocks
# -----------------------


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding=1, bias=True),
    )


def Downsample(dim, dim_out=None):
    # Pixel-unshuffle–style downsample by factor 2 in each spatial dim
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) (d p3) -> b (c p1 p2 p3) h w d", p1=2, p2=2, p3=2),
        nn.Conv3d(dim * 8, default(dim_out, dim * 2), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: (B,) or (B, 1)
        returns: (B, dim)
        """
        if time.dim() == 2 and time.shape[1] == 1:
            time = time.squeeze(1)

        device = time.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv3d(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    Weight standardization + GroupNorm (in Block) works well in practice.
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv3d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """
    Full self-attention over all voxels (O(N^2) memory).
    Only safe to use at aggressively downsampled resolutions.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d x y -> b h c (d x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (d x y) c -> b (h c) d x y", d=d, x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Linear-time attention variant.
    Still heavy at full 3D resolution, so we only apply it at coarse levels.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d x y -> b h c (d x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (d x y) -> b (h c) d x y", d=d, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)



class UNet3DModel(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 1, 2),
        channels=1,
        self_condition=False,
        resnet_block_groups=4,
        attn_dim_head=16,
        attn_heads=4,
        n_block_klass=2,
    ):
        super().__init__()

        # config
        self.channels = channels
        self.self_condition = self_condition
        self.n_block_klass = n_block_klass

        input_channels = channels * (2 if self_condition else 1)

        init_dim = init_dim or dim
        self.init_conv = nn.Conv3d(input_channels, init_dim, 1, padding=0)

        # channel multipliers per resolution
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self._dim_mults = dim_mults  # just for debugging if needed

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Down / Up paths
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # ---- DOWN PATH ----
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # Only use LinearAttention on the *coarsest* two resolutions
            # e.g. for 4 resolutions (0,1,2,3), we use it on 2 and 3.
            use_linear_attn = ind >= max(0, num_resolutions - 2)

            if use_linear_attn:
                attn_module = Residual(
                    PreNorm(
                        dim_in,
                        LinearAttention(
                            dim_in, heads=attn_heads, dim_head=attn_dim_head
                        ),
                    )
                )
            else:
                attn_module = nn.Identity()

            self.downs.append(
                nn.ModuleList(
                    [
                        *[
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim)
                            for _ in range(self.n_block_klass)
                        ],
                        attn_module,
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # how many times we actually downsample by factor 2 (for padding)
        self.num_downs = num_resolutions - 1

        # ---- MID (bottleneck) ----
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(
            PreNorm(
                mid_dim,
                Attention(dim=mid_dim, dim_head=attn_dim_head, heads=attn_heads),
            )
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # ---- UP PATH ----
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            # map back to resolution index used in the down path
            res_index = (num_resolutions - 1) - ind
            use_linear_attn = res_index >= max(0, num_resolutions - 2)

            if use_linear_attn:
                attn_module = Residual(
                    PreNorm(
                        dim_out,
                        LinearAttention(
                            dim_out, heads=attn_heads, dim_head=attn_dim_head
                        ),
                    )
                )
            else:
                attn_module = nn.Identity()

            self.ups.append(
                nn.ModuleList(
                    [
                        *[
                            block_klass(
                                dim_out + dim_in, dim_out, time_emb_dim=time_dim
                            )
                            for _ in range(self.n_block_klass)
                        ],
                        attn_module,
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = out_dim or channels

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    # ---- helpers for padding ----

    def _compute_padding(self, d, h, w):
        """
        Compute how much padding is needed so that each dim is divisible by 2**num_downs.
        Returns pad_d, pad_h, pad_w (amount to add on the 'right' side).
        """
        factor = 2**self.num_downs if self.num_downs > 0 else 1

        def pad_needed(n, f):
            r = n % f
            return 0 if r == 0 else f - r

        pad_d = pad_needed(d, factor)
        pad_h = pad_needed(h, factor)
        pad_w = pad_needed(w, factor)
        return pad_d, pad_h, pad_w

    def _pad_input(self, x):
        """
        x: (B, C, D, H, W)
        Returns:
            x_padded, (D, H, W) original, (pad_d, pad_h, pad_w)
        """
        _, _, d0, h0, w0 = x.shape
        pad_d, pad_h, pad_w = self._compute_padding(d0, h0, w0)

        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return x, (d0, h0, w0), (0, 0, 0)

        # F.pad uses (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(
            x,
            (
                0,
                pad_w,
                0,
                pad_h,
                0,
                pad_d,
            ),
        )
        return x, (d0, h0, w0), (pad_d, pad_h, pad_w)

    def _crop_output(self, x, orig_shape):
        """
        x: (B, C, D', H', W')
        orig_shape: (D, H, W)
        """
        d0, h0, w0 = orig_shape
        return x[..., :d0, :h0, :w0]

    # ---- main forward ----

    def forward(self, x, time, x_self_cond=None):
        """
        x: (B, C, D, H, W)
        time: (B,) or (B, 1)
        """
        # handle self-conditioning first (does not change spatial size)
        if self.self_condition:
            x_self_cond = (
                x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            )
            x = torch.cat((x_self_cond, x), dim=1)

        # pad to make spatial dims divisible by 2**num_downs
        x, orig_shape, _ = self._pad_input(x)

        # initial conv + long skip
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        # down path
        for i in range(len(self.downs)):
            blocks_and_others = self.downs[i]
            # first n_block_klass-1 resnet blocks
            if self.n_block_klass > 1:
                for block in blocks_and_others[: self.n_block_klass - 1]:
                    x = block(x, t)
                    h.append(x)

            # last resnet block + attn + downsample / conv
            for block, attn, downsample in [
                blocks_and_others[self.n_block_klass - 1 :]
            ]:
                x = block(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)

        # mid
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # up path
        for i in range(len(self.ups)):
            blocks_and_others = self.ups[i]
            # first n_block_klass-1 resnet blocks with skip concat
            if self.n_block_klass > 1:
                for block in blocks_and_others[: self.n_block_klass - 1]:
                    x = torch.cat((x, h.pop()), dim=1)
                    x = block(x, t)

            # last resnet block + attn + upsample / conv
            for block, attn, upsample in [blocks_and_others[self.n_block_klass - 1 :]]:
                x = torch.cat((x, h.pop()), dim=1)
                x = block(x, t)
                x = attn(x)
                x = upsample(x)

        # final
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        # crop back to original spatial size
        x = self._crop_output(x, orig_shape)

        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    # instantiate the network
    model = UNet3DModel(
        dim=32,
        init_dim=32,
        out_dim=1,  # same as channels for standard diffusion
        dim_mults=(1, 2, 4, 8),  # 2 downsamples -> factor 4
        channels=1,
        self_condition=False,
        resnet_block_groups=4,
        attn_dim_head=10,
        attn_heads=2,
        n_block_klass=1,
    )
    total, trainable = count_parameters(model)

    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"≈ {total / 1e6:.2f} M parameters")
    print_parameter_breakdown(model)
    device = "cuda"
    model.to(device)
    model.eval()

    # test shapes
    test_shapes = [
        (1, 1, 169, 208, 179),  # your problematic case
        (2, 1, 64, 64, 64),
        (1, 1, 65, 97, 33),
    ]

    for shape in test_shapes:
        b, c, d, h, w = shape
        x = torch.randn(*shape, device=device)
        t = torch.rand(b, device=device) * 100.0  # dummy time

        with torch.no_grad():
            y = model(x, t)

        print(f"Input shape : {tuple(x.shape)}")
        print(f"Output shape: {tuple(y.shape)}")
        assert y.shape == (b, 1, d, h, w), "Output spatial size or channels mismatch!"
        print("OK\n")

    print("All tests passed")
