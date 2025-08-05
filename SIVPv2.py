import math
import numbers

import cv2
import numpy as np
import torch
import torch.nn as nn

import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # [B,C,H,W] -> [B,C,H*W] -> [B,H*W,C]
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src
def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Intra_Attention_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias, N=8):
        super(Intra_Attention_noZeromap, self).__init__()

        self.N = N
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero):
        b, c, h, w = x.shape
        m = x
        x = self.norm1(x)
        h_pad = self.N - h % self.N if not h % self.N == 0 else 0
        w_pad = self.N - w % self.N if not w % self.N == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), 'reflect')

        b, c, H, W = x.shape
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)
        k = rearrange(k, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)
        v = rearrange(v, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(3, 4)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (N1 N2) (h1 w1) -> b (head c) (h1 N1)  (w1 N2)', head=self.num_heads, N1=self.N, N2=self.N, h1=H // self.N,w1 = W // self.N)

        out = self.project_out(out)

        out = out[:, :, :h, :w]

        V1 = out + m

        return V1

class Inter_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Inter_Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x,zero):
        b, c, h, w = x.shape
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(2, 3)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Intra_Attention_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias, N=8):
        super(Intra_Attention_withZeromap, self).__init__()

        self.N = N
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero_map):
        b, c, h, w = x.shape
        m = x
        x = self.norm1(x)
        h_pad = self.N - h % self.N if not h % self.N == 0 else 0
        w_pad = self.N - w % self.N if not w % self.N == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), 'reflect')
        zero_map = F.pad(zero_map, (0, w_pad, 0, h_pad), 'reflect')

        zero_map = F.interpolate(zero_map, (x.shape[2], x.shape[3]), mode='bilinear')

        zero_map[zero_map <= 0.2] = 0
        zero_map[zero_map > 0.2] = 1
        b, c, H, W = x.shape
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v, v1 = qkv.chunk(4, dim=1)

        q = rearrange(q, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)
        k = rearrange(k, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)
        v = rearrange(v, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)
        v1 = rearrange(v1, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)
        q_zero = rearrange(zero_map, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)
        k_zero = rearrange(zero_map, 'b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)', head=self.num_heads, N1=self.N,N2=self.N)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(3, 4)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        attn_zero = (q_zero @ k_zero.transpose(3, 4)) * self.temperature2
        attn_zero = attn_zero.softmax(dim=-1)
        out_zero = (attn_zero @ v1)

        out = rearrange(out, 'b head c (N1 N2) (h1 w1) -> b (head c) (h1 N1)  (w1 N2)', head=self.num_heads, N1=self.N, N2=self.N, h1=H // self.N,w1 = W // self.N)
        out_zero = rearrange(out_zero, 'b head c (N1 N2) (h1 w1) -> b (head c) (h1 N1)  (w1 N2)', head=self.num_heads, N1=self.N, N2=self.N, h1=H // self.N, w1=W // self.N)

        out = self.project_out(out)
        out_zero = self.project_out(out_zero)

        out = out[:, :, :h, :w]
        out_zero = out_zero[:, :, :h, :w]

        V1 = out + m
        V2 = out_zero + m

        return V1+V2


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention_WxW_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_WxW_withZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.conv = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
        # self.LReLU = nn.LeakyReLU(0.2, inplace=False)
        # self.bn = nn.InstanceNorm2d(dim)

    def forward(self, x, zero_map):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)
        # v = self.bn(self.LReLU(v))
        q = rearrange(q, 'b c h w -> b (c h) w')
        k = rearrange(k, 'b c h w -> b (c h) w')
        # v = rearrange(v, 'b c h w -> b (c h) w', head=self.num_heads)
        q_zero = rearrange(zero_map, 'b c h w -> b (c h) w')
        k_zero = rearrange(zero_map, 'b c h w -> b (c h) w')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-1, -2) @ k) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        attn_zero = (q_zero.transpose(-1, -2) @ k_zero) * self.temperature2
        attn_zero = attn_zero.unsqueeze(1)
        attn_zero = attn_zero.softmax(dim=-1)
        fusion_attn = attn + attn_zero

        out = (v @ fusion_attn)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_WxW_unsqueeze_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_WxW_unsqueeze_withZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero_map):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = torch.mean(q, 2).unsqueeze(2)
        k = torch.max(k, 2)[0].unsqueeze(2)
        # k = torch.mean(k, 2).unsqueeze(2)
        q_zero = torch.mean(zero_map, 2).unsqueeze(2)
        k_zero = torch.mean(zero_map, 2).unsqueeze(2)
        #
        #
        # q_zero = torch.mean(zero_map, 2).unsqueeze(2)
        # k_zero = torch.mean(zero_map, 2).unsqueeze(2)

        q = rearrange(q, 'b c h w -> b (c h) w')
        k = rearrange(k, 'b c h w -> b (c h) w')
        # v = rearrange(v, 'b c h w -> b (c h) w', head=self.num_heads)
        q_zero = rearrange(q_zero, 'b c h w -> b (c h) w')
        k_zero = rearrange(k_zero, 'b c h w -> b (c h) w')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-1, -2) @ k) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        attn_zero = (q_zero.transpose(-1, -2) @ k_zero) * self.temperature2
        attn_zero = attn_zero.unsqueeze(1)
        attn_zero = attn_zero.softmax(dim=-1)
        fusion_attn = attn + attn_zero

        out = (v @ fusion_attn)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_HxH_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_HxH_withZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.LReLU = nn.LeakyReLU(0.2, inplace=False)
        # self.bn = nn.InstanceNorm2d(dim)

    def forward(self, x, zero_map):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)
        # v = self.bn(self.LReLU(v))
        q = rearrange(q, 'b c h w -> b h (c w)')
        k = rearrange(k, 'b c h w -> b h (c w)')
        # v = rearrange(v, 'b (head c) h w -> b head c h w')
        q_zero = rearrange(zero_map, 'b c h w -> b h (c w)')
        k_zero = rearrange(zero_map, 'b c h w -> b h (c w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        attn_zero = (q_zero @ k_zero.transpose(-1, -2)) * self.temperature2
        attn_zero = attn_zero.unsqueeze(1)
        attn_zero = attn_zero.softmax(dim=-1)
        fusion_attn = attn + attn_zero

        out = (fusion_attn @ v)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_HxH_unsqueeze_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_HxH_unsqueeze_withZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero_map):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = torch.mean(q, 3).unsqueeze(3)
        k = torch.max(k, 3)[0].unsqueeze(3)
        # k = torch.mean(k, 3).unsqueeze(3)
        #

        q_zero = torch.mean(zero_map, 3).unsqueeze(3)
        k_zero = torch.mean(zero_map, 3).unsqueeze(3)

        q = rearrange(q, 'b c h w -> b h (c w)')
        k = rearrange(k, 'b c h w -> b h (c w)')
        # v = rearrange(v, 'b (head c) h w -> b head c h w')
        q_zero = rearrange(q_zero, 'b c h w -> b h (c w)')
        k_zero = rearrange(k_zero, 'b c h w -> b h (c w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        attn_zero = (q_zero @ k_zero.transpose(-1, -2)) * self.temperature2
        attn_zero = attn_zero.unsqueeze(1)
        attn_zero = attn_zero.softmax(dim=-1)
        fusion_attn = attn + attn_zero

        out = (fusion_attn @ v)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_WxW_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_WxW_noZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.conv = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
        # self.LReLU = nn.LeakyReLU(0.2, inplace=False)
        # self.bn = nn.InstanceNorm2d(dim)

    def forward(self, x):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b (c h) w')
        k = rearrange(k, 'b c h w -> b (c h) w')
        # v = rearrange(v, 'b (head c) h w -> b c h w', head=self.num_heads)
        # v = self.bn(self.LReLU(v))
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-1, -2) @ k) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        out = (v @ attn)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_WxW_unsqueeze_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_WxW_unsqueeze_noZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = torch.mean(q, 2).unsqueeze(2)
        # k = torch.mean(k, 2).unsqueeze(2)
        k = torch.max(k, 2)[0].unsqueeze(2)

        q = rearrange(q, 'b c h w -> b (c h) w')
        k = rearrange(k, 'b c h w -> b (c h) w')
        # v = rearrange(v, 'b (head c) h w -> b c h w', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-1, -2) @ k) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        out = (v @ attn)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_HxH_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_HxH_noZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.LReLU = nn.LeakyReLU(0.2, inplace=False)
        # self.bn = nn.InstanceNorm2d(dim)
    def forward(self, x):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)
        # v = self.bn(self.LReLU(v))
        q = rearrange(q, 'b c h w -> b h (c w)')
        k = rearrange(k, 'b c h w -> b h (c w)')
        # v = rearrange(v, 'b c h w -> b head c h w', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_HxH_unsqueeze_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_HxH_unsqueeze_noZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = torch.mean(q, 3).unsqueeze(3)
        # k = torch.mean(k, 3).unsqueeze(3)
        k = torch.max(k, 3)[0].unsqueeze(3)

        q = rearrange(q, 'b c h w -> b h (c w)')
        k = rearrange(k, 'b c h w -> b h (c w)')
        # v = rearrange(v, 'b c h w -> b head c h w', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # out = rearrange(out, 'b head c h w -> b (head c) h w', head=self.num_heads)

        out = self.project_out(out)

        V1 = out + m

        return V1


class Attention_CWxCW_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_CWxCW_withZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero_map):
        m = x
        _, c_, _, _ = x.shape
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b (c w) h')
        k = rearrange(k, 'b c h w -> b (c w) h')
        v = rearrange(v, 'b c h w -> b h (c w)')
        # v = rearrange(v, 'b c h w -> b (c h) w', head=self.num_heads)
        q_zero = rearrange(zero_map, 'b c h w -> b (c w) h')
        k_zero = rearrange(zero_map, 'b c h w -> b (c w) h')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        # attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        attn_zero = (q_zero @ k_zero.transpose(-1, -2)) * self.temperature2
        # attn_zero = attn_zero.unsqueeze(1)
        attn_zero = attn_zero.softmax(dim=-1)
        fusion_attn = attn + attn_zero

        out = (v @ fusion_attn)
        out = rearrange(out, 'b h (c w) -> b c h w',c=c_)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_CHxCH_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_CHxCH_withZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero_map):
        m = x
        _, c_, _, _ = x.shape
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b (c h) w')
        k = rearrange(k, 'b c h w -> b (c h) w ')
        v = rearrange(v, 'b c h w -> b w (c h)')
        # v = rearrange(v, 'b c h w -> b (c h) w', head=self.num_heads)
        q_zero = rearrange(zero_map, 'b c h w -> b (c h)w')
        k_zero = rearrange(zero_map, 'b c h w -> b (c h)w')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        # attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        attn_zero = (q_zero @ k_zero.transpose(-1, -2)) * self.temperature2
        # attn_zero = attn_zero.unsqueeze(1)
        attn_zero = attn_zero.softmax(dim=-1)
        fusion_attn = attn + attn_zero

        out = (v @ fusion_attn)
        out = rearrange(out, 'b w (c h) -> b c h w', c=c_)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_CWxCW_withnoZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_CWxCW_withnoZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        # self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        m = x
        _, c_, _, _ = x.shape
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b (c w) h')
        k = rearrange(k, 'b c h w -> b (c w) h')
        v = rearrange(v, 'b c h w -> b h (c w)')
        # v = rearrange(v, 'b c h w -> b (c h) w', head=self.num_heads)
        # q_zero = rearrange(zero_map, 'b c h w -> b (c w) h')
        # k_zero = rearrange(zero_map, 'b c h w -> b (c w) h')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        # attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        # attn_zero = (q_zero @ k_zero.transpose(-1, -2)) * self.temperature2
        # # attn_zero = attn_zero.unsqueeze(1)
        # attn_zero = attn_zero.softmax(dim=-1)
        # fusion_attn = attn + attn_zero

        out = (v @ attn)
        out = rearrange(out, 'b h (c w) -> b c h w',c=c_)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_CHxCH_withnoZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_CHxCH_withnoZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        # self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        m = x
        _, c_, _, _ = x.shape
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b (c h) w')
        k = rearrange(k, 'b c h w -> b (c h) w ')
        v = rearrange(v, 'b c h w -> b w (c h)')
        # v = rearrange(v, 'b c h w -> b (c h) w', head=self.num_heads)
        # q_zero = rearrange(zero_map, 'b c h w -> b (c h)w')
        # k_zero = rearrange(zero_map, 'b c h w -> b (c h)w')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature1
        # attn = attn.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        # attn_zero = (q_zero @ k_zero.transpose(-1, -2)) * self.temperature2
        # # attn_zero = attn_zero.unsqueeze(1)
        # attn_zero = attn_zero.softmax(dim=-1)
        # fusion_attn = attn + attn_zero

        out = (v @ attn)
        out = rearrange(out, 'b w (c h) -> b c h w', c=c_)

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_CxHxH_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_CxHxH_noZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        k = rearrange(k, 'b c h w -> b c w h')
        v = rearrange(v, 'b c h w -> b c w h')
        # v = rearrange(v, 'b (head c) h w -> b c h w', head=self.num_heads)
        # v = self.bn(self.LReLU(v))
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k) * self.temperature1
        attn = attn.softmax(dim=-1)

        out = (v @ attn)
        out = rearrange(out, 'b c w h -> b c h w')

        out = self.project_out(out)

        V1 = out + m

        return V1

class Attention_CxWxW_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_CxWxW_noZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones( 1, 1))
        self.temperature2 = nn.Parameter(torch.ones( 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c w h')
        # v = rearrange(v, 'b (head c) h w -> b c h w', head=self.num_heads)
        # v = self.bn(self.LReLU(v))
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k) * self.temperature1
        attn = attn.softmax(dim=-1)

        out = (v @ attn)
        # out = rearrange(out, 'b c w h -> b c h w')

        out = self.project_out(out)

        V1 = out + m

        return V1

class DeepTransformerBlock_withnoZeromap(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.67, bias=True, LayerNorm_type='WithBias',N=4):
        super(DeepTransformerBlock_withnoZeromap, self).__init__()
        self.attn_WxW = Attention_WxW_noZeromap(dim, num_heads, bias)
        self.attn_HxH = Attention_HxH_noZeromap(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, zero_map):
        m = self.attn_WxW(x)
        z = self.attn_HxH(m)
        out = z + self.ffn(self.norm2(z))

        return out

class TransformerBlock_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.67, bias=True, LayerNorm_type='WithBias',N=4):
        super(TransformerBlock_withZeromap, self).__init__()
        self.attn_intra = Intra_Attention_withZeromap(dim, num_heads, bias, N=N)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, zero_map):
        z = self.attn_intra(x, zero_map)
        out = z + self.ffn(self.norm2(z))

        return out

class TransformerBlock_withnoZeromap(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.67, bias=True, LayerNorm_type='WithBias',N=4):
        super(TransformerBlock_withnoZeromap, self).__init__()
        self.attn_intra = Intra_Attention_noZeromap(dim, num_heads, bias, N=N)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, zero_map):
        z = self.attn_intra(x)
        out = z + self.ffn(self.norm2(z))

        return out


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.conv(x)
        return out

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.deconv(x)
        return out

class BUDL(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True
                 , img_range=1., trainning=True,
                 **kwargs):
        super(BUDL, self).__init__()
        self.trainning = trainning
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        p = 1
        #####################################################################################################
        ############################## 1, enlightenGAN head without attention ###############################
        self.conv1_1 = nn.Conv2d(3, 32, 3, stride=1, padding=p)
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.InstanceNorm2d(32)

        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.InstanceNorm2d(64)

        self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.InstanceNorm2d(64)

        self.conv4_2 = nn.Conv2d(64, 64, 3, stride=1, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.InstanceNorm2d(64)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(64, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)         #  6
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.Tran1 = DeepTransformerBlock_withnoZeromap(dim=int(embed_dim), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)
        self.Tran2 = DeepTransformerBlock_withnoZeromap(dim=int(embed_dim), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)
        self.Tran3 = DeepTransformerBlock_withnoZeromap(dim=int(embed_dim), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)
        self.Tran4 = DeepTransformerBlock_withnoZeromap(dim=int(embed_dim), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)

        self.Tran1u = TransformerBlock_withnoZeromap(dim=int(32), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)
        self.Tran2u = TransformerBlock_withnoZeromap(dim=int(64), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)
        self.Tran3u = TransformerBlock_withnoZeromap(dim=int(64), num_heads=4,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=8)
        self.Tran4u = TransformerBlock_withnoZeromap(dim=int(64), num_heads=8,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=8)

        self.Tran1d = TransformerBlock_withnoZeromap(dim=int(32), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)
        self.Tran2d = TransformerBlock_withnoZeromap(dim=int(64), num_heads=2,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=4)
        self.Tran3d = TransformerBlock_withnoZeromap(dim=int(64), num_heads=4,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=8)
        self.Tran4d = TransformerBlock_withnoZeromap(dim=int(64), num_heads=8,
                                             ffn_expansion_factor=2, bias=False,
                                             LayerNorm_type='WithBias', N=8)

        self.conv_last = nn.Conv2d(embed_dim, 64, 3, 1, 1)
##############################################################################################
        self.conv6_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2 = nn.InstanceNorm2d(64)
#####################################################################################################

        self.conv7_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.InstanceNorm2d(64)
##############################################################################################

        self.upsampler0 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsampler1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsampler2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv8_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.InstanceNorm2d(64)

        self.conv9_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_2 = nn.InstanceNorm2d(32)

        self.conv10 = nn.Conv2d(32, 3, 1)
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

        self.down1_1 = Downsample(32, 32 *2)
        self.down1_2 = Downsample(64, 32 * 2)
        self.down1_3 = Downsample(64, 64 * 1)
        self.down1_4 = nn.Conv2d(64, 64, 3, 1, 1)

        self.up1_4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up1_3 = Upsample(64, 64)
        self.up1_2 = Upsample(64, 64)
        self.up1_1 = Upsample(64, 32)

        self.reduce_chan_level00 = nn.Conv2d(128, 64, kernel_size=1, bias=True)
        self.reduce_chan_level11 = nn.Conv2d(128, 64, kernel_size=1, bias=True)
        self.reduce_chan_level22 = nn.Conv2d(128, 64, kernel_size=1, bias=True)
        self.reduce_chan_level33 = nn.Conv2d(64, 32, kernel_size=1, bias=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, input, zero):
        input : torch.Size([2, 3, 256, 256])
        flag = 0

        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)

        x = self.conv1_1(input)

        x = self.bn1_2(self.LReLU1_2(self.conv1_2(x))) # torch.Size([2, 32, 256, 256])
        zero_1 = downmask1(zero, 1, 1)
        conv1 = self.Tran1u(x, zero_1)
        x = self.down1_1(conv1)

        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))  # torch.Size([2, 64, 128, 128])
        zero_2 = downmask1(zero, 2, 2)
        conv2 = self.Tran2u(conv2, zero_2)
        x = self.down1_2(conv2)

        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))  #2 128 64 64
        zero_3 = downmask1(zero, 2, 4)
        conv3 = self.Tran3u(conv3, zero_3)
        x = self.down1_3(conv3)

        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))  #2 256 32 32
        zero_4 = downmask1(zero, 2, 8)
        conv4 = self.Tran4u(conv4, zero_4)
        x1 = self.down1_4(conv4)

        ##################################################################
        # for image denoising and JPEG compression artifact reduction
        x_first = self.conv_first(x1)    # torch.Size([2, 180, 64, 64])

        zero_1 = downmask(zero, int(8))
        Tran1 = self.Tran1(x_first, zero_1)
        Tran2 = self.Tran2(Tran1, zero_1)
        Tran3 = self.Tran3(Tran2, zero_1)
        Tran4 = self.Tran4(Tran3, zero_1)

        conv7 = self.conv_last(Tran4)
        x = x1 + conv7
        x = self.up1_4(x)

        up6 = torch.cat([x, conv4], 1)
        x = self.reduce_chan_level00(up6)
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))
        conv6 = self.Tran4d(conv6, None)
        conv7 = self.up1_3(conv6)

        up7 = torch.cat([conv7, conv3], 1)
        up7 = self.reduce_chan_level11(up7)
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(up7)))
        conv7 = self.Tran3d(conv7, None)
        conv8 = self.up1_2(conv7)

        up8 = torch.cat([conv8, conv2], 1)    # torch.Size([2, 128, 128, 128])
        up8 = self.reduce_chan_level22(up8)
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(up8)))
        conv8 = self.Tran2d(conv8, None)
        conv9 = self.up1_1(conv8)

        up9 = torch.cat([conv9, conv1], 1)
        up9 = self.reduce_chan_level33(up9)
        conv9 = self.bn9_2(self.LReLU9_2(self.conv9_2(up9)))
        conv9 = self.Tran1d(conv9, None)
        latent = self.conv10(conv9) # torch.Size([2, 3, 256, 256])

        latent = latent + input

        latent = self.tanh(latent)
        output = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)

        return output

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample1.flops()
        return flops

def downmask(mask, k):
    _, _, h, w = mask.shape

    m = torch.squeeze(mask, 1)
    b, h, w = m.shape
    m = torch.chunk(mask, b, dim=0)[0]
    m = torch.squeeze(m)
    size = (int(w // k), int(h // k))
    mc = m.cpu()
    m_n = mc.numpy()
    m = cv2.resize(m_n, size, interpolation=cv2.INTER_LINEAR)
    m[m <= 0.2] = 0
    m[m > 0.2] = 1
    m = torch.from_numpy(m)

    m = torch.unsqueeze(m, 0)
    m = m.expand(b, 64, -1, -1)
    out_mask = m.cuda()
    return out_mask

def downmask1(mask, k, s):
    _, _, h, w = mask.shape

    m = torch.squeeze(mask, 1)
    b, h, w = m.shape
    m = torch.chunk(mask, b, dim=0)[0]
    m = torch.squeeze(m)
    size = (int(w // s), int(h // s))
    mc = m.cpu()
    m_n = mc.numpy()
    m = cv2.resize(m_n, size, interpolation=cv2.INTER_LINEAR)
    m[m <= 0.2] = 0
    m[m > 0.2] = 1
    m = torch.from_numpy(m)

    m = torch.unsqueeze(m, 0)
    m = m.expand(b, 32 * k, -1, -1)

    out_mask = m.cuda()
    return out_mask

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 600
    width = 400
    model = BUDL(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[2, 2, 2, 2],
                   embed_dim=64, num_heads=[6, 6, 6, 6], mlp_ratio=2, resi_connection='dbconv',upsampler='pixelshuffledirect')
    print(model)

    x = torch.randn((1, 3, height, width))
    zero = torch.randn((1, 1, height, width))
    x = model(x,zero)
    # print(x.shape)

    print("Parameters of full network %.4f " % (sum([m.numel() for m in model.parameters()]) / 1e6))
    # fft_blocks = nn.ModuleList([FFCResnetBlock(64) for _ in range(4)])