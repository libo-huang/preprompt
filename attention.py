import torch
import torch.nn as nn
import torch.nn.functional as F

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads  # 12
        head_dim = dim // num_heads  # 64
        self.scale = head_dim ** -0.5  # 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 768, 2304, True
        self.attn_drop = nn.Dropout(attn_drop)  # 0.0
        self.proj = nn.Linear(dim, dim)  # 768, 768
        self.proj_drop = nn.Dropout(proj_drop)  # 0.0

    def forward(self, x, prompt):
        B, N, C = x.shape  # [24, 197, 768]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [24,197,3,12,64] --> [3,24,12,197,64]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple); [24,12,197,64]

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads; [24,2,5,12,64] --> [2,24,12,5,64]
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads; [24,12,5,64]
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads; [24,12,5,64]

            expected_shape = (B, self.num_heads, C // self.num_heads)  # [24,12,64}
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)  # [24,12,202,64]
            v = torch.cat([value_prefix, v], dim=2)  # [24,12,202,64]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [24,12,197,64] @ [24,12,64,202] --> [24,12,197,202]. [24,12,197,64]@[24,12,64,197]-->[24,12,197,197]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [24,12,197,202]@[24,12,202,64]->[24,12,197,64]->[24,197,12,64]->[24,197,768]. [24,12,197,197]@[24,12,197,64]->[24,12,197,64]->[24,197,768]
        x = self.proj(x)  # [24, 197, 768]
        x = self.proj_drop(x)
        return x

