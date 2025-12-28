import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.model_utils import RMSNorm, FeedForward
# from timm.models.crossvit import CrossAttention


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=RMSNorm,
            dtype=torch.float32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype)
        self.q_norm = norm_layer(self.head_dim, dtype=dtype) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, dtype=dtype) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, dtype=dtype)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Fused Attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=RMSNorm,
            dtype=torch.float32,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 为查询（queries）定义线性层
        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias, dtype=dtype)

        # 为键（keys）和值（values）定义线性层
        self.key_value_proj = nn.Linear(dim, dim * 2, bias=qkv_bias, dtype=dtype)

        self.q_norm = norm_layer(self.head_dim, dtype=dtype) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, dtype=dtype) if qk_norm else nn.Identity()

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, dtype=dtype)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        B_q, N_q, C_q = query.shape
        B_kv, N_kv, C_kv = key_value.shape

        # 将输入张量重塑为多头的形式
        q = self.query_proj(query).reshape(B_q, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.key_value_proj(key_value).reshape(B_kv, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        # Fused Attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop,
        )

        # 重塑输出张量
        x = x.permute(0, 2, 1, 3).reshape(B_q, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    # [Self Attention, Cross Attention, FeedForward] with Residual
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0.,
                 norm_layer=RMSNorm, dtype=torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer, dtype)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer, dtype)
        self.ffn = FeedForward(dim, mlp_ratio, dtype)

        self.self_norm = RMSNorm(dim, dtype)
        self.cross_norm = RMSNorm(dim, dtype)
        self.ff_norm = RMSNorm(dim, dtype)

    def forward(self, features, anchor_embeds):
        # Self-Attention
        features = self.self_norm(features + self.self_attn(features))
        # Cross Attention
        features = self.cross_norm(features + self.cross_attn(features, anchor_embeds))
        # FFN
        features = self.ff_norm(features + self.ffn(features))
        return features
