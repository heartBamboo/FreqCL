import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from backbone.model_utils import _init_weights
from backbone.attn import CrossAttentionBlock


class AnchorEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, num_heads, num_classes, proj_drop=0., dtype=torch.float32):
        super().__init__()
        self.out_projection = nn.Sequential(
            nn.Linear(hidden_dim*2, out_dim, bias=False),
            nn.Dropout(proj_drop)
        )

        self.cross_attn = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, qk_norm=True, attn_drop=0.1, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.class_anchors = nn.Parameter(torch.empty((1, num_classes, hidden_dim), dtype=dtype))
        trunc_normal_(self.class_anchors.data, std=0.02)
        self.class_anchors._no_weight_decay = True

        self.apply(_init_weights)


    def forward(self, features):
        N, H = features.shape
        C = self.class_anchors.shape[1]

        anchor_embeds = self.class_anchors.expand(N, C, H)

        anchor_indices = F.cosine_similarity(features.unsqueeze(1), anchor_embeds, dim=-1).argmax(dim=-1)
        selected_anchors = anchor_embeds[torch.arange(N, device=anchor_indices.device), anchor_indices]
        features = torch.cat((selected_anchors, features), dim=1)
        return self.out_projection(features)


if __name__ == '__main__':
    batch_size = 2
    hidden_dim = 32

    backbone_output = torch.randn((batch_size, 1, hidden_dim))

    model = AnchorEncoder(2, 2, 4)
    model.eval()

    with torch.no_grad():
        out = model(backbone_output)
        print(out.size())
