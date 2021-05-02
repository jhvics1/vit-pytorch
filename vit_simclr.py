import torch.nn as nn
from einops import rearrange, repeat


class ViTSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ViTSimCLR, self).__init__()

        self.backbone = base_model

        # add mlp projection head
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.backbone.dim),
            nn.Linear(self.backbone.dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_dim))

        # self.pred = nn.Sequential(
        #     # nn.LayerNorm(128),
        #     nn.Linear(128, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128))

    def forward(self, x, depth=None):
        x = self.backbone(x, depth)
        return self.mlp(x)

    def layer_repr_mlp(self):
        return self.backbone.layer_repr()

    def layer_repr_pred(self):
        # layer_repr = self.backbone.layer_repr()
        # d, _, _ = layer_repr.shape
        # layer_repr = self.pred(rearrange(layer_repr, 'd b c -> (d b) c'))
        # return rearrange(layer_repr, '(d b) c -> d b c', d=d)
        return self.backbone.layer_pred()
