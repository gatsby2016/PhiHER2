import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

from .transformer_utils import Attention, FeedForward, PreNorm
from .model_utils import Attn_Net_Gated

class BaseAggregator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, hidden_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, hidden_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x


class Transformer(BaseAggregator):
    def __init__(
        self,
        *,
        num_classes,
        input_dim=2048,
        depth=2,
        heads=8,
        dim_head=64,
        hidden_dim=512,
        pool='cls',
        dropout=0.,
        emb_dropout=0.,
        pos_enc=None,
    ):
        super(BaseAggregator, self).__init__()
        assert pool in {
            'cls', 'mean', 'mil'
        }, 'pool type must be either cls (class token), mean (mean pooling) or mil'

        emb_dim = heads * dim_head
        self.projection = nn.Sequential(nn.Linear(input_dim, emb_dim, bias=True), nn.ReLU())
        self.transformer = TransformerBlocks(emb_dim, depth, heads, dim_head, hidden_dim, dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes))

        self.pool = pool
        if self.pool == "mil":
            self.mil_agg = nn.Sequential(*[nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout),
                                           Attn_Net_Gated(L=emb_dim, D=emb_dim, dropout=dropout, n_classes=1)])
        elif self.pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.pos_enc = pos_enc

    def forward(self, x, coords=None, register_hook=False, **kwargs):
        x = x.unsqueeze(0)
        b, _, _ = x.shape

        x = self.projection(x)

        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'mil':
            A, h_path = self.mil_agg(x.squeeze())
            A = torch.transpose(A, 1, 0)
            A = F.softmax(A, dim=1) 
            x = torch.mm(A, h_path)

        logits = self.mlp_head(self.norm(x))

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.argmax(Y_prob, dim= 1)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]

        results_dict = {"embedding": x}
        return logits, Y_prob, Y_hat, _, results_dict



if __name__ == "__main__":
    transformer = Transformer(num_classes=2)
    transformer(torch.rand(1, 1, 2048))