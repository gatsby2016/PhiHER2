from typing import Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

from .transformer_utils import Attention, FeedForward, PreNorm
from .model_utils import Attn_Net_Gated


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


class Transformer(nn.Module):
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
        super(Transformer, self).__init__()
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

    def forward(self, x, coords=None, register_hook=False, label=None):
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

        logits = self.mlp_head(x)

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.argmax(Y_prob, dim= 1)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]

        results_dict = {}
        return logits, Y_prob, Y_hat, _, results_dict




"""
prototypical MIL reproduced.
"""
class ProtoMIL(nn.Module):
    def __init__(self, feature_size=2048, hidden_size=256, cls_hidden_size=16, 
                 num_cluster=64, topk_num = 10, instance_loss_fn=nn.CrossEntropyLoss(), instance_eval=False,
                 dropout=0.25, output_class=2):
        super(ProtoMIL, self).__init__()
        self.num_cluster = num_cluster
        self.topk_num = topk_num
        self.instance_loss_fn = instance_loss_fn
        self.instance_eval = instance_eval

        self.fc3_topk = nn.Linear(feature_size, 2)    
        self.fc2_metric_learning = nn.Linear(feature_size, hidden_size)

        self.rho = nn.Linear(num_cluster, cls_hidden_size)
        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden_size, output_class)
        )

    def forward(self, x_path, prototype=None, **kwargs):
        inst_logit = self.fc3_topk(x_path) # N x 2 score for top-k selector
        probs = F.softmax(inst_logit, dim=1)

        origin = False
        if origin:
            _, m_indices = torch.sort(probs, 0,
                                     descending=True)
            top_idx = m_indices[:self.topk_num, 1]
        else: # 与上面等价
            top_idx = torch.topk(probs[:, 1], self.topk_num)[1] # (N, top_num)

        m_features = torch.index_select(x_path, dim=0, index=top_idx)

        ## Metric learning
        f = self.fc2_metric_learning(m_features)
        # prototype = x_path[:self.num_cluster,:]
        p = self.fc2_metric_learning(prototype)

        ## Euclidean
        similarity = self.Euclidean_Similarity(f, p)
        cmax, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity / cmax

        ## SLN Cosine
        # Cosine
        #similarity = torch.mm(f, p.transpose(0,1))
        #similarity = similarity / torch.norm(f, p=2, dim=1, keepdim=True) / torch.norm(p, p=2)

        sim_coding = torch.mean(similarity, dim=0, keepdim=True)
        bag_logits = self.classifier(self.rho(sim_coding))
        
        Y_prob = F.softmax(bag_logits, dim = 1)
        Y_hat = torch.argmax(Y_prob, dim= 1)

        if self.instance_eval and "label" in kwargs:
            _, max_id = torch.max(probs[:, 1], 0)
            inst_logit = inst_logit[[max_id], :]
            instance_loss = self.instance_loss_fn(inst_logit, kwargs["label"])
            results_dict = {'instance_loss': instance_loss}
        else:
            results_dict = {}
        return bag_logits, Y_prob, Y_hat, sim_coding, results_dict

    @staticmethod
    def Euclidean_Similarity(tensor_a, tensor_b):
        device = tensor_a.device
        output = torch.zeros(tensor_a.size(0), tensor_b.size(0), device=device)
        for i in range(tensor_a.size(0)):
            output[i, :] = torch.pairwise_distance(tensor_a[i, :], tensor_b)
        return output
    

#################################################
# borrowed from https://github.com/benbergner/ips/blob/main/architecture/transformer.py

def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class ScaledDotProductAttention(nn.Module):
    ''' Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def compute_attn(self, q, k):
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))

        return attn

    def forward(self, q, k, v):
        
        attn = self.compute_attn(q, k)
        output = torch.matmul(attn, v)

        return output


class MultiHeadCrossAttention(nn.Module):
    ''' Multi-head cross-attention module '''

    def __init__(self, num_cluster, num_head, input_dim, dim_k, dim_v, init_query=False, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_cluster = num_cluster

        self.num_head = num_head
        self.dim_k = dim_k
        self.dim_v = dim_v

        if init_query:
            query = nn.Parameter(torch.empty((1, self.num_cluster, self.input_dim)))
            q_init_val = math.sqrt(1 / self.dim_k)
            nn.init.uniform_(query, a=-q_init_val, b=q_init_val)
            self.query = query
        else:
            self.query = None

        self.q_w = nn.Linear(self.input_dim, self.num_head * self.dim_k, bias=False)
        self.k_w = nn.Linear(self.input_dim, self.num_head * self.dim_k, bias=False)
        self.v_w = nn.Linear(self.input_dim, self.num_head * self.dim_v, bias=False)

        self.fc = nn.Linear(num_head * self.dim_v, self.input_dim, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=self.dim_k ** 0.5,
            attn_dropout=attn_dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.input_dim, eps=1e-6)


    def forward(self, x, prototype=None): # x is feats
        if self.query is not None and prototype is None:
            query = self.query
        elif self.query is None and prototype is not None:
            query = prototype
        else:
            raise NotImplementedError
        
        B, len_seq = x.shape[:2]
        # project and separate heads
        q = self.q_w(query).view(1, self.num_cluster, self.num_head, self.dim_k)
        k = self.k_w(x).view(B, len_seq, self.num_head, self.dim_k)
        v = self.v_w(x).view(B, len_seq, self.num_head, self.dim_v)

        # transpose for attention dot product: B x num_head x len_seq (or num_cluster) x dim_k (or dim_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # cross-attention
        x = self.attention(q, k, v) # x is B x num_head x num_cluster x dim_v

        # transpose to: B x num_cluster x num_head x dim_v
        # concat heads: B x num_cluster x (num_head * dim_v)
        x = x.transpose(1, 2).contiguous().view(B, self.num_cluster, -1) # combine heads       
        x = self.dropout(self.fc(x))
        # residual connection + layernorm
        x += query
        x = self.layer_norm(x) # B x num_cluster x input_dim

        if self.query is not None and prototype is None:
            self.query = query
        
        return x

    def get_attn(self, x, prototype=None):
        if self.query is not None and prototype is None:
            query = self.query
        elif self.query is None and prototype is not None:
            query = prototype
        else:
            raise NotImplementedError
        
        B, len_seq = x.shape[:2]

        q = self.q_w(query).view(1, self.num_cluster, self.num_head, self.dim_k)
        k = self.k_w(x).view(B, len_seq, self.num_head, self.dim_k)

        q, k = q.transpose(1, 2), k.transpose(1, 2)

        attn = self.attention.compute_attn(q, k)

        return attn
    

class MLP(nn.Module):
    ''' MLP consisting of two feed-forward layers '''

    def __init__(self, D, D_inner, dropout=0.1):
        super().__init__()
        
        self.w_1 = nn.Linear(D, D_inner)
        self.w_2 = nn.Linear(D_inner, D)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.w_2(torch.relu(self.w_1(inputs)))
        x = self.dropout(x)

        return self.layer_norm(x+inputs)


"""
cluster-Prototypes-based Transformer for cls.
"""
class ProtoTransformer(nn.Module):
    def __init__(self, feature_size=2048, hidden_size=512, num_head=4,  ff_inner_size=512,
                 num_cluster=64, topk_num = 10, 
                 attn_dropout=0.1, dropout=0.25, output_class=2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cross_attn = MultiHeadCrossAttention(num_cluster, num_head, input_dim=feature_size, 
                                                  dim_k=hidden_size//num_head, dim_v=hidden_size//num_head, 
                                                  attn_dropout=attn_dropout, dropout=dropout)
        self.mlp = MLP(feature_size, ff_inner_size, dropout=dropout)

        self.transf = Transformer(num_classes=output_class, input_dim=feature_size, depth=1, 
                            heads=num_head, dim_head=64, hidden_dim=512, 
                            pool='cls', dropout=dropout, emb_dropout=0., pos_enc=None)
        
    def get_scores(self, x, prototype=None):
        attn = self.cross_attn.get_attn(x, prototype)
        # Average scores over heads and tasks
        # Average over tasks is only required for multi-task learning (mnist).
        return attn.mean(dim=1).transpose(1, 2) # B x num_inst x num_prototype

    def forward(self, x_feats, prototype=None, label=None):
        x_feats = x_feats.unsqueeze(0)
        x = self.cross_attn(x_feats, prototype) # B x num_cluster x feature_size
        x = self.mlp(x) # B x num_cluster x feature_size
        logits, Y_prob, Y_hat, _, results_dict = self.transf(x.squeeze(0))

        return logits, Y_prob, Y_hat, _, results_dict