from typing import Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

from .transformer_utils import Attention, FeedForward, PreNorm
from .model_utils import Attn_Net_Gated, initialize_weights


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
        self.init_query = init_query

        if self.init_query:
            query = nn.Parameter(torch.empty((1, self.num_cluster, self.input_dim)))
            q_init_val = math.sqrt(1 / self.dim_k)
            nn.init.uniform_(query, a=-q_init_val, b=q_init_val)
            self.query = query


        self.q_w = nn.Linear(self.input_dim, self.num_head * self.dim_k, bias=False)
        self.k_w = nn.Linear(self.input_dim, self.num_head * self.dim_k, bias=False)
        self.v_w = nn.Linear(self.input_dim, self.num_head * self.dim_v, bias=False)

        self.fc = nn.Linear(self.num_head * self.dim_v, self.input_dim, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=self.dim_k ** 0.5,
            attn_dropout=attn_dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.input_dim, eps=1e-6)


    def forward(self, x, prototype=None): # x is feats
        if not self.init_query:
            self.query = prototype
        # elif self.init_query and prototype is not None:
        #     self.query = nn.Parameter((self.query.data + prototype.unsqueeze(0))/2, requires_grad=True)
        
        B, len_seq = x.shape[:2]
        # project and separate heads
        q = self.q_w(self.query).view(1, self.num_cluster, self.num_head, self.dim_k)
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
        x += self.query
        x = self.layer_norm(x) # B x num_cluster x input_dim
        
        return x

    def get_attn(self, x, prototype=None):
        if not self.init_query:
            self.query = prototype
        # elif self.init_query and prototype is not None:
        #     self.query = nn.Parameter((self.query.data + prototype.unsqueeze(0))/2, requires_grad=True)
        
        B, len_seq = x.shape[:2]

        q = self.q_w(self.query).view(1, self.num_cluster, self.num_head, self.dim_k)
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
weighted sum classifier for output prototype-based similarity feats
"""
class weightedSumClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_size=256, dropout=0.0, output_class=2, mean_dim=0):
        super().__init__()

        self.mean_dim = mean_dim
        self.mlp_head = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                      nn.ReLU(True),
                                      nn.Dropout(dropout),
                                      nn.Linear(hidden_size, output_class))
    
    def forward(self, x): 
        sim_coding = torch.mean(x, dim=self.mean_dim, keepdim=True)
        
        sim_coding = sim_coding.squeeze().unsqueeze(0)
        b, _ = sim_coding.shape # B x num_cluster (or input_dim)

        logits = self.mlp_head(sim_coding)

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.argmax(Y_prob, dim= 1)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]

        results_dict = {}
        return logits, Y_prob, Y_hat, _, results_dict
    

"""
# Attention MIL Implementation #
"""
class AttenMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_size=256, dropout=0.0, inst_num=None, output_class=2):
        super(AttenMIL, self).__init__()

        self.top_num_inst = inst_num
        self.attention_net = nn.Sequential(*[nn.Linear(input_dim, hidden_size), 
                                             nn.ReLU(), 
                                             nn.Dropout(dropout),
                                             Attn_Net_Gated(L=hidden_size, D=hidden_size, dropout=dropout, n_classes=1)])

        self.classifier = nn.Sequential(*[nn.Linear(hidden_size, hidden_size), 
                                          nn.ReLU(), 
                                          nn.Dropout(dropout), 
                                          nn.Linear(hidden_size, output_class)])
        initialize_weights(self)


    @torch.no_grad()
    def iterative_embed_selection(self, x_path, top_num=1):
        A, _ = self.attention_net(x_path.squeeze())
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        top_idx = torch.topk(A, top_num, dim = -1)[1] # (N, top_num)
        return  torch.index_select(x_path, dim=0, index=top_idx.squeeze())


    def forward(self, x, attention_only=False, test=False, return_select_x=False): 
        x = x.squeeze() #  N x feats_dim

        if not test and self.top_num_inst is not None and self.top_num_inst < x.shape[0]:
            x = self.iterative_embed_selection(x_path=x, top_num=self.top_num_inst) 

        atten_score, h_path = self.attention_net(x) # atten_score: N x 1, h_path: N x hidden_size
        atten_score = torch.transpose(atten_score, 1, 0)
        if attention_only:  
            return atten_score # this is instance-level attention scores
        
        h_path = torch.mm(F.softmax(atten_score, dim=1), h_path) # 1 x hidden_size

        logits  = self.classifier(h_path) # logits: [1 x output_class] vector 

        if return_select_x:
            return logits, x, atten_score
        else:
            Y_prob = F.softmax(logits, dim = 1)
            Y_hat = torch.argmax(Y_prob, dim= 1)
            # Y_hat = torch.topk(logits, 1, dim = 1)[1]

            results_dict = {}
            return logits, Y_prob, Y_hat, atten_score, results_dict
    

class inst_selector(nn.Module):
    def __init__(self, input_dim=2048, inst_num=1, random=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear = nn.Linear(input_dim, 2)
        self.inst_num = inst_num
        self.random = random

    def forward(self, x):
        if self.inst_num >= len(x):
            return x
        
        if self.random:
            device = x.device
            top_idx = torch.Tensor(np.random.choice(len(x), self.inst_num, replace=False)).to(device).type(torch.int64)
        
        else:
            inst_logit = self.linear(x) # N x 2 score for top-k selector
            probs = F.softmax(inst_logit, dim=1)

            top_idx = torch.topk(probs[:, 1], self.inst_num)[1] # (N, top_num)

        x = torch.index_select(x, dim=0, index=top_idx)
        return x
    

"""
cluster-Prototypes-based Transformer for cls.
"""
class ProtoTransformer(nn.Module):
    def __init__(self, feature_size=2048, embed_size=512, hidden_size=256, num_head=4,
                 num_cluster=64, inst_num = None, inst_num_twice=None, random_inst=False,
                 attn_dropout=0.1, dropout=0.25, output_class=2,
                 cls_method=None, aux_loss_fn=nn.CrossEntropyLoss(), abmil_branch=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_method = cls_method
        self.inst_num = inst_num
        self.abmil_branch = abmil_branch
        self.inst_num_twice = inst_num_twice

        self.projection = nn.Sequential(nn.Linear(feature_size, embed_size, bias=True), nn.ReLU())
        # self.prototye_projection = nn.Sequential(nn.Linear(feature_size, embed_size, bias=True), nn.ReLU())

        if self.inst_num_twice is not None: # inst_selection_twice:
            self.inst_selection = inst_selector(input_dim=embed_size, inst_num = self.inst_num_twice, random=random_inst)

        if self.abmil_branch:
            self.aux_loss_fn = aux_loss_fn
            self.abmil = AttenMIL(embed_size, hidden_size, dropout, inst_num=self.inst_num, output_class=output_class)

        self.cross_attn = MultiHeadCrossAttention(num_cluster, num_head, input_dim=embed_size, 
                                                  dim_k=embed_size//num_head, dim_v=embed_size//num_head, init_query=False,
                                                  attn_dropout=attn_dropout, dropout=dropout)
        self.mlp = MLP(embed_size, hidden_size, dropout=dropout)

        if self.cls_method == "trans":
            self.transf = Transformer(num_classes=output_class, input_dim=embed_size, depth=1, 
                                      heads=num_head, dim_head=64, hidden_dim=hidden_size, 
                                      pool='cls', dropout=dropout, emb_dropout=0., pos_enc=None)
        elif self.cls_method == "cls_keep_prototype_dim":
            self.transf = weightedSumClassifier(num_cluster, hidden_size, dropout, output_class, mean_dim=1)

        elif self.cls_method == "cls_keep_embedd_dim":
            self.transf = weightedSumClassifier(embed_size, hidden_size, dropout, output_class, mean_dim=0)
        elif self.cls_method == "cls_abmil":
            self.transf = AttenMIL(embed_size, hidden_size, dropout, inst_num=None, output_class=output_class)
        
    def get_scores(self, x, prototype=None):
        x = self.projection(x.unsqueeze(0))
        prototype = self.projection(prototype)
        attn = self.cross_attn.get_attn(x, prototype)
        # Average scores over heads and tasks
        # Average over tasks is only required for multi-task learning (mnist).
        return attn.mean(dim=1).transpose(1, 2) # B x num_inst x num_prototype

    def forward(self, x_feats, prototype=None, label=None):
        x_feats = self.projection(x_feats)
        prototype = self.projection(prototype)

        if self.abmil_branch and label is not None:
            abmil_logit, x_feats, _ = self.abmil(x_feats, return_select_x=True)
            aux_abmil_loss = self.aux_loss_fn(abmil_logit, label)

        if self.inst_num_twice is not None and self.inst_num_twice < x_feats.shape[0]: # inst_selection_twice:
            # x_feats = self.abmil.iterative_embed_selection(x_path=x_feats, top_num=self.inst_num_twice)     
            x_feats = self.inst_selection(x_feats)          

        x = self.cross_attn(x_feats.unsqueeze(0), prototype) # B x num_cluster x feature_size
        x = self.mlp(x) # B x num_cluster x feature_size

        logits, Y_prob, Y_hat, _, results_dict = self.transf(x.squeeze(0))

        if self.abmil_branch and label is not None:
            results_dict.update({'instance_loss': aux_abmil_loss})

        return logits, Y_prob, Y_hat, _, results_dict