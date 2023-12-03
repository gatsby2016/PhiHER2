import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import Attn_Net_Gated, initialize_weights


"""
Implement Attention MIL for the unimodal (WSI only) and multimodal setting (pathways + WSI). The combining of modalities 
can be done using bilinear fusion or concatenation. 
"""

################################
# Attention MIL Implementation #
################################
class ABMIL(nn.Module):
    def __init__(self, size_arg = "ccl2048", dropout=0.25, n_classes=4, top_num_inst=None, device="cpu"):
        r"""
        Attention MIL Implementation

        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(ABMIL, self).__init__()
        self.top_num_inst = top_num_inst
        self.size_dict_path = {"ViT_small": [768, 512, 256], 
                               "ViT_big": [768, 512, 384],
                               "resnet_small": [1024, 512, 256], 
                               "resnet_big": [1024, 512, 384],
                                "ccl2048": [2048, 512, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        if size[0] > 1024:
            fc = [nn.Linear(size[0], 1024), nn.ReLU(), nn.Dropout(dropout),
                  nn.Linear(1024, size[1]), nn.ReLU(), nn.Dropout(dropout)]
        else:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[2], n_classes)

        # self.activation = nn.ReLU()
        initialize_weights(self)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    @torch.no_grad()
    def iterative_embed_selection(self, x_path, top_num=1):
        A, _ = self.attention_net(x_path.squeeze())
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        top_idx = torch.topk(A, top_num, dim = -1)[1] # (N, top_num)
        return top_idx

    def forward(self, x_path, attention_only=False, train=True, **kwargs):
        x_path = x_path.squeeze() #---> need to do this to make it work with this set up
        if train and self.top_num_inst is not None and self.top_num_inst < x_path.shape[0]:
            top_idx = self.iterative_embed_selection(x_path=x_path, top_num=self.top_num_inst)
            x_path = x_path[top_idx].squeeze()
        
        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        if attention_only:  
            return A # this is instance-level attention scores
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        logits  = self.classifier(h_path).unsqueeze(0) # logits needs to be a [1 x 4] vector 

        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.argmax(Y_prob, dim= 1)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]

        results_dict = {"embedding": h_path}
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def captum(self, x_wsi):
        x_wsi = x_wsi.squeeze() #---> need to do this to make it work with this set up
        A, h_path = self.attention_net(x_wsi)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        #---> get risk 
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        #---> return risk 
        return risk