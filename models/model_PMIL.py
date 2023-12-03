import torch.nn as nn
import torch.nn.functional as F
import torch
from .resnet_custom import resnet50_baseline


class IClassifier(nn.Module):
    def __init__(self, mil_model, feature_size=512, out_class=2):
        super(IClassifier, self).__init__()
        temp = resnet50_baseline(True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(mil_model, map_location='cpu')
        temp.load_state_dict(ch['state_dict'])
        self.features = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self, features):
        x = self.fc(features)
        return features, x


class BClassifier(nn.Module):
    def __init__(self, num_cluster, feature_size=512, input_size=128, output_class=2, dropout=0):
        super(BClassifier, self).__init__()
        self.linear = nn.Linear(feature_size, input_size)

        self.classifier = nn.Sequential(
            nn.Linear(num_cluster, 128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, output_class)
        )


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, probs, prototype):
        test, m_indices = torch.sort(probs, 0,
                                     descending=True)
        test_array = test.detach().cpu().numpy()
        m_features = torch.index_select(features, dim=0,
                                        index=m_indices[:5, 1])

        ## Metric learning
        f = self.linear(m_features)
        p = self.linear(prototype)

        ## Euclidean
        similarity = Euclidean_Similarity(f, p)
        cmax, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity / cmax

        ## SLN Cosine
        # Cosine
        #similarity = torch.mm(f, p.transpose(0,1))
        #similarity = similarity / torch.norm(f, p=2, dim=1, keepdim=True) / torch.norm(p, p=2)

        coding = torch.mean(similarity, dim=0, keepdim=True)
        x = self.classifier(coding)
        return x, coding


class PBMIL(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(PBMIL, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x, prototype):
        features, score = self.i_classifier(x)
        probs = F.softmax(score, dim=1)
        output, codings = self.b_classifier(features, probs, prototype)
        return score, output, codings


"""
prototypical MIL reproduced.
"""
class ProtoMIL(nn.Module):
    def __init__(self, feature_size=2048, hidden_size=256, cls_hidden_size=16, 
                 num_cluster=64, topk_num = 10, instance_loss_fn=nn.CrossEntropyLoss(), instance_eval=False,
                 dropout=0.25, output_class=2, similarity_method="Euclidean", aggregation_method="mean"):
        super(ProtoMIL, self).__init__()
        self.num_cluster = num_cluster
        self.topk_num = topk_num
        self.instance_loss_fn = instance_loss_fn
        self.instance_eval = instance_eval
        self.similarity_method = similarity_method
        self.aggregation_method = aggregation_method

        self.fc3_topk = nn.Linear(feature_size, 2)    
        self.fc2_metric_learning = nn.Linear(feature_size, hidden_size)

        if self.aggregation_method == "weightedsum_prototype":
            self.rho = nn.Linear(hidden_size, cls_hidden_size)
        else:
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
        self.topk_num = min(self.topk_num, len(x_path))
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
        if self.similarity_method == "Euclidean":
            similarity = self.Euclidean_Similarity(f, p)
            cmax, _ = torch.max(similarity, dim=1, keepdim=True)
            similarity = similarity / cmax
        
        ## SLN Cosine
        elif self.similarity_method == "Cosine":
            similarity = torch.mm(f, p.transpose(0,1))
            similarity = similarity / torch.norm(f, p=2, dim=1, keepdim=True) / torch.norm(p, p=2)

        else:
            raise NotImplementedError
        
        if self.aggregation_method == "mean":
            sim_coding = torch.mean(similarity, dim=0, keepdim=True) # 1 x p

        elif self.aggregation_method in ["weightedsum_feat", "weightedsum_prototype"]:
            sim_coding = torch.mm(f.transpose(0,1), similarity) # feats_dim x p
            if self.aggregation_method == "weightedsum_prototype":
                sim_coding = torch.mean(sim_coding, dim=1, keepdim=True).t() # 1 x feats_dim
            else:
                sim_coding = torch.mean(sim_coding, dim=0, keepdim=True) # 1 x p if mean_dim == 0
        
        bag_logits = self.classifier(self.rho(sim_coding))
        
        Y_prob = F.softmax(bag_logits, dim = 1)
        Y_hat = torch.argmax(Y_prob, dim= 1)

        if self.instance_eval and "label" in kwargs:
            _, max_id = torch.max(probs[:, 1], 0)
            inst_logit = inst_logit[[max_id], :]
            instance_loss = self.instance_loss_fn(inst_logit, kwargs["label"])
            results_dict = {'instance_loss': instance_loss}
        else:
            results_dict = {"embedding": sim_coding}
        return bag_logits, Y_prob, Y_hat, sim_coding, results_dict

    @staticmethod
    def Euclidean_Similarity(tensor_a, tensor_b):
        device = tensor_a.device
        output = torch.zeros(tensor_a.size(0), tensor_b.size(0), device=device)
        for i in range(tensor_a.size(0)):
            output[i, :] = torch.pairwise_distance(tensor_a[i, :], tensor_b)
        return output