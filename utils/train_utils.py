from ast import Lambda
import numpy as np
import os

#----> pytorch imports
import torch
import torch.optim as optim


from .optim_utils import Lamb
from models.model_ABMIL import ABMIL
from models.model_CLAM import CLAM_SB
from models.transformer import Transformer
from models.model_PMIL import ProtoMIL
from models.model_ProtoTrans import ProtoTransformer

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv


from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.core_utils import EarlyStopping, train_loop, validate, summary
from utils.loss_func import NLLSurvLoss, CrossEntropySurvLoss, MultiSurvLoss
from utils.utils import l1_reg_all


def _get_splits(datasets, cur):
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split, test_split = datasets
    # _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split, val_split, test_split


def _init_loss_function(loss_func=None, alpha_surv=0.5, beta_surv=0.5, device="cpu"):
    r"""
    Init the survival loss function
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    """
    print('\nInit loss function...', end=' ')
    if loss_func == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=alpha_surv)
    elif loss_func == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=alpha_surv)
    elif loss_func == 'multi_surv':
        loss_fn = MultiSurvLoss(alpha=alpha_surv, beta=beta_surv)
    elif loss_func == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError  
    loss_fn = loss_fn.to(device)  
    return loss_fn


def _init_optim(model, optim_func=None, lr=1e-4, reg=1e-5, scheduler_func=None, lr_adj_iteration=100):
    print('\nInit optimizer ...', end=' ')

    if optim_func == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_func == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg)
    elif optim_func == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=reg)
    elif optim_func == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=reg) 
    elif optim_func == "lamb":
        optimizer = Lamb(model.parameters(), lr=lr, weight_decay=reg)
    else:
        raise NotImplementedError

    if scheduler_func == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=lr_adj_iteration, eta_min=lr*0.1, verbose=True) #设置余弦退火算法调整学习率，每个epoch调整
    elif scheduler_func == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                base_lr=lr*0.25, max_lr=lr, step_size_up=lr_adj_iteration//6, 
                                                cycle_momentum=False, verbose=True) #
    elif scheduler_func == "LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, 
                                                start_factor=1, end_factor=0.1, total_iters=lr_adj_iteration//2, verbose=True)
    elif scheduler_func == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                  max_lr=lr, total_steps=lr_adj_iteration, pct_start=0.2, div_factor=10, final_div_factor=10, verbose=True)
    elif scheduler_func == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=int(lr_adj_iteration*0.75), gamma=0.3, verbose=True)
    elif scheduler_func is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=lr_adj_iteration, gamma=0.3, verbose=True) # no change for lr
    # 不同的scheduler策略可参考 https://zhuanlan.zhihu.com/p/538447997

    return optimizer, scheduler


def _init_model(model_type=None, model_size="ccl2048", input_size=2048, drop_out=0., n_classes=2, 
                top_num_inst=None, top_num_inst_twice=None, n_cluster=1, device="cpu"):
    print('\nInit Model...', end=' ')
    if model_type == "ABMIL":
        model_dict = {"size_arg":model_size, "dropout" : drop_out, "n_classes" : n_classes, "top_num_inst": top_num_inst, "device" : device}
        model = ABMIL(**model_dict)
    
    elif model_type == "Transformer":
        model = Transformer(num_classes=n_classes, input_dim=input_size, depth=1, 
                            heads=4, dim_head=64, hidden_dim=512, 
                            pool='cls', dropout=drop_out, emb_dropout=0., pos_enc=None)
    elif model_type == "CLAM":
        model = CLAM_SB(gate=True, size_arg=model_size, dropout=False, k_sample=8, instance_eval=True, n_classes=n_classes)
    elif model_type == "ProtoMIL":
        model = ProtoMIL(feature_size=input_size, hidden_size=512, cls_hidden_size=128,
                         num_cluster=n_cluster, topk_num=top_num_inst_twice, instance_eval=False,
                         dropout=drop_out, output_class=n_classes, similarity_method="Cosine",
                           aggregation_method="mean")
    elif model_type == "PhiHER2":
        model = ProtoTransformer(feature_size=input_size, embed_size=512, hidden_size=128, num_head=1,
                                 num_cluster=n_cluster, inst_num=top_num_inst, inst_num_twice=top_num_inst_twice, random_inst=False,
                                 attn_dropout=drop_out, dropout=drop_out, output_class=n_classes,
                                 cls_method="cls_keep_prototype_dim", abmil_branch=False, 
                                 init_query=False, query_is_parameter=False,
                                 only_similarity=True)
    else:
        raise ValueError('Unsupported model_type:', model_type)
    model = model.to(device)
    
    return model


def _init_loaders(args, train_split, val_split, test_split):

    print('\nInit Loaders...', end='\n')
    if train_split is not None:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split is not None:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None

    if test_split is not None:
        test_loader = _get_split_loader(args, test_split,  testing=False, batch_size=1)
    else:
        test_loader = None        

    return train_loader, val_loader, test_loader


def _init_writer(save_dir, cur, log_data=True):
    writer_dir = os.path.join(save_dir, str(cur))
    
    os.makedirs(writer_dir, exist_ok=True)

    if log_data:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    return writer


def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn"]:
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["survpath"]:

        data_WSI = data[0].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask

def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

    if modality == "coattn":  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  
        
    elif modality == "coattn_sa":

        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        out = model(**input_args)
        
    else:
        out = model(
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)
    return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, loss_fn):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
        
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]
        
        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        total_loss += loss_value 

        loss.backward()

        optimizer.step()

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():

        for data in loader:

            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality == "coattn":  
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                    )  
            elif modality == "survpath":

                input_args = {"x_path": data_WSI.to(device)}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                
                h = model(**input_args)
                
            else:
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]


            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _step(cur, args, loss_fn, model, optimizer, train_loader, val_loader, test_loader):
    all_survival = _extract_survival_metadata(train_loader, val_loader, test_loader)
    
    for epoch in range(args.max_epochs):
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, loss_fn)
    
    # save the trained model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
    
    print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
        val_cindex, 
        val_cindex_ipcw,
        val_IBS,
        val_iauc
        ))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)


def train_val(timeidx, cur, args, **kwargs):
    dataset_factory = kwargs["dataset_factory"]
    if "dataset_independent" in kwargs.keys():
        dataset_independent = kwargs["dataset_independent"]
        indep_loader = _get_split_loader(args, dataset_independent,  testing=False, batch_size=1)
    else:
        indep_loader = None

    print(f"Created train and val datasets for time {timeidx} fold {cur}")
    if args.split_dir.split("/")[-1][-12:] == "TrainValTest":
        csv_path = f'{args.split_dir}/splits_time{timeidx}.csv'
    else:
        csv_path = f'{args.split_dir}/splits_time{timeidx}_fold{cur}.csv'
        
    datasets = dataset_factory.return_splits(from_id=False, csv_path=csv_path)
    datasets[0].set_split_id(split_id=cur)
    datasets[1].set_split_id(split_id=cur)

    train_split, val_split, test_split = _get_splits(datasets, cur)
    train_loader, val_loader, test_loader = _init_loaders(args, train_split, val_split, test_split)

    if args.model_type in ["ProtoMIL", "PhiHER2"]:
        data = torch.load(os.path.join(args.cluster_path, f"time_{timeidx}_fold_{cur}_prototypes.pt"))
        prototype_feats = data["global_centroid_feats"].to(args.device)
        print(f"Loading, [GLOBAL] AP cluster for prototypes, Number of cluster: {len(prototype_feats)}")
    else:
        prototype_feats = []
    
    loss_fn = _init_loss_function(args.loss_func, args.alpha_surv, args.beta_surv, args.device)
    model = _init_model(args.model_type, args.model_size, args.encoding_dim, 
                        args.drop_out, args.n_classes, args.top_num_inst, args.top_num_inst_twice, n_cluster=len(prototype_feats), device=args.device)
    
    optimizer, scheduler = _init_optim(model, args.optim, args.lr, args.reg, args.scheduler, args.max_epochs)
    _print_network(args.results_dir, model)

    # results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, train_loader, val_loader, test_loader)
    writer = _init_writer(args.results_dir, cur, args.log_data)
    
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 50, stop_epoch=300, verbose = True)
    else:
        early_stopping = None
    
    print('Done!')
    
    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, scheduler, args.n_classes, writer, loss_fn, l1_reg_all, args.lambda_reg, args.gc,
                   prototype=prototype_feats)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, prototype=prototype_feats)
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    val_results_dict, val_error, val_auc, acc_logger = summary(model, val_loader, args.n_classes, prototype=prototype_feats)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    if test_loader is not None: # test_loader or dataset_independent
        test_results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, prototype=prototype_feats)
    elif indep_loader is not None:
        test_results_dict, test_error, test_auc, acc_logger = summary(model, indep_loader, args.n_classes, prototype=prototype_feats)
    
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        # if writer:
        #     writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    # else:
    #     test_results_dict, test_error, test_auc, acc_logger = None, 1, None, None

        
    writer.close()
    return val_results_dict, test_results_dict, val_auc, test_auc, 1-val_error, 1-test_error