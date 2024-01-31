import torch
import os, argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score, roc_curve, precision_recall_curve

from utils.train_utils import _init_model
import matplotlib.pyplot as plt



# 指定特定exp name进行对比 比较summary结果
def print_spec_exp_comparison(comparsion1_name, comparsion2_name, respath=None):
    if comparsion1_name is not None:
        print("comparsion1_name")
        if os.path.exists(os.path.join(respath, comparsion1_name, "summary_alltimes_kfolds.csv")):
            print(pd.read_csv(os.path.join(respath, comparsion1_name, "summary_alltimes_kfolds.csv")))
        elif os.path.exists(os.path.join(respath, comparsion1_name, "summary.csv")):
            print(pd.read_csv(os.path.join(respath, comparsion1_name, "summary.csv")))

    if comparsion2_name is not None:
        print("comparsion2_name")
        if os.path.exists(os.path.join(respath, comparsion2_name, "summary_alltimes_kfolds.csv")):
            print(pd.read_csv(os.path.join(respath, comparsion2_name, "summary_alltimes_kfolds.csv")))
        elif os.path.exists(os.path.join(respath, comparsion2_name, "summary.csv")):
            print(pd.read_csv(os.path.join(respath, comparsion2_name, "summary.csv")))


"""
对文件夹内特定taskpattern的所有exps进行结果的summary print
"""
def print_exps_summary(respath=None, taskpattern=None):
    expnames = glob(os.path.join(respath, taskpattern))
    [print(epn) for epn in expnames]
    
    if taskpattern == "task_2*":
        header =  "val_cindex"
    elif taskpattern == "task_1*":
        header = "test_auc"
    else:
        header = "val_cindex"

    best_cindex = 0.0
    best_cindex_expname = "exp"
    for epname in expnames:
        if not os.path.exists(os.path.join(epname, "summary_alltimes_kfolds.csv")) and not os.path.exists(os.path.join(epname, "summary.csv")):             
            continue # 不存在多times的summary_alltimes_kfolds.csv文件和单times的summary.csv；说明exp还没跑完，continue
        elif os.path.exists(os.path.join(epname, "summary_alltimes_kfolds.csv")):
            print("Below is MULTI times Experiment:")
            resfilename = "summary_alltimes_kfolds.csv"

        elif os.path.exists(os.path.join(epname, "summary.csv")):
            print("Below is single time Experiment:")
            resfilename = "summary.csv"

        final_df = pd.read_csv(os.path.join(epname, resfilename))
        # print(final_df)
        print(f"Experiment: {epname}")
        print("{} mean val: {}".format(header, final_df[header].mean()))
        print("{} std val: {}".format(header, final_df[header].std()))

        if final_df[header].mean() > best_cindex:
            best_cindex = final_df[header].mean()
            best_cindex_expname = epname

    print(f"|@_@/@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\@_@| print_exps_summary finished.\n Best cindex expname: {best_cindex_expname}")
    return best_cindex_expname


def set_args():
    parser = argparse.ArgumentParser(description='Kfolds res evaluation.')
    parser.add_argument('-p', '--respath', type=str, 
                        default="/home/cyyan/Projects/HER2proj/models/TJMUCH/survival/", help='respath to be summarized')
    parser.add_argument('-n', '--comparsion_exp_name', type=str, 
                        default=None, # or as, "OS_ce_alpha0.1_adam1e3_dropout0.25_ccl_5foldnotest_StepLR"
                        help='Spec comparison exp name for comparison and evaluation.')
    parser.add_argument('-pa', '--taskpattern', type=str, 
                        default="*", # or as, "OS_ce*_adam*"
                        help='task pattern of summary exp names filtered for printing.') 
    parser.add_argument('-csvname', '--csv_filename_path', type=str, 
                        default="/home/cyyan/Projects/HER2proj/data/HEcasesFullInfo0831清洗后整理_去无用信息.csv",
                        help='csv_filename_path to be processed')     
    args = parser.parse_args()
    return args


# load data from csv path
def load_data(csv_path, label_mapping=None, filter_dict = None):
    data = pd.read_csv(csv_path)

    filter_mask = np.full(len(data), True, bool)
    # assert 'label' not in filter_dict.keys()
    for key, val in filter_dict.items():
        mask = data[key].isin(val)
        filter_mask = np.logical_and(filter_mask, mask)
    data = data[filter_mask]

    slide_ids, label = data["slide_id"], data["HER2status"]
    label = label.tolist()
    labels = [label_mapping[label[idx]] for idx in range(len(label))]

    return slide_ids, labels


def load_data_cvfoldtest(csv_path, filename="", label_mapping=None):
    data = pd.read_csv(os.path.join(csv_path, filename))
    
    slide_ids = data['test'].dropna().reset_index(drop=True)

    labels = [label_mapping[slide.split('_')[0].split('Her2')[-1]] for slide in slide_ids]
    return slide_ids, labels


def load_model(params_dict, num_cluster, trained_pt_loc=None, fold=0, device=None):
    model = _init_model(params_dict["model_type"], params_dict["model_size"], params_dict["encoding_dim"], 
                        params_dict["drop_out"], params_dict["n_classes"], params_dict["top_num_inst"], 
                        params_dict["top_num_inst_twice"], n_cluster=num_cluster, device=device)

    if trained_pt_loc is None:
        raise FileNotFoundError
    else:
        model.load_state_dict(torch.load(os.path.join(trained_pt_loc, "s_{}_checkpoint.pt".format(fold))))
        print("\nLoading model state dict from {}.\n ".format(os.path.join(trained_pt_loc, "s_{}_checkpoint.pt".format(fold))))
    
    return model


def read_params_file(filename):
    with open(filename, 'r') as file:
        js = file.read()
    dic = json.loads(js)   
    print(dic)
    return dic



if __name__ == "__main__":
    # params on trained dataset 
    # trained_dataset_name = "Yale" # or TCGA
    # task_name = "new_PhiHER2_tileAll_twice1000_0abmil_0sel"
    # task_name = "new_PMIL_Cosine_1insteval_tileAll_twice1000"
    # task_name = "new_PMIL_Euclidean_1insteval_tileAll_twice1000"
    # task_name = "CLAM_k8_tileAll"
    # task_name = "CLAM_k32_tileAll"
    # task_name = "ABMIL_tileAll"
    # task_name = "Transformer_tile5k"


    trained_dataset_name = "HEROHE" # or TCGA
    task_name = "new_PhiHER2_align_sim_tile5ktwice500"
    # task_name = "new_ProtoMIL_sim_Cosine_mean_tileAlltwice500_0insteval"
    # task_name = "new_ProtoMIL_sim_Euclidean_mean_tile5ktwice500_0insteval"
    # task_name = "new_CLAM_8"
    # task_name = "new_CLAM"
    # task_name = "new_ABMIL_tile5k_all"
    # task_name = "new_Transformer"
    

    common_model_dir = "/home/cyyan/Projects/HER2proj/models/"
    model_dir = os.path.join(common_model_dir, trained_dataset_name, "HER2status", task_name)
    print(">>>>>>>>>>>>>>>>>>>{}".format(model_dir))

    # params on dataset for indepedent evaluation
    indepedent_dataset = "TJMUCH70genes"     # TCGA
    common_result_dir = "/home/cyyan/Projects/HER2proj/results/"
    if indepedent_dataset in ["Yale"]:
        feats_path = "".join((common_result_dir, indepedent_dataset, "_2FeatsCCL"))
        csv_path = "/home/cyyan/Projects/HER2proj/data_ModPath_HER2_v3/Yale_HER2status.csv"
        if indepedent_dataset == trained_dataset_name:
            csv_path = "/home/cyyan/Projects/HER2proj/results/Yale_3CaseSplits/her2status_KFoldsCV"

    elif indepedent_dataset in ["HEROHE_test"]:
        feats_path = "".join((common_result_dir, indepedent_dataset, "_2FeatsCCL_40x"))
        csv_path = "/mnt/DATA/HEROHE_challenge/HEROHE_TestGTinfo.csv"
    elif indepedent_dataset in ["TCGA"]:
        # feats_path = "".join((common_result_dir, indepedent_dataset, "_2FeatsCCL_20x"))
        feats_path = "".join((common_result_dir, indepedent_dataset, "_2FeatsCCL"))
        csv_path = "/home/cyyan/Projects/HER2proj/data_TCGABRCA/TCGABRCA_AllSlides_ClinInfo_Status0927.csv"
    elif indepedent_dataset in ["TJMUCH70genes"]:
        feats_path = "".join((common_result_dir, indepedent_dataset, "_2FeatsCCL_40x"))
        csv_path = "/mnt/DATA/TJMUCH_data_total/70genes_clinicalinfo_full_1202_OK.csv"

    slide_id_list, slide_label_list = load_data(csv_path, label_mapping={"Negative": 0, "Positive": 1},
                                                filter_dict={"HER2status": ["Negative", "Positive"]})


    ## 读取保存的参数
    params_dict = read_params_file(os.path.join(model_dir, "params_setting.txt"))
    if torch.cuda.is_available() and params_dict["gpu"] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = params_dict["gpu"]
        device = torch.device('cuda:' + params_dict["gpu"])
    else:
        device = torch.device('cpu')

    alltimes_summary = []

    for tidx in range(params_dict["times"]):
        os.environ['PYTHONHASHSEED'] = str(params_dict["seed"])
        np.random.seed(params_dict["seed"])
        torch.manual_seed(params_dict["seed"])
        if device.type == 'cuda':
            torch.cuda.manual_seed(params_dict["seed"])
            torch.cuda.manual_seed_all(params_dict["seed"]) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        allfold_summary = []
        for kidx in range(params_dict["k"]): # each fold in each time

            # first load prototpype data for this `fold` this `time`
            # prototype_data = torch.load(os.path.join(model_dir, "time"+str(tidx), "fold" + str(kidx) + 'apcluster_global_prototypes.pt'))
            prototype_data = torch.load(os.path.join(params_dict["cluster_path"], f"time_{tidx}_fold_{kidx}_prototypes.pt"))

            global_cents_feats = prototype_data["global_centroid_feats"]
            
            print("Estimate number of cluster (GLOBAL): {} in fold{} for time {}".format(len(global_cents_feats), kidx, tidx))
            
            global_cents_feats = global_cents_feats.to(device)

            # 20231224 add for random prototypes evaluation
            # global_cents_feats = np.random.uniform(size=global_cents_feats.shape, low=global_cents_feats.min(), high=global_cents_feats.max())
            # global_cents_feats = torch.tensor(global_cents_feats,dtype=torch.float32).to(device)           
            # global_cents_feats = torch.randn(global_cents_feats.shape).to(device)

            # global_cents_feats = torch.rand_like(global_cents_feats).to(device)
            
            model = load_model(params_dict, len(global_cents_feats), 
                               trained_pt_loc= os.path.join(model_dir, "time"+str(tidx)),
                                fold=kidx, device=device)
            
            model.eval()

            patient_results = []
            all_probs = []
            all_preds = []
            all_labels_used = []
            all_cases_embedding = []

            if indepedent_dataset in ["Yale"] and indepedent_dataset == trained_dataset_name:
                slide_id_list, slide_label_list = load_data_cvfoldtest(csv_path, 
                                                                       filename=f"splits_time{tidx}_fold{kidx}.csv",
                                                                       label_mapping={"Neg": 0, "Pos": 1})
            
            with tqdm(total= len(slide_id_list)) as _tqdm: # 使用需要的参数对tqdm进行初始化
                for sidx, slide_id in enumerate(slide_id_list):
                    if indepedent_dataset in ["HEROHE_test"]:
                        slide_id = str(slide_id)+".mrxs"
                    elif indepedent_dataset in ["TJMUCH70genes"]:
                        slide_id = str(slide_id)+".svs"

                    _tqdm.set_postfix(slide_id="{}".format(slide_id))

                    if slide_id.split('.')[-1] in ["svs", "mrxs"]:
                        slide_id = '.'.join(slide_id.split('.')[:-1])
                    
                    full_path = os.path.join(feats_path, 'pt_files', '{}.pt'.format(slide_id))
                    
                    if not os.path.exists(full_path): # 存在个别样本经过s1 tissue seg后没有tissue区域，这个slide不会被用到
                        continue
                    features = torch.load(full_path) # load feats data
                    features = features[np.random.choice(len(features), params_dict["num_perslide"]), :] \
                        if params_dict["num_perslide"] is not None and params_dict["num_perslide"] < len(features) else features 

                    with torch.no_grad():
                        logits, Y_prob, Y_hat, _, res_dict = model(features.to(device).type(torch.float32), 
                                                            prototype=global_cents_feats, proj_proto=True)
                    embedding = res_dict['embedding'].cpu().numpy().squeeze()

                    probs = Y_prob.cpu().numpy().squeeze(0)
                    all_probs.append(probs)
                    all_preds.append(Y_hat.cpu().numpy()[0])
                    all_labels_used.append(slide_label_list[sidx])
                    all_cases_embedding.append(embedding)

                    patient_results.append({'slide_id': slide_id, 
                                            'prob_neg': probs[0], 'prob_pos': probs[1], 
                                            'pred': Y_hat.cpu().numpy()[0], 'label': slide_label_list[sidx]})
                    
                    _tqdm.update(1)
            
            metric_res_dict = {"time": tidx, "fold": kidx}
            eval_res = pd.concat((pd.DataFrame(patient_results), 
                                    pd.DataFrame(np.array(all_cases_embedding), columns=[f"sim{i}" for i in  list(range(len(all_cases_embedding[0])))])),
                                    axis=1)     
            eval_res.to_csv(os.path.join(model_dir, "time"+str(tidx), f"eval_{indepedent_dataset}_{kidx}_res.csv"))
            
            eval_res_dict = classification_report(all_labels_used, np.array(all_preds), 
                                                target_names=['neg', 'pos'], output_dict=True)
            for key, vals in eval_res_dict.items():
                if type(vals) is dict:
                    for sub_key, sub_vals in vals.items():
                        print(f"{sub_key}: {sub_vals}")
                        metric_res_dict[key+"_"+sub_key] = sub_vals

                else:
                    print(f"{key}: {vals}")
                    metric_res_dict[key] = vals
            
            balanced_acc = balanced_accuracy_score(all_labels_used, np.array(all_preds)) # balanced accuracy is defined as the average of recall obtained on each class.
            metric_res_dict.update({"balanced acc": balanced_acc})
            
            auc = roc_auc_score(all_labels_used, np.array(all_probs)[:, 1])
            metric_res_dict.update({"auc": auc})
            
            allfold_summary.append(metric_res_dict)

            print(">>>>>>>>>>>>>>>>>>>Estimate auc value: {} in fold{} for time {}".format(auc, kidx, tidx))
        
            # torch.save({'global_cents_feats': global_cents_feats,
            #             'projection_prototype': res_dict['projection_prototype'], 
            #             'query_prototype': res_dict['query_prototype']
            #             },
            #             os.path.join(model_dir, "time"+str(tidx), f"fold_{kidx}_prototypes_model_projection.pt"))

        allfold_summary = pd.DataFrame(allfold_summary)
        # if kidx > 0:
        #     allfold_summary.loc['mean'] = allfold_summary.apply(lambda x: x.mean())
        #     allfold_summary.loc['std'] = allfold_summary.apply(lambda x: x[:-1].std()) # 计算std时 新增的mean行不算在内
 
        alltimes_summary.append(allfold_summary)   

    alltimes_summary = pd.concat(alltimes_summary)
    alltimes_summary.loc['mean'] = alltimes_summary.apply(lambda x: x.mean())
    alltimes_summary.loc['std'] = alltimes_summary.apply(lambda x: x[:-1].std()) # 计算std时 新增的mean行不算在内
    
    # os.makedirs(os.path.join(common_model_dir, trained_dataset_name, "HER2status", "randomprototypes_eval_on_PhiHER2"), exist_ok=True)
    # alltimes_summary.to_csv(os.path.join(common_model_dir, trained_dataset_name, "HER2status", 
    #                                      "randomprototypes_eval_on_PhiHER2", "summary_metrics_alltimes_kfolds.csv"))
    alltimes_summary.to_csv(os.path.join(model_dir, f"summary_metrics_alltimes_kfolds_evaluated_{indepedent_dataset}.csv"))    
    print(f'>>>>>>>>>>>>>>>> {indepedent_dataset} DATASET evaluated on TASK {task_name}:\n {alltimes_summary}')