#----> general imports
import pandas as pd
import numpy as np
import os, json, argparse, copy

import utils.config_utils as cfg_utils
from utils.general_utils import set_seed_torch
from datasets.dataset_survpath import SurvivalDatasetFactory
from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from datasets.dataset_generic import Generic_MIL_Dataset
from utils.train_utils import train_val


def get_parser():
    parser = argparse.ArgumentParser(description='Configurations for MIL HER2 Prediction Training by PyTorch')
    parser.add_argument('--config', type=str, default=None, help='parameters config file') # 'scripts/HE2HER2/cfgs/Yale.yaml'
    parser.add_argument('--opts', help='see cfgs/defaults.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.config is not None, "Please provide config file for parameters."
    cfg = cfg_utils.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = cfg_utils.merge_cfg_from_list(cfg, args.opts)

    cfg.results_dir = os.path.join(cfg.results_dir, cfg.datasource, cfg.task, cfg.exp_code)
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    with open(os.path.join(cfg.results_dir, cfg.cfg_to_name), 'w') as f:
        json.dump(cfg, f, indent=2) # save the params setting

    print(f"[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*] CONFIG PARAMS:\n{cfg}")
    return cfg


"""
one time run for one fold training and validataion;
output: saving `split_latest_val_results.csv` and `split_latest_test_results.csv` in args.results_dir
return val_cindex and test_cindex
"""
def one_fold_run(tidx, kidx, dataset_factory, args):
    print(f"Created train and val datasets for time {tidx} fold {kidx}")
    if False:
        datasets = dataset_factory.return_splits(args, 
                                                csv_path='{}/splits_{}.csv'.format(args.split_dir, kidx), 
                                                fold=kidx)
    else:
        datasets = dataset_factory.return_splits(from_id=False, csv_path=f'{args.split_dir}/splits_time{tidx}_fold{kidx}.csv')
        datasets[0].set_split_id(split_id=kidx)
        datasets[1].set_split_id(split_id=kidx)
    
    # results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = train_val(datasets, i, args)
    # metrics_results.append([val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss])
    val_results_dict, test_results_dict, val_auc, test_auc, val_acc, test_acc = train_val(datasets, tidx, kidx, args)

    pd.DataFrame(val_results_dict).to_csv(os.path.join(args.results_dir, f"split_latest_val_{kidx}_results.csv"))
    if test_results_dict is not None:
        pd.DataFrame(test_results_dict).to_csv(os.path.join(args.results_dir, f"split_latest_test_{kidx}_results.csv"))

    return val_auc, test_auc, val_acc, test_acc


"""
one time run for k folds cross validation
output: saving `summary.csv` or `summary_partial_start_end.csv` in args.results_dir
return: final_df 
"""
def one_time_kfolds_run(tidx, dataset_factory, args):
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    metrics_results = []
    for kidx in folds:
        val_auc, test_auc, val_acc, test_acc = one_fold_run(tidx, kidx, dataset_factory, args)
        metrics_results.append([tidx, kidx, val_auc, test_auc, val_acc, test_acc])
                                
    if len(folds) != args.k:
        save_name = 'summary_partial_k{}_k{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'

    # final_df = pd.DataFrame(np.array(metrics_results), index=folds, 
    #                         columns=['val_cindex', 'val_cindex_ipcw', 'val_BS', 'val_IBS', 'val_iauc', 'val_loss'])    
    final_df = pd.DataFrame(np.array(metrics_results), columns=['time', 'fold', 'val_auc', 'test_auc', 'val_acc', 'test_acc'])   
    final_df.loc['mean'] = final_df.apply(lambda x: x.mean())
    final_df.loc['std'] = final_df.apply(lambda x: x[:-1].std()) # 计算std时 新增的mean行不算在内

    final_df.to_csv(os.path.join(args.results_dir, save_name))
    return final_df[:-2] # only return res without summary mean and std


"""
multi times runs for k folds cross validation
output: saving `summary_alltimes_kfolds.csv` or `summary_partial_tstart_tend.csv` in results_root_dir
"""
def multimes_kfolds_run(dataset_factory, args):
    tstart = 0 if args.t_start == -1 else args.t_start
    tend = args.times if args.t_end == -1 else args.t_end
    times = np.arange(tstart, tend)

    alltimes_summary = []
    results_root_dir = copy.deepcopy(args.results_dir)
    for tidx in times: # multi times loop
        args.results_dir = os.path.join(results_root_dir, "time"+str(tidx))
        os.makedirs(args.results_dir, exist_ok=True)

        onetime_metric_summary = one_time_kfolds_run(tidx, dataset_factory=dataset_factory, args=args)
        alltimes_summary.append(onetime_metric_summary)

    if len(times) != args.times:
        save_name = 'summary_partial_t{}_t{}.csv'.format(tstart, tend)
    else:
        save_name = 'summary_alltimes_kfolds.csv'

    alltimes_summary = pd.concat(alltimes_summary)
    alltimes_summary.loc['mean'] = alltimes_summary.apply(lambda x: x.mean())
    alltimes_summary.loc['std'] = alltimes_summary.apply(lambda x: x[:-1].std()) # 计算std时 新增的mean行不算在内

    alltimes_summary.to_csv(os.path.join(results_root_dir, save_name))


def main(args):
    if args.task == "HER2status":
        labels_dict = {'Negative':0, 'Positive':1}
        filter_dict = {"HER2status": ["Negative", "Positive"]}
        dataset_factory = Generic_MIL_Dataset(csv_path = args.csv_info_path,
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = labels_dict,
                                filter_dict = filter_dict,
                                label_col = args.label_col,
                                patient_voting='maj', # maj合理；max是对于一个patient多个slide标签选标签值最大，不合理
                                patient_strat= True, # TRUE 表示按照patient进行划分split，且保证同一个patient的所有slide被split在同一区间内，如train或val；否则直接按slideID进行split
                                ignore=[],
                                num_perslide=args.num_perslide
                                )
    else:
        dataset_factory = Generic_MIL_Survival_Dataset(csv_path = args.csv_info_path,
                                            apply_sig = False,
                                            data_dir= args.data_root_dir,
                                            mode = "path",
                                            shuffle = False, 
                                            seed = args.seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=args.n_classes,
                                            label_col = args.label_col,
                                            ignore=[],
                                            num_perslide=args.num_perslide)
        
    multimes_kfolds_run(dataset_factory=dataset_factory, args=args)


if __name__ == "__main__":
    args = get_parser()   
    args.device = set_seed_torch(args.gpu, args.seed)
    main(args)
    print("finished!")