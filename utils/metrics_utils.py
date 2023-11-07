import os
import pandas as pd

import numpy as np
import json

from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# https://blog.csdn.net/weixin_43945848/article/details/122061718

"""
calcuate metrics according to the {csv_name} in {data_path}
"""
def calc_metrics(data_path, folds, times):
    print(f">>>>>>>>>>> cal metrics (precision / recall / f1 score ...) on {data_path}")
    target_names = ['neg', 'pos']

    all_times_metric_res_dict = []
    for tidx in range(times):

        for kidx in range(folds):
            metric_res_dict = {"time": tidx, "fold": kidx}

            csv_path = os.path.join(data_path, "time{}".format(tidx), f"split_latest_test_{kidx}_results.csv")
            res = pd.read_csv(csv_path)
            
            labels = np.array(res["label"])
            prob_pred = np.array(res.iloc[:, [2,3]])
            preds = np.argmax(prob_pred, 1)

            res_dict = classification_report(labels, preds, target_names=target_names, output_dict=True)
            
            for key, vals in res_dict.items():
                if type(vals) is dict:
                    for sub_key, sub_vals in vals.items():
                        metric_res_dict[key+"_"+sub_key] = sub_vals
                else:
                    metric_res_dict[key] = vals
            
            balanced_acc = balanced_accuracy_score(labels, preds) # balanced accuracy is defined as the average of recall obtained on each class.
            metric_res_dict.update({"balanced acc": balanced_acc})

            auc = roc_auc_score(labels, prob_pred[:, 1])
            metric_res_dict.update({"auc": auc})

            # auc_curve_data = roc_curve(labels, prob_pred[:, 1])    
            # RocCurveDisplay.from_predictions(labels, prob_pred[:, 1])

            # pr_curve_data = precision_recall_curve(labels, prob_pred[:, 1])
            # PrecisionRecallDisplay.from_predictions(labels, prob_pred[:, 1])

            all_times_metric_res_dict.append(metric_res_dict)
        # print(metric_res_dict, '\n')

    res_df = pd.DataFrame(all_times_metric_res_dict)
    res_df.loc['mean'] = res_df.apply(lambda x: x.mean())
    res_df.loc['std'] = res_df.apply(lambda x: x[:-1].std()) # 计算std时 新增的mean行不算在内

    res_df.to_csv(os.path.join(data_path, "summary_metrics_alltimes_kfolds.csv"))
    print(f">>>>>>>>>>> metrics on {data_path}\n {res_df}")

    return res_df


if __name__ == "__main__":

    data_path = "/home/cyyan/Projects/HER2proj/models/HEROHE/HER2status/ProtoTransformer40x_0mlp/"
    # data_path = "/home/cyyan/Projects/HER2proj/models/Yale/HER2status/ProtoTransformer_linearcls_embedd_abmilall_twice500split_share_h1_time5/"
    times = 5
    folds = 1

    calc_metrics(data_path, folds, times)