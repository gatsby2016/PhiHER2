import os, argparse
import numpy as np

from datasets.dataset_generic import Generic_WSI_Classification_Dataset, split_slideinfo


"""
class: Case Splitting
"""
class CaseSplitting(object):
    def __init__(self, task_name = None, csv_info_path=None, split_to_dir=None, label_column_name="label", label_list=['0', '1'], filter_dict={},
                 seed=2020, shuffle=False, tvtmode=False, slide_featspath=None):
        f"""
        task_name:          task name; such as "recurrence" or "survival"
        csv_info_path:      clinical info csv file with slide_id column and label info
        split_to_dir:       dir for saving cases splitting files
        label_column_name:  (default: `label`) name of label column to be used.
        label_list:        (default: ['0', '1']) label lists for mapping class names in label column name to 0, 1, 2 ...
        seed:               (default: 2020) 'random seed 
        shuffle:            (default: False) shuffle slide data or not
        tvtmode             (default: False) default is cross validation mode, if True, train-val-test mode 
        """
        assert task_name is not None, "WHY not assign a TASK NAME??? tell me the task name please!"
        assert split_to_dir is not None, f"directory to save cases splits files {split_to_dir}"
        if tvtmode:
            split_to_dir = os.path.join(split_to_dir, task_name + "_TrainValTest")
        else:
            split_to_dir = os.path.join(split_to_dir, task_name + "_KFoldsCV")
        os.makedirs(split_to_dir, exist_ok=True)
        self.split_dir = split_to_dir

        assert csv_info_path is not None and os.path.isfile(csv_info_path), f"csv info path {csv_info_path} must be given or file do not EXIST :)"
        dataset = Generic_WSI_Classification_Dataset(csv_path = csv_info_path, 
                                                     label_col = label_column_name, label_dict = dict(zip(label_list, range(len(label_list)))),
                                                     filter_dict=filter_dict,
                                                     seed = seed, shuffle = shuffle, print_info = True, 
                                                     patient_strat= True,  # TRUE 表示按照patient进行划分split，保证一个patient的所有slide被split在同一区间内，如train或val；防止数据泄露 任何时候都应该这样
                                                     patient_voting='maj', # maj合理；max是对于一个patient多个slide标签选标签值最大，不合理
                                                     ignore=[],
                                                     slide_featspath=slide_featspath
                                                     )
        self.dataset = dataset


    def run(self, times=1, kfold=0, val_frac=0.2, test_frac=0.2):
        num_slides_cls = np.array([len(cls_ids) for cls_ids in self.dataset.patient_cls_ids])

        test_num = np.round(num_slides_cls * test_frac).astype(int)
        # train_num = num_slides_cls - test_num

        if kfold: # 在kfold不为0时，表示采用kfolds Cross validation mode mode，此时args.val_frac无效
            assert kfold > 1, "K folds num parameter should great than 1"
            for t in range(times):
                print(f"[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*] Time: {t} Cross Validation Folds {kfold}")
                self.mode_cv_kfolds(t, kfold, test_num)

        elif kfold == 0: # 在kfold为0时，表示采用train-val-test mode，此时args.val_frac有效
            val_num = np.round(num_slides_cls * val_frac).astype(int)
            # train_num = train_num - val_num # 确定train  val test num

            for t in range(times):
                print(f"[*(//@_@)*]@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@[*(//@_@)*] Time: {t} train-val-test")
                self.mode_train_val_test(t, val_num, test_num)
    

    def mode_cv_kfolds(self, times=1, kfold=10, test_num=0):
        self.dataset.create_splits_CrossValidation(k = kfold, test_num = test_num)

        for kval in range(kfold):
            print(f">>>>>>>>>>>>>>>>>>>>>>>> Splitting, Now Fold {kval}")
            self.dataset.set_splits()

            descriptor_df = self.dataset.split_trainval_cls_num_gen(return_descriptor=True)

            splits = self.dataset.return_splits(from_id=True)
            df_split_list = split_slideinfo(splits, ['train', 'val', 'test'])
            df_splist_bool = split_slideinfo(splits, ['train', 'val', 'test'], boolean_style=True)
            
            descriptor_df.to_csv(os.path.join(self.split_dir, f'splits_descriptor_time{times}_fold{kval}.csv'))
            df_split_list.to_csv(os.path.join(self.split_dir, f'splits_time{times}_fold{kval}.csv'))
            df_splist_bool.to_csv(os.path.join(self.split_dir, f'splits_bool_time{times}_fold{kval}.csv'))


    def mode_train_val_test(self, times=1, val_num=0, test_num=0):        
        self.dataset.create_splits(k = 1, val_num = val_num, test_num = test_num, label_frac=1)
        self.dataset.set_splits()
       
        descriptor_df = self.dataset.split_trainval_cls_num_gen(return_descriptor=True)

        splits = self.dataset.return_splits(from_id=True)
        df_split_list = split_slideinfo(splits, ['train', 'val', 'test'])
        df_splist_bool = split_slideinfo(splits, ['train', 'val', 'test'], boolean_style=True)

        descriptor_df.to_csv(os.path.join(self.split_dir, f'splits_descriptor_time{times}.csv'))
        df_split_list.to_csv(os.path.join(self.split_dir, f'splits_time{times}.csv'))
        df_splist_bool.to_csv(os.path.join(self.split_dir, f'splits_bool_time{times}.csv'))


def set_args():
    parser = argparse.ArgumentParser(description='Creating cases splits for whole slide classification')
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--csv_info_path', type=str, default=None, help='csv file with slide_id and label information')
    parser.add_argument('--split_to_dir', type=str, default=None)

    parser.add_argument('--times', type=int, default=1,
                        help='(default: 1) number of times, valid both in Cross validation mode and train-val-test mode')
    
    parser.add_argument('--kfold', type=int, default=10,
                        help='(default: 10) number of folds, common: Cross validation mode; if 0, means NOT CrossValidation, use train-val-test mode')
    parser.add_argument('--val_frac', type=float, default= 0.2,
                        help='(default: 0.2) fraction of samples for validation, only use in train-val-test mode, not used for cross validation')
    
    parser.add_argument('--test_frac', type=float, default= 0.2,
                        help='(default: 0.2) fraction of samples for test, use in both cross validation and train-val-test modes')
    
    parser.add_argument('--label_column_name', type=str, default="label",
                        help='(default: label) name of label column to be used.')
    parser.add_argument('--label_list', nargs='+', default=['0', '1'],
                        help=f'(default: [0, 1]) label lists for mapping class names in label column name to 0, 1, 2 ... IN Order.')
    parser.add_argument('--slide_featspath', type=str, default=None, help='slide feats path for useful slide selection')

    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed (default: 2020)')
    parser.add_argument('--shuffle', default=False, action='store_true')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = set_args()

    # args.task_name = "tmptest"
    # args.csv_info_path = "/home/cyyan/Projects/HER2proj/data/HEcasesFullInfo再整理0317.csv"
    # args.split_to_dir = "/home/cyyan/Projects/HER2proj/results/TJMUCH_3CaseSplits"
    # args.label_column_name = "复发"
    # args.test_frac = 0
    # args.times = 5
    # args.kfold = 5
    args.filter_dict = {"HER2status": ["Negative", "Positive"]}

    casesplit = CaseSplitting(args.task_name, args.csv_info_path, args.split_to_dir, args.label_column_name, args.label_list, args.filter_dict,
                              args.seed, args.shuffle, tvtmode=args.kfold==0, slide_featspath=args.slide_featspath)

    casesplit.run(args.times, args.kfold, args.val_frac, args.test_frac)