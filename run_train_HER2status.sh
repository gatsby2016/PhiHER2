#!/bin/bash

## >>>>>>>>>>>>>>>> 下述针对具体任务，通常设定好后不用改变 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
task=HER2status
datasource=Yale # 同时用于确定yaml文件
results_dir=/home/cyyan/Projects/HER2proj/models

list_num_perslide=(1000)

# 遍历数组中的每个整数
for top_num_inst_twice in "${list_num_perslide[@]}"; do
    echo $top_num_inst_twice

    ## >>>>>>>>>>>>>>>> 下面这里需要根据实验情况每次进行修改 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    exp_code=new_PhiHER2_tileAll_twice${top_num_inst_twice}_0abmil_0sel
    model_type=PhiHER2 # ['PhiHER2', 'CLAM', 'ProtoMIL', 'ABMIL', 'Transformer']

    top_num_inst=None
    # top_num_inst_twice=None
    num_perslide=None


    ## >>>>>>>>>>>>>>>> 下述根据上述参数设定，确定logpath，不用动 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    logpath=${results_dir}/${datasource}/${task}/${exp_code}
    if [ ! -d $logpath ]
            then mkdir -p $logpath
    fi


    ## >>>>>>>>>>>>>>>> 下述根据上述参数设定，进行training 通常不用动，如果有新加参数需要update <<<<<
    time python s4_ModelTraining.py \
        --config cfgs/${datasource}.yaml \
        --opts \
        exp_code $exp_code model_type $model_type \
        num_perslide $num_perslide \
        top_num_inst $top_num_inst \
        top_num_inst_twice $top_num_inst_twice  2>&1 | tee $logpath/train_logs.log
done