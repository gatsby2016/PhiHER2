## >>>>>>>>>>>>>>>> 下述针对具体任务，通常设定好后不用改变 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
task=MMPrisk
datasource=TJMUCH70genesMP # 同时用于确定yaml文件
results_dir=/home/cyyan/Projects/HER2proj/models


## >>>>>>>>>>>>>>>> 下面这里需要根据实验情况每次进行修改 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
exp_code=data317_PhiTrans_tile5ksel0_selfqueryparam_uniform
model_type=ProtoTransformer # ['ProtoTransformer', 'CLAM', 'ProtoMIL', 'ABMIL', 'Transformer']
num_perslide=5000
top_num_inst=None
top_num_inst_twice=None


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
