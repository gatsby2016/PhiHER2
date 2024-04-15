# TCGA wsi dataset
# datapath="/mnt/DATA/TCGA/TCGA-BRCA/harmonized/Biospecimen/SVSimage/"
# patchpath="/home/cyyan/Projects/HER2proj/results/TCGA_1WsiPatching_20x"
# featspath="/home/cyyan/Projects/HER2proj/results/TCGA_2FeatsCCL_20x"
# splitspath="/home/cyyan/Projects/HER2proj/results/TCGA_3CaseSplits"
# csvinfopath="/home/cyyan/Projects/HER2proj/data_TCGABRCA/TCGABRCA_AllSlides_ClinInfo_Status0927.csv"

# HEROHE biopsy dataset
# datapath="/mnt/DATA/HEROHE_challenge/TrainSet/"
# patchpath="/home/cyyan/Projects/HER2proj/results/HEROHE_train_1WsiPatching_40x"
# featspath="/home/cyyan/Projects/HER2proj/results/HEROHE_train_2FeatsCCL_40x_tumor"
# splitspath="/home/cyyan/Projects/HER2proj/results/HEROHE_3CaseSplits"
# csvinfopath="/mnt/DATA/HEROHE_challenge/HEROHE_TrainGTinfo.csv"

# Yale biopsy dataset
datapath="/home/cyyan/Projects/HER2proj/data_ModPath_HER2_v3/pkg_v3/Yale_HER2_cohort/SVS"
patchpath="/home/cyyan/Projects/HER2proj/results/Yale_1WsiPatching"
featspath="/home/cyyan/Projects/HER2proj/results/Yale_2FeatsCCL"
splitspath="/home/cyyan/Projects/HER2proj/results/Yale_3CaseSplits"
csvinfopath="/home/cyyan/Projects/HER2proj/data_ModPath_HER2_v3/Yale_HER2status.csv"
labelname="HER2status"


tocsvpath=$patchpath"/process_list_autogen.csv"
cclmodelpth="/home/cyyan/Projects/HER2proj/models/CCL_best_ckpt.pth" # CCL_best_ckpt.pth ctranspath.pth


if true; then
tile_size=256
overlap_size=256
echo "WsiPatching..."
python s1_WsiTiling.py \
    -s  $datapath \
    -d  $patchpath \
    -ps $tile_size \
    -ss $overlap_size \
    --patch \
    --bgtissue \
    --stitch
fi

if true; then
echo "FeatsExtraction..."
python s2_FeatsExtracting.py \
    --feat_to_dir $featspath \
    --csv_path $tocsvpath \
    --h5_dir $patchpath \
    --retccl_filepath $cclmodelpth \
    --slide_dir $datapath \
    --slide_ext ".svs" \
    --batch_size 320 \
    --float16
fi
# --auto_skip \
#--gaussian_blur \


if true; then
echo "Cross Validation splitting ..."
echo "N times K folds cross validation mode split."
python s3_CaseSplitting.py \
    --task_name "her2status" \
    --csv_info_path $csvinfopath \
    --split_to_dir $splitspath \
    --times 5 \
    --kfold 5 \
    --val_frac 0 \
    --test_frac 0.2 \
    --label_column_name $labelname \
    --label_list "Negative" "Positive" \
    --slide_featspath $featspath\
    --seed 2020
fi

if true; then
echo "N times train-val-test mode split."
python s3_CaseSplitting.py \
    --task_name "her2status" \
    --csv_info_path $csvinfopath \
    --split_to_dir $splitspath \
    --times 5 \
    --kfold 0 \
    --val_frac 0.1 \
    --test_frac 0.2 \
    --label_column_name $labelname \
    --label_list "Negative" "Positive" \
    --slide_featspath $featspath\
    --seed 2020
fi


if true; then
echo "HP cluster for prototypes."
python s4_HPcluster_prototypes.py --config_file "cfg/Yale.yaml"
fi