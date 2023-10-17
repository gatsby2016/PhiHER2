# datapath="/mnt/DATA/TCGA/TCGA-BRCA/harmonized/Biospecimen/SVSimage/"
# patchpath="/home/cyyan/Projects/HER2proj/results/TCGA_1WsiPatching"
# featspath="/home/cyyan/Projects/HER2proj/results/TCGA_2FeatsCCL_blur"
# splitspath="/home/cyyan/Projects/HER2proj/results/TCGA_3CaseSplits"
# csvinfopath="/home/cyyan/Projects/HER2proj/data_TCGABRCA/TCGABRCA_AllSlides_ClinInfo_Status0927.csv"

datapath="/home/cyyan/Projects/HER2proj/data_ModPath_HER2_v3/pkg_v3/Yale_HER2_cohort/SVS"
patchpath="/home/cyyan/Projects/HER2proj/results/Yale_1WsiPatching"
featspath="/home/cyyan/Projects/HER2proj/results/Yale_2FeatsCCL_ctrans"
splitspath="/home/cyyan/Projects/HER2proj/results/Yale_3CaseSplits"
csvinfopath="/home/cyyan/Projects/HER2proj/data_ModPath_HER2_v3/Yale_HER2status.csv"

tocsvpath=$patchpath"/process_list_autogen.csv"
cclmodelpth="/home/cyyan/Projects/HER2proj/models/ctranspath.pth" # CCL_best_ckpt.pth
resize_size=224
# trainrespath="/home/cyyan/Projects/HER2proj/results/train"
labelname="HER2status"



if false; then
datapath="/home/cyyan/Projects/HER2proj/data/230928svs_biopsy/"

echo "Check data valid or not, running..."
python s1_WsiTiling.py \
	-s  $datapath \
	-d  "/home/cyyan/Projects/HER2proj/results/0928_check_SVS_patching" \
	--patch \
	--bgtissue \
	--stitch

python s2_FeatsExtracting.py \
	--feat_to_dir "/home/cyyan/Projects/HER2proj/results/0928_check_SVS_feats" \
	--csv_path "/home/cyyan/Projects/HER2proj/results/0928_check_SVS_patching/process_list_autogen.csv" \
	--h5_dir "/home/cyyan/Projects/HER2proj/results/0928_check_SVS_patching" \
	--slide_dir $datapath \
	--retccl_filepath $cclmodelpth \
	--slide_ext ".svs" \
	--batch_size 480 \
	--gaussian_blur \
	--float16
fi

if false; then
echo "WsiPatching..."
python s1_WsiTiling.py \
	-s  $datapath \
	-d  $patchpath \
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
    --resize_size $resize_size \
	--slide_dir $datapath \
	--slide_ext ".svs" \
	--batch_size 320 \
	--float16
fi
#--gaussian_blur \


if false; then
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
	--label_list "Negative" "Positive"\
    --slide_featspath $featspath\
	--seed 2020

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
	--label_list "Negative" "Positive"\
    --slide_featspath $featspath\
	--seed 2020
fi
