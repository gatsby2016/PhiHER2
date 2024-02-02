#############################plot histogram of the number of tiles on WSIs (dataset)
rm(list=ls())

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

library(readxl)
library(readr)
library(scales)


max_limit = 60000
breaksp = seq(0,1e6,1000)
herohe_feats_path = "/home/cyyan/Projects/HER2proj/results/HEROHE_train_1WsiPatching_40x"

herohe_patchdata = read_csv(paste(herohe_feats_path, "process_list_autogen.csv", sep = "/"))
herohe_patch_num = herohe_patchdata$stat_patch_num

hist(herohe_patch_num, labels = F, breaks = breaksp, xlim=c(0,max_limit),
     freq = F, col='#5698c3',border=F,
     main = "Histogram of the number of tiles on WSIs (HEROHE train/test)")# patch num
print("HEROHE")
median(herohe_patch_num)
mean(herohe_patch_num)
max(herohe_patch_num)

herohe_test_feats_path = "/home/cyyan/Projects/HER2proj/results/HEROHE_test_1WsiPatching_40x"

herohe_test_patchdata = read_csv(paste(herohe_test_feats_path, "process_list_autogen.csv", sep = "/"))
hherohe_test_patch_num = herohe_test_patchdata$stat_patch_num

hist(hherohe_test_patch_num, labels = F, breaks = breaksp, xlim=c(0,max_limit),
     freq = F, col=scales::alpha('#ec2d7a',.3),border=F,
     add=T)

lines(density(herohe_patch_num), lwd = 1.5, lty = 1) # 叠加密度曲线
lines(density(hherohe_test_patch_num), lwd = 1.5, lty = 2) # 叠加密度曲线

print("HEROHE test")
median(hherohe_test_patch_num)
mean(hherohe_test_patch_num)
max(hherohe_test_patch_num)

yale_feats_path = "/home/cyyan/Projects/HER2proj/results/Yale_1WsiPatching"
yale_patchdata = read_csv(paste(yale_feats_path, "process_list_autogen.csv", sep = "/"))
patch_num = yale_patchdata$stat_patch_num

hist(patch_num, labels = F, breaks = seq(0,1e6,250), xlim=c(0,10000),
     freq = F, col=scales::alpha('#d2357d',.5), border=F,
     add=F,
     main = "Histogram of the number of tiles on WSIs (Yale)")# patch num
lines(density(patch_num)) # 叠加密度曲线

print("Yale")
median(patch_num)
mean(patch_num)
max(patch_num)



#############################density plot for pred scores with features on WSIs
rm(list=ls())

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

library(readxl)
library(readr)
library(scales)
# install.packages("ggdensity")
# install.packages("ggplot2")
library("ggdensity")
library("ggplot2")
library(cowplot)
library(ggpubr)

root_path = "/home/cyyan/Projects/HER2proj/models/HEROHE/HER2status"
# exp_name = "new_PhiHER2_sim_tile5ktwice500_selfqueryparam_uniform"
# exp_name = "new_PhiHER2_align_sim_tile5k_1abmil0sel"
exp_name = "new_PhiHER2_align_sim_tile5ktwice500"

# for (i in 1:5){
#   timeidx = paste0("time", i)
#    = read_csv(paste(root_path, exp_name, timeidx, "eval_HEROHE_test_res.csv", sep = "/"))
#   eval_res = rbind(eval_res, single_time_res)
#   print(dim(eval_res))
# }

timeidx = "time0"
eval_res = read_csv(paste(root_path, exp_name, timeidx, "eval_HEROHE_test_res.csv", sep = "/"))

eval_res$label = as.factor(eval_res$label)

# for (idx in 0:25){
# ggscatterhist(eval_res, x = "prob_pos", y = paste0("sim", idx),
#               color = "#ffffff",
#               xlim = c(-0.5, 1.5),  ylim= c(-0.4, 0.4),
#               margin.params = list(fill = "lightgray"))
# }


cbPalette <- c("#56B4E9","#CC71AF", "#0072B2", "#000000", "#E69F00",  "#009E73", "#F0E442", "#D55E00")
base2 = ggplot(eval_res, aes(x = prob_pos, y = sim0)) +  
  scale_colour_manual(values=cbPalette) + xlim(-0.5, 1.5)+ ylim(-0.4, 0.4)+
  theme_bw() + theme(axis.line = element_line(color='black'),
        plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank())
# 
# base + geom_density_2d()

base2 + geom_density_2d(aes(colour = as.factor(label))) 
# base + geom_density_2d_filled() + facet_wrap(vars(as.factor(pred)))



################################### 画每个slide对应的similarity 直方图 有正负值

