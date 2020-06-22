# KITTI Dataset for trajectory prediction

This repository contains Kitti dataset for trajectory prediction used in "MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction" published on [CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Marchetti_MANTRA_Memory_Augmented_Networks_for_Multiple_Trajectory_Prediction_CVPR_2020_paper.html) ([arxiv](https://arxiv.org/abs/2006.03340))

## KITTI dataset

To obtain samples, we collect 6 seconds chunks (2 seconds for past and 4 for future) in a sliding window fashion from all trajectories in the dataset, including the egovehicle. For training set, we use the following videos: 5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91. For test set, the rest videos: 1, 2, 15, 18, 29, 32, 52, 70.
We obtain 8613 top-view trajectories for training and 2907 for testing. Further details are in the paper (section 4.1).
