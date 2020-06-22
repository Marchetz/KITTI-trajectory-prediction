# KITTI Dataset for trajectory prediction

This repository contains Kitti dataset for trajectory prediction used in "MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction" published on [CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Marchetti_MANTRA_Memory_Augmented_Networks_for_Multiple_Trajectory_Prediction_CVPR_2020_paper.html) ([arxiv](https://arxiv.org/abs/2006.03340))

## KITTI dataset

To obtain samples, we collect 6 seconds chunks (2 seconds for past and 4 for future) in a sliding window fashion from all trajectories in the dataset, including the egovehicle. 

For training set, we use the following videos (KITTI enumeration): 5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91.
For test set, the rest videos: 1, 2, 15, 18, 29, 32, 52, 70.

We obtain 8613 top-view trajectories for training and 2907 for testing. Further details are in the paper (section 4.1).

## FILE DESCRIPTION 

The files _dataset_KITTI_train.json_ and _dataset_KITTI_test.json_ contain respectively train and test set.
The _dataset_pytorch.py_ file is a script to create a pytorch-style dataset with class and methods to visualize the single examples.

The _maps_ folder contains whole top-view context of a specific video. For each example, we crop an area relative (180x180 meters) to present position of the agent to predict.

## USAGE

To create a dataset and relative dataloader to train their own models:

```
        from torch.utils.data import DataLoader
        import dataset_pytorch
        import tqdm
        
        data_train   = dataset_pytorch.TrackDataset('dataset_kitti_test.json')
        train_loader = DataLoader(self.data_train, batch_size=32, num_workers=1, shuffle=True)
        data_test    = dataset_pytorch.TrackDataset('dataset_kitti_test.json')
        test_loader  = DataLoader(self.data_test, batch_size=32, num_workers=1, shuffle=False)
        
        for step, (index, past, future, scene_one_hot, _, _, _, _, _, _, _) in enumerate(tqdm.tqdm(train_loader)):
            #code to call own model
              

```

## AUTHORS AND CONTACTS

* **Francesco Marchetti** (MICC - Università degli Studi di Firenze)
* **Federico Becattini**  (MICC - Università degli Studi di Firenze)
* **Lorenzo Seidenari**   (MICC - Università degli Studi di Firenze)

For questions and explanations, you can contact by e-mail to francesco.marchetti@unifi.it

