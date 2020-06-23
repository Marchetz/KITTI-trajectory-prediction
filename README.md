# KITTI Dataset for trajectory prediction

This repository contains the Kitti dataset for trajectory prediction used in **"MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction"** published at [CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Marchetti_MANTRA_Memory_Augmented_Networks_for_Multiple_Trajectory_Prediction_CVPR_2020_paper.html) ([arxiv](https://arxiv.org/abs/2006.03340))

## KITTI dataset

To obtain samples, we collect 6 seconds chunks (2 seconds for past and 4 for future) in a sliding window fashion from all trajectories in the dataset, including the egovehicle. 

For the training set, we use the following videos (KITTI enumeration): 5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91.
For the test set, the remaining videos: 1, 2, 15, 18, 29, 32, 52, 70.

We obtain 8613 top-view trajectories for training and 2907 for testing. Further details are in the paper (section 4.1).

## FILE DESCRIPTION 

The files _dataset_KITTI_train.json_ and _dataset_KITTI_test.json_ contain respectively the train and test set.
The _dataset_pytorch.py_ file is a script to create a pytorch-style dataset with class and methods to visualize the single examples.

The _maps_ folder contains whole top-view context of a specific video. For each example, we crop an area relative (180x180 meters) to the present position of the agent to predict.

## USAGE

To create a dataset and relative dataloader to train your own models:

```
        from torch.utils.data import DataLoader
        import dataset_pytorch
        from tqdm import tqdm
        
        data_train   = dataset_pytorch.TrackDataset('dataset_kitti_test.json')
        train_loader = DataLoader(data_train, batch_size=32, num_workers=1, shuffle=True)
        data_test    = dataset_pytorch.TrackDataset('dataset_kitti_test.json')
        test_loader  = DataLoader(data_test, batch_size=32, num_workers=1, shuffle=False)
        
        for step, (index, past, future, scene_one_hot, video, class, num_vehicles, step, scene) in enumerate(tqdm(train_loader)):
            #code to call own model
              

```

*scene_one_hot* ([dim_batch,180,180,4] dimension) is used for training while *scene* ([dim_batch,180,180,1] dimension) is used for qualitative visualization.  

## AUTHORS AND CONTACTS

* **Francesco Marchetti** (MICC - Università degli Studi di Firenze)
* **Federico Becattini**  (MICC - Università degli Studi di Firenze)
* **Lorenzo Seidenari**   (MICC - Università degli Studi di Firenze)
* **Alberto Del Bimbo**   (MICC - Università degli Studi di Firenze)

For questions and explanations, you can contact by e-mail to francesco.marchetti@unifi.it

If you use this code, please cite the paper:

```
@inproceedings{marchetti2020mantra,
  title={Mantra: Memory augmented networks for multiple trajectory prediction},
  author={Marchetti, Francesco and Becattini, Federico and Seidenari, Lorenzo and Bimbo, Alberto Del},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7143--7152},
  year={2020}
}
```
