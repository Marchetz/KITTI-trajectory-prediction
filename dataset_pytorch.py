import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import cv2
import json

# context colormap
colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.29, 0.57, 0.25)]
cmap_name = 'scene_list'
cm = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=4)


class TrackDataset(data.Dataset):
    """
    Dataset class for KITTI.

    The building class is merged into the background class
    0:background, 1:street, 2:sidewalk, 3: vegetation
    """
    def __init__(self, json_dataset):

        tracks = json.load(open(json_dataset))

        self.index = []
        self.pasts = []             # [len_past, 2]
        self.futures = []           # [len_future, 2]
        self.positions_in_map = []  # position in complete scene
        self.rotation_angles = []   # trajectory angle in complete scene
        self.scenes = []            # [360, 360, 1]
        self.videos = []            # '0001'
        self.classes = []           # 'Car'
        self.num_vehicles = []      # 0 is ego-vehicle, >0 other agents
        self.step_sequences = []

        # Preload data
        for t in tracks.keys():

            past = np.asarray(tracks[t]['past'])
            future = np.asarray(tracks[t]['future'])
            position_in_map = np.asarray(tracks[t]['position_in_map'])
            rotation_angle = tracks[t]['angle_rotation']
            video = tracks[t]['video']
            class_vehicle = tracks[t]['class']
            num_vehicle = tracks[t]['num_vehicle']
            step_sequence = tracks[t]['step_sequence']

            # extract context information from entire map of video
            path_scene = 'maps/2011_09_26__2011_09_26_drive_' + video + '_sync_map.png'
            scene_track = cv2.imread(path_scene, 0)

            scene_track[np.where(scene_track == 3)] = 0
            scene_track[np.where(scene_track == 4)] -= 1
            scene_track = scene_track[
                                      int(position_in_map[1]) * 2 - 180:int(position_in_map[1]) * 2 + 180,
                                      int(position_in_map[0]) * 2 - 180:int(position_in_map[0]) * 2 + 180]

            matRot_scene = cv2.getRotationMatrix2D((180, 180), rotation_angle, 1)
            scene_track = cv2.warpAffine(scene_track, matRot_scene,
                                         (scene_track.shape[0], scene_track.shape[1]),
                                         borderValue=0,
                                         flags=cv2.INTER_NEAREST)

            self.index.append(t)
            self.pasts.append(past)
            self.futures.append(future)
            self.positions_in_map.append(position_in_map)
            self.rotation_angles.append(rotation_angle)
            self.videos.append(video)
            self.classes.append(class_vehicle)
            self.num_vehicles.append(num_vehicle)
            self.step_sequences.append(step_sequence)
            self.scenes.append(scene_track)

        self.pasts = torch.FloatTensor(self.pasts)
        self.futures = torch.FloatTensor(self.futures)
        self.positions_in_map = torch.FloatTensor(self.positions_in_map)


    def show_track(self, index):
        """
        Show past and future trajectory of an example.
        :param index: example index in dataset
        :return: None
        """

        past = self.pasts[index]
        future = self.futures[index]
        plt.plot(past[:, 0], past[:, 1], c='blue')
        plt.plot(future[:, 0], future[:, 1], c='green')
        plt.axis('equal')
        plt.show()
        plt.close()

    def show_track_in_scene(self, index):
        """
        Show past and future trajectory of an example localized in context.
        :param index: example index in dataset
        :return: None
        """
        past = self.pasts[index]
        future = self.futures[index]
        scene = self.scenes[index]
        plt.imshow(scene, cmap=cm, origin='lower')
        plt.plot(past[:, 0] * 2 + 180, past[:, 1] * 2 + 180, c='blue')
        plt.plot(future[:, 0] * 2 + 180, future[:, 1] * 2 + 180, c='green')
        plt.show()
        plt.close()

    def save_example(self, past, future, scene, video, vehicle, number, step, path):
        """
        Plot past and future trajectory in the context and save it to 'path' folder.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param scene: the observed scene where is the trajectory
        :param video: video index of the trajectory
        :param vehicle: vehicle type of the trajectory
        :param number: number of the vehicle
        :param step: step of example in the vehicle sequence
        :param path: saving folder of example
        :return: None
        """

        plt.imshow(scene, cmap=cm, origin='lower')
        plt.plot(past[:, 0] * 2 + 180, past[:, 1] * 2 + 180, c='blue')
        plt.plot(future[:, 0] * 2 + 180, future[:, 1] * 2 + 180, c='green')
        plt.title('video: %s vehicle: %s_%s step: %s' % (video, vehicle, number, step))
        plt.savefig(path + str(step) + '.png')
        plt.close()

    def save_dataset(self, folder):
        """
        Save plots of entire dataset divided by videos and vehicles.
        :param folder: saving folder of all dataset
        :return: None
        """

        folder = folder + '/'
        for i in range(self.pasts.shape[0]):
            past = self.pasts[i]
            future = self.futures[i]
            scene = self.scenes[i]
            video = self.videos[i]
            vehicle = self.classes[i]
            number = self.num_vehicles[i]
            step = self.step_sequences[i]

            if not os.path.exists(folder + video):
                os.makedirs(folder + video)
            video_path = folder + video + '/'
            if not os.path.exists(video_path + vehicle + number):
                os.makedirs(video_path + vehicle + number)
            vehicle_path = video_path + '/' + vehicle + number + '/'
            self.save_example(past, future, scene, video, vehicle, number, step, vehicle_path)

    def __len__(self):
        return len(self.pasts)

    def __getitem__(self, idx):
        return self.index[idx], self.pasts[idx], self.futures[idx], np.eye(4, dtype=np.float32)[self.scenes[idx]], \
               self.positions_in_map[idx], self.rotation_angles[idx], self.videos[idx], \
               self.classes[idx], self.num_vehicles[idx], self.step_sequences[idx], self.scenes[idx],


