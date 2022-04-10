from numpy.lib.function_base import append
import torch
import os
import scipy.io
import numpy as np
from . import preprocessing
from . import data_augmentation as da

class DataReader(torch.utils.data.Dataset):
    def __init__(self, subjects, sessions, gestures, trials, window_size, window_step, dataset_name, dataset_path):
        super(DataReader, self).__init__()
        self.subjects = subjects
        self.sessions = sessions
        self.gestures = gestures
        self.trials = trials
        self.window_size = window_size
        self.window_step = window_step
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        self.load_dataset()
        self.preprocess()
        # self.augment()
        self.generate()
        self.shuffle()

        # self.X: --list, list of all trials, self.X[i]: [frame, channel]
        # self.y: --list, list of gestures of all trials
        # self.x_offsets: --list of all windows, self.x_offset[i]: tuple(index_in_self.X, window_start)
        # self.indexes: --list, indexes of self.x_offset

    def __getitem__(self, i):
        idx = self.indexes[i]
        trial_and_window = self.x_offsets[idx]

        trial_index = trial_and_window[0]
        window_start = trial_and_window[1]
        window_size = self.window_size
        train_data = self.X[trial_index]
        if window_size != 0:
            x = train_data[window_start:window_start + window_size, :]  # (frame, channel)
        else:
            if train_data.shape[0] < self.max_len:
                zero_left = (self.max_len-train_data.shape[0]) // 2
                zero_right = self.max_len-train_data.shape[0] - zero_left
                x = np.pad(train_data, ((zero_left, zero_right), (0, 0)), "constant")
            else:
                x = train_data

        if self.dataset_name == "armband":
            for channel in range(x.shape[1]):
                preprocessing.butter_highpass_filter(x[:,channel], 2, 200)

        y = self.y[trial_index]
        x = np.expand_dims(x, axis=0) # (1, frame, channel)

        # if self.dataset_name in ["capgmyo-dbb", "capgmyo-dbc", "bandmyo", "ninapro", "armband"]:
        y = y - 1
        return x, y

    def __len__(self):
        return len(self.indexes)

    def shuffle(self):
        np.random.shuffle(self.indexes)

    def load_dataset(self):
        X, y = [], []
        max_len, min_len = 0, 1e10
        root = self.dataset_path

        for subject in self.subjects:
            # if self.args.dataset_name == "armband":
            #     self.final_shifting = [0]*len(self.sessions)
            for session in self.sessions:
                # if self.args.dataset_name == "armband":
                #     session_X, session_y = [], []
                for gesture in self.gestures:
                    for trial in self.trials:
                        path = os.path.join(root, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}", f"trial-{trial}")
                        mat = scipy.io.loadmat(path)
                        X.append(mat["emg"]) # (1000, 128), (52, 8)
                        y.append(gesture)
                        # session_X.append(mat["emg"].transpose()) # (8, 52)
                        # session_y.append(gesture)
                        max_len = mat["emg"].shape[0] if mat["emg"].shape[0] > max_len else max_len
                        min_len = mat["emg"].shape[0] if mat["emg"].shape[0] < min_len else min_len
                # if self.args.dataset_name == "armband":
                #     self.final_shifting[session] = preprocessing.get_final_shifting()

        self.X = X
        self.y = y
        self.max_len = max_len
        self.min_len = min_len
        # print(f"Load [{self.args.dataset_name}] successfully, for all trials, max_len = {self.max_len}, min_len = {self.min_len}")

    def preprocess(self):
        pass
        
    def generate(self):
        self.make_segments()
        self.indexes = np.arange(len(self.x_offsets))

    def make_segments(self):
        window_size = self.window_size
        window_step = self.window_step
        x_offsets = []
        if window_size != 0:
            for i in range(len(self.X)):
                trial_data = self.X[i]  # (frame, channel)
                frame = trial_data.shape[0]
                for j in range(0, frame - window_size, window_step):
                    x_offsets.append((i, j))
        else:
            x_offsets = [(i, 0) for i in range(len(self.X))]

        self.x_offsets = x_offsets

    def augment(self):
        X_aug, y_aug = [], []

        self.size_factor = 0
        self.time_warping = 0.2
        self.mag_warping = 0.2
        self.noise_snr_db = 25
        self.permutation = 0
        self.rotation = 0
        self.rotation_mask=None
        self.scale_sigma = 0

        for i in range(len(self.X)):
            for _ in range(self.size_factor):
                x = np.copy(self.X[i])
                if self.permutation != 0:
                    x = da.permute(x, nPerm=self.permutation)
                if self.rotation != 0:
                    x = da.rotate(x, rotation=self.rotation, mask=self.rotation_mask)
                if self.time_warping != 0:
                    x = da.time_warp(x, sigma=self.time_warping)
                if self.scale_sigma != 0:
                    x = da.scale(x, sigma=self.scale_sigma)
                if self.mag_warping != 0:
                    x = da.mag_warp(x, sigma=self.mag_warping)
                if self.noise_snr_db != 0:
                    x = da.jitter(x, snr_db=self.noise_snr_db)

                if self.permutation or self.rotation or self.time_warping or self.scale_sigma or self.mag_warping or self.noise_snr_db:
                    X_aug.append(x)
                    y_aug.append(self.y[i])

            X_aug.append(self.X[i])
            y_aug.append(self.y[i])

        self.X = X_aug
        self.y = y_aug


class PairReader(torch.utils.data.Dataset):
    def __init__(self, subjects, sessions, gestures, trials, window_size, window_step, dataset_name, dataset_path):
        super(PairReader, self).__init__()
        self.subjects = subjects
        self.sessions = sessions
        self.gestures = gestures
        self.trials = trials
        self.window_size = window_size
        self.window_step = window_step
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        self.load_dataset()
        self.preprocess()
        # self.augment()
        self.generate()
        self.shuffle()

        # self.X: --list, list of all trials, self.X[i]: [frame, channel]
        # self.y: --list, list of gestures of all trials
        # self.x_offsets: --list of all windows, self.x_offset[i]: tuple(index_in_self.X, window_start)
        # self.indexes: --list, indexes of self.x_offset

    def __getitem__(self, i):
        idx = self.indexes[i]
        trial_and_window = self.x_offsets[idx]

        trial_index = trial_and_window[0]
        window_start1 = trial_and_window[1]
        window_start2 = trial_and_window[2]
        window_size = self.window_size
        train_data = self.X[trial_index]
        if window_size != 0:
            x1 = train_data[window_start1:window_start1 + window_size, :]  # (frame, channel)
            x2 = train_data[window_start2:window_start2 + window_size, :]
        else:
            if train_data.shape[0] < self.max_len:
                zero_left = (self.max_len-train_data.shape[0]) // 2
                zero_right = self.max_len-train_data.shape[0] - zero_left
                x = np.pad(train_data, ((zero_left, zero_right), (0, 0)), "constant")
            else:
                x = train_data

        if self.dataset_name == "armband":
            for channel in range(x1.shape[1]):
                preprocessing.butter_highpass_filter(x1[:,channel], 2, 200)
                preprocessing.butter_highpass_filter(x2[:,channel], 2, 200)

        y = self.y[trial_index]
        x1 = np.expand_dims(x1, axis=0) # (1, frame, channel)
        x2 = np.expand_dims(x2, axis=0)

        # if self.dataset_name in ["capgmyo-dbb", "capgmyo-dbc", "bandmyo", "ninapro", "armband"]:
        y = y - 1
        return x1, x2, y

    def __len__(self):
        return len(self.indexes)

    def shuffle(self):
        np.random.shuffle(self.indexes)

    def load_dataset(self):
        X, y = [], []
        max_len, min_len = 0, 1e10
        root = self.dataset_path

        for subject in self.subjects:
            for session in self.sessions:
                for gesture in self.gestures:
                    for trial in self.trials:
                        path = os.path.join(root, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}", f"trial-{trial}")
                        mat = scipy.io.loadmat(path)
                        X.append(mat["emg"])
                        y.append(gesture)
                        max_len = mat["emg"].shape[0] if mat["emg"].shape[0] > max_len else max_len
                        min_len = mat["emg"].shape[0] if mat["emg"].shape[0] < min_len else min_len
        self.X = X
        self.y = y
        self.max_len = max_len
        self.min_len = min_len
        # print(f"Load [{self.args.dataset_name}] successfully, for all trials, max_len = {self.max_len}, min_len = {self.min_len}")

    def preprocess(self):
        pass
        
    def generate(self):
        self.make_segments()
        self.indexes = np.arange(len(self.x_offsets))

    def make_segments(self):
        window_size = self.window_size
        window_step = self.window_step
        x_offsets = []
        if window_size != 0:
            for i in range(len(self.X)):
                trial_data = self.X[i]  # (frame, channel)
                frame = trial_data.shape[0]
                for j in range(0, frame - window_size, window_step):
                    x_offsets.append((i, j))
        else:
            x_offsets = [(i, 0) for i in range(len(self.X))]

        self.x_offsets = []
        end = len(x_offsets)-1 if len(x_offsets)%2 else len(x_offsets)
        for i in range(0, end-1, 1): # change step from 2->1
            example1, example2 = x_offsets[i], x_offsets[i+1]
            if example1[0] != example2[0]: # not same trial
                i = i+1
            else: # (trail, window1, window2)
                self.x_offsets.append((example1[0], example1[1], example2[1]))
        
        # print(len(x_offsets), len(self.x_offsets))


    def augment(self):
        X_aug, y_aug = [], []

        self.size_factor = 0
        self.time_warping = 0.2
        self.mag_warping = 0.2
        self.noise_snr_db = 25
        self.permutation = 0
        self.rotation = 0
        self.rotation_mask=None
        self.scale_sigma = 0

        for i in range(len(self.X)):
            for _ in range(self.size_factor):
                x = np.copy(self.X[i])
                if self.permutation != 0:
                    x = da.permute(x, nPerm=self.permutation)
                if self.rotation != 0:
                    x = da.rotate(x, rotation=self.rotation, mask=self.rotation_mask)
                if self.time_warping != 0:
                    x = da.time_warp(x, sigma=self.time_warping)
                if self.scale_sigma != 0:
                    x = da.scale(x, sigma=self.scale_sigma)
                if self.mag_warping != 0:
                    x = da.mag_warp(x, sigma=self.mag_warping)
                if self.noise_snr_db != 0:
                    x = da.jitter(x, snr_db=self.noise_snr_db)

                if self.permutation or self.rotation or self.time_warping or self.scale_sigma or self.mag_warping or self.noise_snr_db:
                    X_aug.append(x)
                    y_aug.append(self.y[i])

            X_aug.append(self.X[i])
            y_aug.append(self.y[i])

        self.X = X_aug
        self.y = y_aug
