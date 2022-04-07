import scipy.io
import scipy.signal as scsig
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

target_dir = "..\\dataset\\ninapro\\db1"
source_dir = "ninapro\\db1"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# subect: 1-27, e: 1-3
for subject in range(1, 28):
    emg = []
    gestures = []
    trials = []
    for e in range(1, 4):
        path = os.path.join(source_dir, f"s{subject}", f"S{subject}_A1_E{e}.mat")
        mat = scipy.io.loadmat(path)
        # print(mat.keys())
        # dict_keys(['__header__', '__version__', '__globals__', 'subject', 'exercise', 'stimulus', 'emg', 'glove', 'restimulus', 'repetition', 'rerepetition'])
        # print(np.unique(mat["subject"])) # 1-27
        # print(np.unique(mat["exercise"])) # 1, 2, 3
        # print(mat["emg"].shape) # (100720, 10)
        # print(mat["glove"].shape) # (100720, 22)
        # print(np.unique(mat["restimulus"])) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12], [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17], [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
        # print(np.unique(mat["repetition"])) # [ 0  1  2  3  4  5  6  7  8  9 10]
        # print(np.unique(mat["rerepetition"])) # [ 0  1  2  3  4  5  6  7  8  9 10]
        # continue

        not0 = np.where(mat["restimulus"]!=0)[0]
        e_emg = mat["emg"][not0]
        e_gesture = mat["restimulus"][not0]
        e_rerepetition = mat["rerepetition"][not0]
        # print(np.unique(e_gesture), np.unique(e_rerepetition)) # [ 1  2  3  4  5  6  7  8  9 10 11 12] [ 1  2  3  4  5  6  7  8  9 10]
        # print(mat["emg"].shape, e_emg.shape, e_gesture.shape, e_rerepetition.shape) # (101014, 10) (37700, 10) (37700, 1) (37700, 1)

        emg.append(e_emg)
        if e == 2:
            e_gesture += 12
        elif e == 3:
            e_gesture += 12 + 17
        gestures.append(e_gesture)
        trials.append(e_rerepetition)



    emg = np.vstack(emg)
    gestures = np.vstack(gestures)
    trials = np.vstack(trials)
    # print(emg.shape, gestures.shape, trials.shape)
    # print(np.unique(gestures))

    # print(np.unique(gestures))
    # print(np.unique(trials))
    for gesture in np.unique(gestures):
        gesture_idx = np.where(gestures==gesture)[0]
        for trial in np.unique(trials):
            trial_idx = np.where(trials==trial)[0]
            idx = np.array(list(set(gesture_idx)&set(trial_idx)))

            emg_data = emg[idx]
            # print(emg_data.shape) # (305, 10)

            sub_dir = os.path.join(target_dir, f"subject-{subject}", f"session-1", f"gesture-{gesture}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            new_path = os.path.join(sub_dir, f"trial-{trial}.mat")
            scipy.io.savemat(new_path, {"emg": emg_data}, do_compression=True)

    break

