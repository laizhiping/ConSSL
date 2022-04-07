import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

target_dir = "..\\dataset\\cslhdemg"
source_dir = "cslhdemg"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# subject: 1-5, session: 1-5, gesture: 0-26
for subject in range(1, 2):
    for session in range(1, 6):
        for gesture in range(1, 27):
            path = os.path.join(source_dir, f"subject{subject}", f"session{session}", f"gest{gesture}.mat")
            mat = scipy.io.loadmat(path)
            emg = mat["gestures"]
            for trial in range(emg.shape[0]):
                 #deleting edge channels
                # print(emg[trial, 0].shape) # (192, 6144)
                trial_data = np.delete(emg[trial, 0], np.s_[7:192:8], 0)
                # print(trial_data.shape) # (168, 6144)

                plt.suptitle(f"subject {subject} session {session} gesture {gesture} trial {trial}")
                for x in range(0,168):
                    plt.subplot(7, 24, x+1)
                    plt.plot(trial_data[x,:])
                    plt.axis('off')
                    # plt.title(f'Exemplary plot of a single channel (channel {x})')
                plt.show()
                continue

                trial_data = trial_data.transpose()
                # print(trial_data.shape)  # (6144, 168)

                sub_dir = os.path.join(target_dir, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                new_path = os.path.join(sub_dir, f"trial-{trial+1}.mat")
                scipy.io.savemat(new_path, {"emg": trial_data}, do_compression=True)
                # print(new_path)
            print(f"Finish subject {subject}, session {session}, gesture {gesture}")
    break