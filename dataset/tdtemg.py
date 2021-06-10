import scipy.io
import os
import numpy as np
from tdt import *

target_dir = "..\\dataset\\tdtemg"
source_dir = "tdtemg"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# subject: 1-18, session: 1, trial:1-10, gesture: 1-8 (trial 1, gesture 100-101)
for subject in range(1, 2):
    for session in range(1, 2):
            for gesture in range(1, 7):
                for trial in range(1, 11):
                    path = os.path.join(source_dir, f"{subject}-{session}-{gesture}-{trial}")
                    data = read_block(path)
                    # print(data.streams.EMG1.data.shape)
                    print(data.keys())
                    emg = data.streams.EMG1.data
                    # print(emg.shape) # (64, 1792)
                    emg1 = emg[0:0+8]
                    emg2 = emg[17:17+8]
                    emg3 = emg[31:31+8]
                    emg4 = emg[46:46+8]
                    emg = np.vstack((emg1, emg2, emg3, emg4)).transpose()
                    # print(emg.shape) # (1792, 32)


                    sub_dir = os.path.join(target_dir, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}")
                    if not os.path.exists(sub_dir):
                        os.makedirs(sub_dir)
                    new_path = os.path.join(sub_dir, f"trial-{trial}.mat")
                    scipy.io.savemat(new_path, {"emg": emg}, do_compression=True)
                    # print(new_path)
                    print(f"Finish subject {subject}, session {session}, gesture {gesture}, trial {trial}")

                print(f"Finish subject {subject}, session {session}, gesture {gesture}, trial {trial}")
    # break
