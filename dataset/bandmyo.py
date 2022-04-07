import scipy.io
import os
import numpy as np

target_dir = "..\\dataset\\bandmyo"
source_dir = "BandMyo-Dataset\\data"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# subject: 0-5, session: 1, gesture: 1-15, trial:0-7,
for subject in range(0, 6):
    for gesture in range(1, 16):
        for trial in range(0, 8):
            path = os.path.join(source_dir, f"{subject:03d}", f"{gesture:03d}", f"{subject:03d}-{gesture:03d}-{trial:03d}.mat")
            mat = scipy.io.loadmat(path)
            # print(mat.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'data', 'gesture', 'subject', 'trail'])
            emg = mat["data"]
            # print(emg.shape) # (T, 8) 600 < T < 1600

            sub_dir = os.path.join(target_dir, f"subject-{subject+1}", "session-1", f"gesture-{gesture}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            new_path = os.path.join(sub_dir, f"trial-{trial+1}.mat")
            scipy.io.savemat(new_path, {"emg": emg}, do_compression=True)
            # print(new_path)
            print(f"Finish subject {subject}, gesture {gesture}, trial {trial}")

    # break
