import scipy.io
import os

source_dir = "../../source-dataset/capgmyo/dba"
target_dir = "capgmyo/dba"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# subject: 1-18, trial:1-10, gesture: 1-8 (trial 1, gesture 100-101)
for subject in range(1, 19):
    for gesture in range(1, 9):
        for trial in range(1, 11):
            path = os.path.join(source_dir, f"dba-preprocessed-{subject:03d}", f"{subject:03d}-{gesture:03d}-{trial:03d}.mat")
            mat = scipy.io.loadmat(path)
            # print(mat.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'trial', 'data', 'gesture', 'subject'])
            emg = mat["data"]
            # print(emg.shape) # (1000, 128)

            session = 1
            sub_dir = os.path.join(target_dir, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            new_path = os.path.join(sub_dir, f"trial-{trial}.mat")
            scipy.io.savemat(new_path, {"emg": emg}, do_compression=True)
            # print(new_path)
            print(f"Finish subject {subject}, session {session}, gesture {gesture}, trial {trial}")

    for gesture in range(100, 102):
        trial = 1
        path = os.path.join(source_dir, f"dba-preprocessed-{subject:03d}", f"{subject:03d}-{gesture:03d}-{trial:03d}.mat")
        mat = scipy.io.loadmat(path)
        # print(mat.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'trial', 'data', 'gesture', 'subject'])
        emg = mat["data"]
        # print(emg.shape) # (1000, 128)

        session = 1
        sub_dir = os.path.join(target_dir, f"subject-{subject}", f"session-{session}", f"gesture-{gesture}")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        new_path = os.path.join(sub_dir, f"trial-{trial}.mat")
        scipy.io.savemat(new_path, {"emg": emg}, do_compression=True)
        # print(new_path)
        print(f"Finish subject {subject}, session {session}, gesture {gesture}, trial {trial}")
