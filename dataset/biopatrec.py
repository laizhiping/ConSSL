import scipy.io
import os
import numpy as np

target_dir = "..\\dataset\\biopatrec\\6mov8chFUS"
source_dir = "biopatrec\\6mov8chFUS"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# subject: 1-17
for subject in range(1, 18):
    path = os.path.join(source_dir, f"{subject}.mat")
    mat = scipy.io.loadmat(path)
    # print(mat.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'recSession'])
    print(mat["recSession"])


    # scipy.io.savemat(new_path, {"emg": emg}, do_compression=True)
    # print(new_path)
    # print(f"Finish subject {subject}, session {session}, gesture {gesture}, trial {trial}")
    break
