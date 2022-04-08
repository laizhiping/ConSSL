### Contrastive Domain Adaptation: A Self-Supervised Learning Framework for sEMG-Based Gesture Recognition

This repository contains the source codes of our work: *Contrastive Domain Adaptation: A Self-Supervised Learning Framework for sEMG-Based Gesture Recognition*.

#### 1. Requirements
```
python==3.8.10
pandas==1.3.0
PyYAML==5.4.1
tensorboard==2.5.0
thop==0.0.31.post2005241907
torch==1.7.1
tqdm==4.61.2
```

##### 2. Datasets
The datasets should be organized by:
```
- dataset
  - capgmyo
    - dbb
      - subject-1
        - session-1
          - gesture-1
            - trial-1.mat
            - trial-2.mat
              ...
```
The dataset file *.mat* includes the sEMG channel data and the shape is *num_frames \* num_channels*.

##### 3. Pretrain
```
python3 run.py -cfg config/inter_session/capgmyo_dbb.yaml -t inter_session -sg pretrain [options]
python3 run.py -cfg config/inter_subject/capgmyo_dbb.yaml -t inter_subject -sg pretrain [options]
```
##### 4. Train
```
python3 run.py -cfg config/inter_session/capgmyo_dbb.yaml -t inter_session -sg train [options]
python3 run.py -cfg config/inter_subject/capgmyo_dbb.yaml -t inter_subject -sg train [options]
```

*-cfg/--config* indicates the path of configuration file, *-t/--task* indicates task (**inter_session/inter_subject**), *-sg/--stage* indicates stage (**pretrain/train**), *options* can be *-ss/-wz/-ws/-k*, and more details can be seen in *run.py*.
