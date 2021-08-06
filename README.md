## Constrastive domain adaptation: A self-supervised Learning Framework for sEMG-Based gesture recognition


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
python run.py -cfg config/inter-session/capgmyo-dbb.yaml -sg pretrain -ne 30 -bs 16 [options]
```
##### 4. Train
```
python run.py -cfg config/inter-session/capgmyo-dbb.yaml -sg train -ne 100 -bs 16 [options]
```
##### 5. Test
```
python run.py -cfg config/inter-session/capgmyo-dbb.yaml -sg test
```

*-cfg/--config* indicates the path of configuration file, *-sg/--stage* indicates stage (**pretrain/train/test**), *options* can be *-s/-wz/-ws*, and more details can be seen in *run.py*.