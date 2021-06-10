#!/bin/bash

i=1
while [ $i -le 500 ]
do
    python run.py -cfg config/bandmyo.yaml
    i=$[$i+1]
done