#!/usr/bin/env bash

root_path_name=./training/
model_suffixes=/model.pkl
dataset_list=("electricity" "traffic" "exchange_rate" "weather")
length=${#dataset_list[@]}

dset_pretrain=ETTh1_ETTh2_ETTm1_ETTm2
epoch=150
record=1
device=6

load_path=$root_path_name$dset_pretrain'_epochs_'$epoch$model_suffixes
# python pretraining on dset_pretrain

 python -u train.py \
  --gpu $device  \
  --epochs $epoch \
  --muti_dataset $dset_pretrain \
  --batch-size 128 \
  --max-train-length 201 \
  --max-threads 8

for ((j=0; j<$length; j++))
do
  dset_finetune=${dataset_list[$j]}
  echo n2one setting $dset_pretrain "->" $dset_finetune

  # python fine-tuning on dset_finetune
   python -u finetune.py \
    --target $dset_finetune \
    --model_path $load_path \
    --muti_dataset $dset_pretrain \
    --gpu $device  \
    --record $record \
    --batch-size 128 \
    --max-train-length 201 \
    --max-threads 8
  
done
