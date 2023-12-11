#!/usr/bin/env bash

root_path_name=./training/
model_suffixes=/model.pkl
dataset_list=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange" "traffic" "electricity" "weather")
length=${#dataset_list[@]}
epoch=250
record=1
device=6

for random_seed in $(seq 2021 2025); 
do
    

for ((i=0; i<$length; i++))
do
  dset_pretrain=${dataset_list[$i]}
  # python pretraining on dset_pretrain
   python -u train.py \
    --gpu $device  \
    --epochs $epoch \
    --muti_dataset $dset_pretrain \
    --random_seed $random_seed

load_path=$root_path_name$dset_pretrain'_epochs_'$epoch'_seed_'$random_seed$model_suffixes
  for ((j=0; j<$length; j++))
  do
    dset_finetune=${dataset_list[$j]}
    echo n2one setting $dset_pretrain "->" $dset_finetune

    # python fine-tuning on dset_fintune

    python -u finetune.py \
      --target $dset_finetune \
      --model_path $load_path \
      --muti_dataset $dset_pretrain \
      --record $record \
      --random_seed $random_seed \
      --gpu $device
  done
done
done