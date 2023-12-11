#!/usr/bin/env bash

root_path_name=./training/
model_suffixes=/model.pkl

dset_pretrain1=etth1_etth2_ettm1_ettm2_electricity_traffic_exchange_weather_pems08
dset_pretrain2=etth1_etth2_ettm1_ettm2_electricity_traffic_exchange_weather_solar
epoch=150
record=1
device=4
random_seed=2021
load_path=$root_path_name$dset_pretrain1'_epochs_'$epoch'_seed_'$random_seed$model_suffixes
# python pretraining on dset_pretrain

 python -u train.py \
  --gpu $device  \
  --epochs $epoch \
  --muti_dataset $dset_pretrain1 \
  --random_seed $random_seed 

  dset_finetune=solar
  echo n2one setting $dset_pretrain1 "->" $dset_finetune

  # python fine-tuning on dset_finetune
   python -u finetune.py \
    --target $dset_finetune \
    --model_path $load_path \
    --muti_dataset $dset_pretrain1 \
    --gpu $device  \
    --record $record \
    --random_seed $random_seed \
    --model_name "TS2vec"
  
load_path=$root_path_name$dset_pretrain2'_epochs_'$epoch'_seed_'$random_seed$model_suffixes
 python -u train.py \
  --gpu $device  \
  --epochs $epoch \
  --muti_dataset $dset_pretrain2 \
  --random_seed $random_seed 


  dset_finetune=pems08
  echo n2one setting $dset_pretrain2 "->" $dset_finetune

  # python fine-tuning on dset_finetune
   python -u finetune.py \
    --target $dset_finetune \
    --model_path $load_path \
    --muti_dataset $dset_pretrain2 \
    --gpu $device  \
    --record $record \
    --model_name "TS2vec" \
    --random_seed $random_seed 