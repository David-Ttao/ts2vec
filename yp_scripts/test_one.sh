#!/usr/bin/env bash
root_path_name=./training/
model_suffixes=/model.pkl
dataset_list=("etth1" )
finetune_list=("etth1" )
length=${#dataset_list[@]}
fine_length=${#finetune_list[@]}
epoch=150
record=1
device=2
random_seed=2021
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

  for ((j=0; j<$fine_length; j++))
  do
    dset_finetune=${finetune_list[$j]}
    echo n2one setting $dset_pretrain "->" $dset_finetune

    # python fine-tuning on dset_fintune

    python -u finetune.py \
          --target $dset_finetune \
          --model_path $load_path \
          --muti_dataset $dset_pretrain \
          --record $record \
          --gpu $device \
          --random_seed $random_seed \
          --model_name "TS2vec"
  done
done


