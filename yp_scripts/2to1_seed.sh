#!/usr/bin/env bash
root_path_name=./training/
model_suffixes=/model.pkl
dataset_list=("etth1" "etth2" "ettm1" "ettm2" "electricity" "traffic" "weather" "exchange")
finetune_list=("electricity" "weather" "exchange")

epoch=250
record=1
device=2

for random_seed in $(seq 2021 2024); do
    for ((i=0; i<${#dataset_list[@]}; i++)); do
        for ((j=i+1; j<${#dataset_list[@]}; j++)); do
            target1=${dataset_list[i]}
            target2=${dataset_list[j]}
            dset_pretrain=${target1}_${target2}
            
            python -u train.py \
                --gpu $device  \
                --epochs $epoch \
                --muti_dataset $dset_pretrain \
                --random_seed $random_seed 
                
            for ((m=0; m<${#finetune_list[@]}; m++)); do
                dset_finetune=${finetune_list[m]}
                if [[ $dset_pretrain =~ $dset_finetune ]]
                then                       
                    continue
                fi
                load_path=$root_path_name$dset_pretrain'_epochs_'$epoch'_seed_'$random_seed$model_suffixes

                echo n2one setting $dset_pretrain "->" $dset_finetune
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
    done
done