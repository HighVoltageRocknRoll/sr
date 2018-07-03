#!/bin/bash

train_eval () {
    if [ $(($3)) -ne 0 ]; then
        for ((i = 0; $i < $1; i += $2));
        do 
            if [ $i -eq 0 ]; then
                CKPT_OPTION=""
            else
                CKPT_OPTION="--ckpt_path=$TRAINING_LOGDIR/model_$i.ckpt"
            fi
            python train.py --logdir=$TRAINING_LOGDIR --dataset_path=$TRAINING_DATASET_PATH --dataset_info_path=$TRAINING_DATASET_INFO_PATH --num_epochs=$2 $CKPT_OPTION --epochs_per_save=$2 --steps_per_log=$STEPS_PER_LOG --learning_rate=$LEARNING_RATE $USE_LR_DECAY_FLAG $STAIRCASE_LR_DECAY_FLAG --lr_decay_rate=$LR_DECAY_RATE --lr_decay_epochs=$LR_DECAY_EPOCHS --model=$MODEL --optimizer=$OPTIMIZER --batch_size=$BATCH_SIZE --shuffle_buffer_size=$SHUFFLE_BUFFER_SIZE $USE_MC $MC_INDEPENDENT
            ret=$?
            if [ $ret -ne 0 ]; then
                break
            fi
        
            python evaluate.py --logdir=$EVAL_LOGDIR --dataset_path=$TESTING_DATASET_PATH --dataset_info_path=$TESTING_DATASET_INFO_PATH --ckpt_path=$TRAINING_LOGDIR --model=$MODEL $USE_MC
            ret=$?
            if [ $ret -ne 0 ]; then
                break
            fi
        done
    else
        python train.py --logdir=$TRAINING_LOGDIR --dataset_path=$TRAINING_DATASET_PATH --dataset_info_path=$TRAINING_DATASET_INFO_PATH --num_epochs=$1 $CKPT_OPTION --epochs_per_save=$2 --steps_per_log=$STEPS_PER_LOG --learning_rate=$LEARNING_RATE $USE_LR_DECAY_FLAG $STAIRCASE_LR_DECAY_FLAG --lr_decay_rate=$LR_DECAY_RATE --lr_decay_epochs=$LR_DECAY_EPOCHS --model=$MODEL --optimizer=$OPTIMIZER --batch_size=$BATCH_SIZE --shuffle_buffer_size=$SHUFFLE_BUFFER_SIZE $USE_MC $MC_INDEPENDENT
    fi
}
