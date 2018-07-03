#!/bin/bash

source train_eval.sh

TRAINING_LOGDIR=logdir/srcnn_batch_32_lr_1e-3_decay_adam_std/train
EVAL_LOGDIR=logdir/srcnn_batch_32_lr_1e-3_decay_adam_std/test
TRAINING_DATASET_PATH=datasets/train_std/dataset.tfrecords
TRAINING_DATASET_INFO_PATH=datasets/train_std/dataset_info.txt
TESTING_DATASET_PATH=datasets/test/dataset.tfrecords
TESTING_DATASET_INFO_PATH=datasets/test/dataset_info.txt

MODEL=srcnn
BATCH_SIZE=32
OPTIMIZER=adam
LEARNING_RATE=1e-3
USE_LR_DECAY_FLAG=--use_lr_decay
LR_DECAY_RATE=0.1
LR_DECAY_EPOCHS=30
STAIRCASE_LR_DECAY_FLAG=--staircase_lr_decay
STEPS_PER_LOG=1000
NUM_EPOCHS=100
EPOCHS_PER_EVAL=1
SHUFFLE_BUFFER_SIZE=100000

train_eval $NUM_EPOCHS $EPOCHS_PER_EVAL
