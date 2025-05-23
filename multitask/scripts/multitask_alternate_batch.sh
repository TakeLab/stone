#!/bin/bash

LR=0.00001
MODEL="classla/bcms-bertic"
DO_LOWER_CASE=False
DATASET_TRAIN_PATH="../data/processed/train_gold.csv"
DATASET_VALID_PATH="../data/processed/valid_gold.csv"
DATASET_TEST_PATH="../data/processed/test_gold.csv"
EPOCHS=10
BATCH_SIZE=16
NAME="bertic_multitask_sentiment_tone_alternate_batch_update"
SEED=42
DEVICE="cuda:1"
MULTITASK_MODE="alternate_batch"
EVAL_MODE="macro"
SCHEDULER_WARMUP_STEPS=0

python ../baselines/multitask_training.py \
	--lr $LR \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_train_path "$DATASET_TRAIN_PATH" \
	--dataset_valid_path "$DATASET_VALID_PATH" \
	--dataset_test_path "$DATASET_TEST_PATH" \
	--epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
	--name "$NAME" \
	--seed $SEED \
	--device "$DEVICE" \
    --multitask_mode "$MULTITASK_MODE" \
	--eval_mode "$EVAL_MODE" \
    --scheduler_warmup_steps $SCHEDULER_WARMUP_STEPS