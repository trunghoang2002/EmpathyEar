#! /usr/bin/env bash

set -ex

LR=1e-4
#NUM_GPUS=1
#LORA_RANK=8
#LORA_ALPHA=32
#LORA_DROUPOUT=0.1

MAX_SOURCE_LEN=2048
MAX_TARGET_LEN=256
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=1
MAX_STEP=5000
SAVE_INTERVAL=500
MAX_SEQ_LEN=2048

RUN_NAME=ECAC_full
BASE_MODEL_PATH=/mnt/haofei/MSA/ChatGLM3/chatglm3-6b-base
DATASET_PATH=/mnt/haofei/MSA/flan-T5/data/ChatGLM-ECAC/ECAC-prompt/train.json
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
#OUTPUT_DIR=/mnt/haofei/MSA/ChatGLM3/finetune_basemodel_demo/chatglm3-6b-finetuned
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

#torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
python finetune.py \
      --train_format input-output \
      --train_file $DATASET_PATH \
      --max_seq_length $MAX_SEQ_LEN \
      --preprocessing_num_workers 1 \
      --model_name_or_path $BASE_MODEL_PATH \
      --output_dir $OUTPUT_DIR \
      --per_device_train_batch_size $DEV_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
      --max_steps $MAX_STEP \
      --logging_steps 50 \
      --save_steps $SAVE_INTERVAL \
      --learning_rate $LR 2>&1 | tee ${OUTPUT_DIR}/train.log
