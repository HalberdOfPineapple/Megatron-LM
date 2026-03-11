#!/bin/bash

set -euo pipefail

HF_CHECKPOINT_PATH=$1
MEGATRON_CHECKPOINT_PATH=$2
TOKENIZER_MODEL=${3:-microsoft/Phi-mini-MoE-instruct}
TARGET_TP_SIZE=${TARGET_TP_SIZE:-1}
TARGET_PP_SIZE=${TARGET_PP_SIZE:-1}
TARGET_EP_SIZE=${TARGET_EP_SIZE:-8}
MEGATRON_PATH=${MEGATRON_PATH:-$(pwd)}

export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader loader_phimoe_hf \
    --saver core \
    --target-tensor-parallel-size ${TARGET_TP_SIZE} \
    --target-pipeline-parallel-size ${TARGET_PP_SIZE} \
    --target-expert-parallel-size ${TARGET_EP_SIZE} \
    --load-dir ${HF_CHECKPOINT_PATH} \
    --save-dir ${MEGATRON_CHECKPOINT_PATH} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --megatron-path ${MEGATRON_PATH}
