#!/bin/bash

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

source ~/.profile

SCRIPT_DIR=$(dirname $(realpath $0))
export CUDA_HOME=/usr/local/cuda

EXPR_HOME=/scratch/Megatron-LM
cd ${EXPR_HOME}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}

CHECKPOINT_PATH=${SCRIPT_DIR}/checkpoints
mkdir -p ${CHECKPOINT_PATH}
TOKENIZER_MODEL=${2:-microsoft/Phi-mini-MoE-instruct}
DATA_PATH=/scratch/data_store/bookcorpus_megatron/bookcorpus_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node ${GPUS_PER_NODE}
    --nnodes ${NNODES}
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --attention-backend flash
    --disable-bias-linear
    --add-qkv-bias
    --seq-length 4096
    --max-position-embeddings 4096
    --window-size 2047,0
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 960
    --moe-ffn-hidden-size 960
    --num-attention-heads 32
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization LayerNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
    --norm-epsilon 1e-5
    --vocab-size 32064
    --make-vocab-size-divisible-by 64
)

MOE_ARGS=(
    --num-experts 16
    --moe-layer-freq 1
    --moe-router-topk 2
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --moe-input-jitter-eps 0.01
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --trust-remote-code
    --data-path ${DATA_PATH}
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1e-5
    --train-iters 20
    --lr-decay-iters 10000
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.1
    --lr-warmup-iters 100
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $GPUS_PER_NODE
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size $GPUS_PER_NODE
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1000
    --eval-interval 1000
    --eval-iters 20
    --save ${CHECKPOINT_PATH}
    --load ${CHECKPOINT_PATH}
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
    --no-load-optim
    --no-load-rng
)

if [ -n "${WANDB_API_KEY:-}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"PhiMoE"}
        --wandb-exp-name ${WANDB_NAME:-"Phi-mini-MoE"}
    )
fi


LOG_PATH=${SCRIPT_DIR}/train_phimoe_distributed.log
/blob/utils/kill_nv.sh 1 true

echo "Logging to ${LOG_PATH}"
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} > ${LOG_PATH} 2>&1

/blob/utils/kill_nv.sh 1
echo "Log saved to ${LOG_PATH}"