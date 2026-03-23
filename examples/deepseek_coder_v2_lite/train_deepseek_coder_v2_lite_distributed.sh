#!/bin/bash

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1


EXPR_HOME=/scratch/Megatron-LM
cd ${EXPR_HOME}

EXPR_DATA_DIR=/blob/megatron_data_store/dsv2_coder_lite
mkdir -p ${EXPR_DATA_DIR}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
ENABLE_NSYS_PROFILE=${ENABLE_NSYS_PROFILE:-0}

CHECKPOINT_PATH=${EXPR_DATA_DIR}/checkpoints
mkdir -p ${CHECKPOINT_PATH}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct}
DATA_PATH=/scratch/data_store/bookcorpus_megatron/bookcorpus_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node ${GPUS_PER_NODE}
    --nnodes ${NNODES}
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

# ============================================================================
# DeepSeek-Coder-V2-Lite-Instruct model architecture
#   - 27 layers: 1 dense + 26 MoE
#   - 16B total parameters (2.4B active per token)
#   - MLA (Multi-Latent Attention) with KV LoRA rank 512
#   - 64 routed experts, top-6 routing, 2 shared experts
#   - SwiGLU activation, RMSNorm, YaRN rope scaling
# ============================================================================

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 27
    --hidden-size 2048
    --ffn-hidden-size 10944
    --num-attention-heads 16
    --kv-channels 16
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-percent 1.0
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --vocab-size 102400
    --make-vocab-size-divisible-by 3200
    --attention-softmax-in-fp32
)

# Multi-Latent Attention (MLA) configuration
MLA_ARGS=(
    --multi-latent-attention
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --rotary-scaling-factor 40
    --mscale 0.707
    --mscale-all-dim 0.707
)

# MoE configuration: 64 routed experts (top-6) + 2 shared experts
# First layer is dense, remaining 26 layers are MoE
MOE_ARGS=(
    --num-experts 64
    --moe-layer-freq '([0]+[1]*26)'
    --moe-ffn-hidden-size 1408
    --moe-shared-expert-intermediate-size 2816
    --moe-router-topk 6
    --moe-router-topk-scaling-factor 1.0
    --moe-router-score-function softmax
    --moe-router-pre-softmax
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type alltoall
    --moe-token-drop-policy probs
    --moe-grouped-gemm
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

PROFILE_CMD=()
if [ "${ENABLE_NSYS_PROFILE}" = "1" ]; then
    NSYS_OUTPUT_PREFIX=${EXPR_DATA_DIR}/train_deepseek_coder_v2_lite_nsys
    PROFILE_CMD=(
        nsys profile
        --force-overwrite true
        --trace cuda,nvtx
        -x true
        -o "${NSYS_OUTPUT_PREFIX}"
    )
    TRAINING_ARGS+=(
        --profile
        --profile-step-start 5
        --profile-step-end 20
        --profile-ranks 0
    )
fi

# Parallelism: TP=1, PP=1, EP=8
# With hidden_size=2048, TP=1 is sufficient for the dense layers.
# EP=8 distributes 64 experts across 8 GPUs (8 experts per GPU).
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size ${GPUS_PER_NODE}
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
        --wandb-project ${WANDB_PROJECT:-"DeepSeekCoderV2Lite"}
        --wandb-exp-name ${WANDB_NAME:-"DeepSeek-Coder-V2-Lite-Instruct"}
    )
fi

LOG_PATH=${EXPR_DATA_DIR}/train_deepseek_coder_v2_lite_distributed.log
echo "Logging to ${LOG_PATH}"


"${PROFILE_CMD[@]}" torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} > $LOG_PATH 2>&1

echo "Save log to ${LOG_PATH}"
if [ "${ENABLE_NSYS_PROFILE}" = "1" ]; then
    echo "Nsight Systems enabled, writing traces to ${EXPR_DATA_DIR}"
fi
