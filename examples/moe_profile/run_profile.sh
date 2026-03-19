#!/bin/bash
# Profile the dispatch / compute / combine latency of MCore MoELayer.
#
# Usage:
#   bash examples/moe_profile/run_profile.sh
#
# All knobs can be overridden via environment variables, e.g.:
#   NUM_TOKENS=8192 NUM_EXPERTS=32 TOPK=4 bash examples/moe_profile/run_profile.sh

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
EXPR_HOME=${EXPR_HOME:-/scratch/Megatron-LM}
cd "${EXPR_HOME}"

# ---------- GPU / distributed topology ----------
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6100"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}

# ---------- Model / MoE shape ----------
NUM_TOKENS=2048
HIDDEN_SIZE=2048
MOE_FFN_HIDDEN_SIZE=2816
NUM_EXPERTS=64
TOPK=6
NUM_ATTENTION_HEADS=32

# ---------- Parallelism ----------
EP_SIZE=${GPUS_PER_NODE}
TP_SIZE=1
DISPATCHER="alltoall"
GROUPED_GEMM=1

# ---------- Benchmark control ----------
WARMUP_ITERS=${WARMUP_ITERS:-200}
ADAPTIVE_WARMUP=${ADAPTIVE_WARMUP:-1}
MEASURE_ITERS=${MEASURE_ITERS:-300}
DTYPE=${DTYPE:-"bf16"}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}}

# ---------- Build CLI ----------
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

PROFILE_ARGS=(
    --num-tokens "${NUM_TOKENS}"
    --hidden-size "${HIDDEN_SIZE}"
    --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
    --num-experts "${NUM_EXPERTS}"
    --topk "${TOPK}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --ep-size "${EP_SIZE}"
    --tp-size "${TP_SIZE}"
    --dispatcher "${DISPATCHER}"
    --warmup-iters "${WARMUP_ITERS}"
    --measure-iters "${MEASURE_ITERS}"
    --dtype "${DTYPE}"
    --output-dir "${OUTPUT_DIR}"
)

if [ "${GROUPED_GEMM}" = "1" ]; then
    PROFILE_ARGS+=(--grouped-gemm)
else
    PROFILE_ARGS+=(--no-grouped-gemm)
fi

if [ "${ADAPTIVE_WARMUP}" = "1" ]; then
    PROFILE_ARGS+=(--adaptive-warmup)
fi

echo "=== MoE Layer Latency Profiler ==="
echo "Tokens/rank : ${NUM_TOKENS}"
echo "Hidden      : ${HIDDEN_SIZE}"
echo "FFN hidden  : ${MOE_FFN_HIDDEN_SIZE}"
echo "Experts     : ${NUM_EXPERTS}"
echo "Top-k       : ${TOPK}"
echo "EP size     : ${EP_SIZE}"
echo "TP size     : ${TP_SIZE}"
echo "Dispatcher  : ${DISPATCHER}"
echo "Grouped GEMM: ${GROUPED_GEMM}"
echo "Dtype       : ${DTYPE}"
echo "Warmup      : ${WARMUP_ITERS} (adaptive=${ADAPTIVE_WARMUP})"
echo "Measure     : ${MEASURE_ITERS}"
echo "Output      : ${OUTPUT_DIR}"
echo "=================================="

torchrun "${DISTRIBUTED_ARGS[@]}" \
    examples/moe_profile/profile_moe_latency.py \
    "${PROFILE_ARGS[@]}"
