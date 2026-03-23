# DeepSeek-Coder-V2-Lite-Instruct Training Example

Training example for [DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct), a 16B-parameter Mixture-of-Experts model (2.4B active per token) designed for code generation and understanding.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 27 (1 dense + 26 MoE) |
| Hidden size | 2048 |
| Dense FFN hidden size | 10944 |
| MoE FFN hidden size | 1408 |
| Attention heads | 16 |
| Routed experts | 64 |
| Experts per token (top-k) | 6 |
| Shared experts | 2 |
| Vocab size | 102400 |
| Attention | Multi-Latent Attention (MLA) |
| KV LoRA rank | 512 |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Positional encoding | RoPE with YaRN scaling |

## Parallelism Strategy

The default configuration uses 8 GPUs with:

- **Expert Parallel (EP) = 8**: Distributes 64 experts across 8 GPUs (8 experts/GPU)
- **Tensor Parallel (TP) = 1**: Hidden size 2048 is small enough for single-GPU dense layers
- **Pipeline Parallel (PP) = 1**
- **Sequence Parallel**: Enabled
- **Distributed Optimizer**: Enabled with overlapped param-gather and grad-reduce

## Data Preparation

Prepare the training data in Megatron format. Example using BookCorpus:

```bash
python tools/preprocess_data.py \
    --input <path-to-jsonl> \
    --output-prefix /scratch/data_store/bookcorpus_megatron/bookcorpus \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --trust-remote-code \
    --workers 32 \
    --append-eod
```

## Training

```bash
bash examples/deepseek_coder_v2_lite/train_deepseek_coder_v2_lite_distributed.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPUS_PER_NODE` | 8 | Number of GPUs per node |
| `NNODES` / `SLURM_NNODES` | 1 | Number of nodes |
| `MASTER_ADDR` | localhost | Master node address |
| `MASTER_PORT` | 6000 | Master node port |
| `TOKENIZER_MODEL` | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | HuggingFace tokenizer |
| `ENABLE_NSYS_PROFILE` | 0 | Enable Nsight Systems profiling |
| `WANDB_API_KEY` | (unset) | Set to enable W&B logging |

### Multi-Node Training

For multi-node SLURM jobs:

```bash
srun bash examples/deepseek_coder_v2_lite/train_deepseek_coder_v2_lite_distributed.sh
```
