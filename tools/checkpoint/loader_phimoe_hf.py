#!/usr/bin/env python

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import types

import torch
import transformers
from tqdm import tqdm

from tools.checkpoint.utils import _ConverterFakeProcessGroup


def add_arguments(parser):
    group = parser.add_argument_group(title='PhiMoE HF loader.')

    group.add_argument(
        '--true-vocab-size',
        type=int,
        default=None,
        help='original size of vocab, if specified will trim padding from embedding table.',
    )
    group.add_argument(
        '--vocab-file',
        type=str,
        default=None,
        help='Path to the vocab file. If specified will use this to get vocab size and trim '
        'padding from the embedding table.',
    )
    group.add_argument(
        '--tokenizer-model',
        required=True,
        help='Hugging Face tokenizer path or repo id.',
    )
    group.add_argument(
        '--megatron-path',
        type=str,
        default=None,
        help='Base directory of Megatron repository',
    )


def load_args_from_checkpoint(args):
    from transformers import AutoConfig

    phimoe_config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
    assert phimoe_config.model_type == "phimoe", phimoe_config.model_type

    args.untie_embeddings_and_output_weights = not phimoe_config.tie_word_embeddings
    args.seq_length = phimoe_config.max_position_embeddings
    args.global_batch_size = 1024
    args.iteration = 1  # '0' and 'release' do not work with converter metadata.
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.position_embedding_type = "rope"
    args.swiglu = True
    args.bf16 = True
    args.add_bias_linear = False
    args.add_qkv_bias = True
    args.disable_bias_linear = True
    args.normalization = "LayerNorm"
    args.tokenizer_type = "HuggingFaceTokenizer"
    args.trust_remote_code = True

    args.max_position_embeddings = phimoe_config.max_position_embeddings
    args.hidden_size = phimoe_config.hidden_size
    args.num_attention_heads = phimoe_config.num_attention_heads
    args.num_layers = phimoe_config.num_hidden_layers
    args.norm_epsilon = phimoe_config.rms_norm_eps
    args.vocab_size = phimoe_config.vocab_size
    args.padded_vocab_size = phimoe_config.vocab_size
    args.ffn_hidden_size = phimoe_config.intermediate_size
    args.moe_ffn_hidden_size = phimoe_config.intermediate_size
    args.num_experts = phimoe_config.num_local_experts
    args.moe_router_topk = phimoe_config.num_experts_per_tok
    args.moe_router_load_balancing_type = "none"
    args.moe_aux_loss_coeff = 0.0
    args.moe_input_jitter_eps = phimoe_config.input_jitter_noise
    args.kv_channels = phimoe_config.head_dim
    args.rotary_base = phimoe_config.rope_theta
    args.expert_tensor_parallel_size = 1
    args.sequence_parallel = args.tensor_model_parallel_size > 1

    if phimoe_config.num_key_value_heads:
        args.group_query_attention = True
        args.num_query_groups = phimoe_config.num_key_value_heads

    if getattr(phimoe_config, "sliding_window", None) is not None:
        args.window_size = (phimoe_config.sliding_window, 0)


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major > 4 or (major == 4 and minor >= 43)


def set_preprocess_state(args, model, hf_model):
    """Set embedding params."""
    model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    """Set output layer and final norm params."""
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    model.decoder.final_layernorm.bias.data.copy_(hf_model.model.norm.bias)
    model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    # Current GPT checkpoint converter path does not persist lm_head.bias.


def set_attn_state(args, layer, hf_layer):
    """Set self-attention params."""
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    tp = args.tensor_model_parallel_size
    num_heads = args.num_attention_heads // tp
    num_query_groups = (
        args.num_query_groups if args.group_query_attention else args.num_attention_heads
    ) // tp
    num_queries_per_group = num_heads // num_query_groups
    dim = args.kv_channels
    assert num_heads % num_queries_per_group == 0

    attn.linear_qkv.weight.data.copy_(
        torch.cat(
            [
                hf_attn.q_proj.weight.reshape((num_query_groups, num_queries_per_group * dim, -1)),
                hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
                hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, args.hidden_size))
    )
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

    if attn.linear_qkv.bias is not None:
        attn.linear_qkv.bias.data.copy_(
            torch.cat([hf_attn.q_proj.bias, hf_attn.k_proj.bias, hf_attn.v_proj.bias], dim=0)
        )
    # Megatron MoE checkpoint conversion does not currently preserve output projection bias.


def set_mlp_state(args, layer, hf_layer):
    """Set MoE router and expert params."""
    layer.mlp.router.weight.data.copy_(hf_layer.block_sparse_moe.gate.weight)

    mcore_experts = layer.mlp.experts.local_experts
    hf_experts = hf_layer.block_sparse_moe.experts
    for expert_idx in range(args.num_experts):
        mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
            torch.cat([hf_experts[expert_idx].w1.weight, hf_experts[expert_idx].w3.weight], dim=0)
        )
        mcore_experts[expert_idx].linear_fc2.weight.data.copy_(hf_experts[expert_idx].w2.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer params."""
    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)

    layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.self_attention.linear_qkv.layer_norm_bias.data.copy_(hf_layer.input_layernorm.bias)
    layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    layer.pre_mlp_layernorm.bias.data.copy_(hf_layer.post_attention_layernorm.bias)


def load_checkpoint_to_model(args):
    """Load HF PhiMoE checkpoint into an MCore GPT model."""
    from model_provider import model_provider
    from gpt_builders import gpt_builder
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.load, device_map="cpu", torch_dtype="auto", trust_remote_code=True
    )

    model = model_provider(gpt_builder, pre_process=True, post_process=True).to(args.params_dtype)

    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)
    return model


def _load_checkpoint(queue, args):
    verify_transformers_version()

    sys.path.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
        )
    )
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print(
            "Unable to import Megatron, please specify the path to Megatron using "
            "--megatron-path. Exiting."
        )
        queue.put("exit")
        exit(1)

    sys.argv = [
        'script.py',
        '--use-mcore-models',
        '--disable-bias-linear',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size',
        '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--mock-data',
        '--transformer-impl',
        'transformer_engine',
        '--load',
        args.load_dir,
        '--no-one-logger',
    ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('disable_bias_linear')
    check_for_arg('params_dtype')
    check_for_arg('swiglu')

    assert args.model_type == 'GPT', 'PhiMoE is a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder

    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(
        margs.virtual_pipeline_model_parallel_size
    )
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)

    fake_tp_group = _ConverterFakeProcessGroup(size=margs.tensor_model_parallel_size)
    fake_ep_group = _ConverterFakeProcessGroup(size=margs.expert_model_parallel_size)
    mpu._TENSOR_MODEL_PARALLEL_GROUP = fake_tp_group
    mpu._EXPERT_MODEL_PARALLEL_GROUP = fake_ep_group
    fused_kernels.load(margs)

    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.qkv_bias = margs.add_qkv_bias
    md.norm_has_bias = margs.normalization == "LayerNorm"
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.num_experts = margs.num_experts

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(margs.tokenizer_model, trust_remote_code=True)
    md.true_vocab_size = len(tokenizer)

    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    message = {"word embeddings": model.embedding.word_embeddings.weight.data}
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.embedding, 'position_embeddings')
    queue_put("embeddings", message)

    for layer_idx in range(margs.num_layers):
        layer = model.decoder.layers[layer_idx]
        experts = layer.mlp.experts.local_experts
        message = {
            "input norm weight": layer.self_attention.linear_qkv.layer_norm_weight.data,
            "post norm weight": layer.pre_mlp_layernorm.weight.data,
            "qkv weight": layer.self_attention.linear_qkv.weight.data,
            "dense weight": layer.self_attention.linear_proj.weight.data,
            "router weight": layer.mlp.router.weight.data,
            "mlp l1 weight": torch.stack(
                [local_expert.linear_fc2.weight.data for local_expert in experts], dim=0
            ),
        }

        if md.norm_has_bias:
            message["input norm bias"] = layer.self_attention.linear_qkv.layer_norm_bias.data
            message["post norm bias"] = layer.pre_mlp_layernorm.bias.data
        if md.qkv_bias:
            message["qkv bias"] = layer.self_attention.linear_qkv.bias.data
        if md.swiglu:
            chunked_mlp_l0_weight = [
                torch.chunk(local_expert.linear_fc1.weight.data, 2, dim=0)
                for local_expert in experts
            ]
            message["mlp l0 weight W"] = torch.stack(
                [local_weight[0] for local_weight in chunked_mlp_l0_weight], dim=0
            )
            message["mlp l0 weight V"] = torch.stack(
                [local_weight[1] for local_weight in chunked_mlp_l0_weight], dim=0
            )
        else:
            message["mlp l0 weight"] = torch.stack(
                [local_expert.linear_fc1.weight.data for local_expert in experts], dim=0
            )

        queue_put(f"transformer layer {layer_idx}", message)

    final_norm_message = {"weight": model.decoder.final_layernorm.weight.data}
    if md.norm_has_bias:
        final_norm_message["bias"] = model.decoder.final_layernorm.bias.data
    queue_put("final norm", final_norm_message)

    if md.output_layer:
        queue_put("output layer", {"weight": model.output_layer.weight.data})

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
